"""Run the historical frozen-evidence workflow with supplementary I3 cells."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from mas_faults.cross_benchmark_main_matrix import build_execution_schedule
from mas_faults.cross_benchmark_main_runner import _execute, _read_jsonl, order_schedule, summarize_rows, partition_schedule, select_repeat_indices
from mas_faults.i3_supplement import reconstruct_pair
from mas_faults.llm_client import get_llm_client
from mas_faults.deepseek_schedule import ensure_deepseek_offpeak
from deepseek_inference_policy import settings

CONDITIONS = ['clean','non_delivery_step3','semantic_corruption_step3','non_delivery_step2','non_delivery_step4','semantic_corruption_step4']
DOMAINS = {'reddit':'WebArena Reddit','swe':'SWE-bench Verified','tac':'TheAgentCompany'}


def audit(rows, schedule, required_model='deepseek-v4-flash'):
    errors=[]
    jobs={j.run_id:j for j in schedule}
    if not schedule or len(jobs)!=len(schedule):
        errors.append('empty or duplicate schedule')
    if len(rows)!=len(schedule) or {r['run_id'] for r in rows}!={j.run_id for j in schedule}:
        errors.append('incomplete or duplicate schedule')
    for row in rows:
        job=jobs.get(row['run_id'])
        if (job is None or row.get('model')!=required_model
                or any(row.get(k)!=getattr(job,k) for k in ['benchmark','task_id','topology','condition','repeat_index'])):
            errors.append(row['run_id']+': identity mismatch');continue
        applied=[e for e in row['events'] if e.get('fault_applied')]
        step4=[e for e in row['events'] if e.get('abstract_step')==4]
        if len(step4)!=(2 if job.topology=='flat' else 1):
            errors.append(row['run_id']+': missing or duplicate downstream edge')
        cond=row['condition']
        if cond=='clean':
            if applied:errors.append(row['run_id']+': clean injected')
            continue
        step=int(cond[-1])
        if len(applied)!=1 or applied[0]['abstract_step']!=step:
            errors.append(row['run_id']+': wrong exposure');continue
        e=applied[0]
        if cond.startswith('non_delivery') and e['delivery_count']!=0:
            errors.append(row['run_id']+': drop delivered')
        if cond.startswith('semantic_corruption'):
            if e['delivery_count']!=1 or e['original_message']==e['delivered_messages'][0]:
                errors.append(row['run_id']+': corruption ineffective')
        if step==3:
            for downstream in [x for x in row['events'] if x['abstract_step']==4]:
                if downstream['delivered_messages']!=e['delivered_messages']:
                    errors.append(row['run_id']+': inconsistent branch input')
    return {'passed':not errors,'rows':len(rows),'expected':len(schedule),'errors':errors,
            'model_output_error_count':sum(bool(r.get('error')) for r in rows)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--domain',choices=DOMAINS,required=True);p.add_argument('--phase',choices=['pilot','formal'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--required-model',default='deepseek-v4-flash')
    p.add_argument('--repeat-indices',type=int,nargs='+');p.add_argument('--shard-count',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    os.environ['MAS_DEEPSEEK_OFFPEAK_ONLY']='1'
    refs=[r for r in json.loads(a.reference.read_text()) if r['benchmark']==DOMAINS[a.domain]]
    refs=sorted(refs,key=lambda r:r['task_id'])
    if len(refs)!={'reddit':15,'swe':20,'tac':15}[a.domain] or len({r['task_id'] for r in refs})!=len(refs):
        raise ValueError('historical domain task coverage mismatch')
    pairs={r['task_id']:reconstruct_pair(r) for r in refs}
    conditions=CONDITIONS[:3] if a.phase=='pilot' else CONDITIONS
    schedule=order_schedule(build_execution_schedule([v[0] for v in pairs.values()],conditions=conditions,repeats=3),seed=20260817)
    representatives={}
    if a.phase=='pilot':
        for task in sorted(pairs):
            for repeat in (1,2,3):
                fingerprint=json.dumps(settings(a.domain,task,repeat)['environment'],sort_keys=True)
                representatives.setdefault(fingerprint,(task,repeat))
        selected=set(representatives.values())
        schedule=[j for j in schedule if (j.task_id,j.repeat_index) in selected]
    if a.repeat_indices: schedule=select_repeat_indices(schedule,a.repeat_indices)
    schedule=partition_schedule(schedule,a.shard_count,a.shard_index)
    manifest={'selected_repeat_indices':a.repeat_indices,'shard_count':a.shard_count,'shard_index':a.shard_index,'phase':a.phase,'domain':a.domain,'model':a.required_model,'task_ids':list(pairs),'conditions':conditions,'repeats':1 if a.phase=='pilot' else 3,'schedule_seed':20260817,'jobs':len(schedule),'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted({r['clean']['source_jsonl'] for r in refs})},'historical_payload_checks':len(pairs),'mutation_source':'Exact inner payload recorded in historical semantic_corruption_step4; I3 and I4 use the same operator.'}
    manifest['repeats']=3
    manifest['pilot_policy_representatives']=list(representatives.values())
    manifest['inference_by_job']={j.run_id:settings(a.domain,j.task_id,j.repeat_index) for j in schedule}
    manifest['historical_alignment_note']='Original Reddit five-task shell-resume environment is not archived. Use explicit initial launch request settings and contemporaneous controls; do not claim exact historical resume equivalence.'
    path=a.output_dir/'manifest.json'
    source_root=Path(__file__).resolve().parent
    manifest['code_sha256']={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(list((source_root/'src').rglob('*.py'))+[Path(__file__).resolve(),source_root/'deepseek_inference_policy.py'])}
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('resume manifest mismatch')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    clients={}
    runs=a.output_dir/'main_runs.jsonl';errors=a.output_dir/'run_errors.jsonl'
    existing=_read_jsonl(runs)
    by_id={j.run_id:j for j in schedule}
    if len({r['run_id'] for r in existing})!=len(existing):raise ValueError('duplicate completed records')
    for r in existing:
        if r['run_id'] not in by_id or not audit([r],[by_id[r['run_id']]],a.required_model)['passed']:
            raise ValueError('resume record identity or exposure mismatch')
        current,_=pairs[r['task_id']]
        if r.get('source_run_id')!=current.source_row.get('run_id') or r.get('source_jsonl')!=current.source_jsonl:
            raise ValueError('resume source mismatch')
    completed={r['run_id'] for r in existing}
    async def execute():
        for j in schedule:
            if j.run_id in completed:continue
            current,stale=pairs[j.task_id]
            inference=settings(a.domain,j.task_id,j.repeat_index)
            for key,value in inference['environment'].items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
            client_key=json.dumps(inference['environment'],sort_keys=True)
            if client_key not in clients:clients[client_key]=get_llm_client()
            client=clients[client_key]
            if client.model_info.model!=a.required_model:raise ValueError('model mismatch')
            ensure_deepseek_offpeak(client.model_info.model)
            await _execute(client,schedule=[j],carriers=[current,stale],runs_path=runs,errors_path=errors,completed=completed,max_attempts=inference['max_attempts'])
            if j.run_id not in completed:raise RuntimeError('attempt limit reached: '+j.run_id)
            print(json.dumps({'completed':len(completed),'total':len(schedule),'run_id':j.run_id}),flush=True)
    asyncio.run(execute())
    rows=_read_jsonl(runs);gate=audit(rows,schedule,a.required_model)
    (a.output_dir/'injection_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    (a.output_dir/'summary.json').write_text(json.dumps(summarize_rows(rows),indent=2)+'\n')
    if not gate['passed']:raise RuntimeError('injection gate failed')
    print(json.dumps({'phase':a.phase,'domain':a.domain,'gate':gate}),flush=True)

if __name__=='__main__':main()
