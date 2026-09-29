"""27B paired batch with strict repeat-specific carriers and frozen budgets."""
import argparse,asyncio,hashlib,json,os
from pathlib import Path
from mas_faults.cross_benchmark_main_matrix import build_execution_schedule
from mas_faults.cross_benchmark_main_runner import _execute,_read_jsonl,order_schedule,summarize_rows,partition_schedule
from mas_faults.i3_supplement import reconstruct_pair
from mas_faults.llm_client import get_llm_client
from qwen27_runtime import index_references,index_policy,environment,apply_environment,client_key,pilot_representatives
CONDITIONS=['clean','non_delivery_step3','semantic_corruption_step3','non_delivery_step2','non_delivery_step4','semantic_corruption_step4']
DOMAINS={'reddit':'WebArena Reddit','swe':'SWE-bench Verified','tac':'TheAgentCompany'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(rows, schedule, required_model='Qwen3.8-27B'):
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


def prepare(reference,policy,domain,phase,shard_count=1,shard_index=0):
    refs=[r for r in json.loads(Path(reference).read_text()) if r['benchmark']==DOMAINS[domain]]
    indexed=index_references(refs);policies=index_policy(json.loads(Path(policy).read_text()))
    tasks=sorted({r['task_id'] for r in refs})
    if len(tasks)!={'reddit':15,'swe':20,'tac':15}[domain] or len(indexed)!=len(tasks)*3:raise ValueError('domain coverage mismatch')
    pairs={(task,repeat):reconstruct_pair(indexed[(DOMAINS[domain],task,repeat)]) for task in tasks for repeat in (1,2,3)}
    conditions=CONDITIONS[:3] if phase=='pilot' else CONDITIONS
    schedule=[]
    for repeat in (1,2,3):
        one=build_execution_schedule([pairs[(task,repeat)][0] for task in tasks],conditions=conditions,repeats=3)
        schedule.extend(j for j in one if j.repeat_index==repeat)
    schedule=order_schedule(schedule,seed=20260817)
    selected=pilot_representatives(domain,refs,policies) if phase=='pilot' else set(pairs)
    schedule=[j for j in schedule if (j.task_id,j.repeat_index) in selected]
    schedule=partition_schedule(schedule,shard_count,shard_index)
    if not schedule or len({j.run_id for j in schedule})!=len(schedule):raise ValueError('invalid schedule')
    return refs,pairs,policies,schedule,selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--domain',choices=DOMAINS,required=True);p.add_argument('--phase',choices=['pilot','formal'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();refs,pairs,policies,schedule,selected=prepare(a.reference,a.policy,a.domain,a.phase,a.shard_count,a.shard_index)
    root=Path(__file__).resolve().parent
    source_files=[Path(__file__),root/'qwen27_runtime.py',Path(reconstruct_pair.__code__.co_filename)]
    source_files+=list((root.parent/'src').rglob('*.py'))
    manifest={'model':'Qwen3.8-27B','domain':a.domain,'phase':a.phase,'shard_count':a.shard_count,'shard_index':a.shard_index,'jobs':[{'run_id':j.run_id,'task_id':j.task_id,'topology':j.topology,'repeat_index':j.repeat_index,'condition':j.condition} for j in schedule],'selected_pairs':sorted(selected),'reference_sha256':sha(a.reference),'policy_sha256':sha(a.policy),'code_sha256':{str(f):sha(f) for f in source_files},'source_sha256':{r['clean']['source_jsonl']:sha(r['clean']['source_jsonl']) for r in refs},'inference_by_job':{j.run_id:environment(policies,a.domain,j.task_id,j.repeat_index) for j in schedule},'attempt_policy':'one persisted attempt; terminal execution error blocks this shard; no success-selection retry','batch_scope':'new contemporaneous paired batch; historical settings serve as references'}
    a.output_dir.mkdir(parents=True,exist_ok=True);mf=a.output_dir/'manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('resume manifest mismatch')
    mf.write_text(json.dumps(manifest,indent=2)+'\n')
    if a.plan_only:print(json.dumps({'jobs':len(schedule),'plan_only':True}));return
    runs=a.output_dir/'main_runs.jsonl';errors=a.output_dir/'run_errors.jsonl';started=a.output_dir/'started.jsonl'
    existing=_read_jsonl(runs);by_id={j.run_id:j for j in schedule};completed=set()
    for row in existing:
        key=row['run_id']
        if key not in by_id or key in completed or not audit([row],[by_id[key]])['passed']:raise ValueError('resume identity/exposure mismatch')
        current,_=pairs[(row['task_id'],row['repeat_index'])]
        if row.get('source_jsonl')!=current.source_jsonl or row.get('source_run_id')!=current.source_row.get('run_id'):raise ValueError('resume repeat source mismatch')
        completed.add(key)
    if {r['run_id'] for r in _read_jsonl(started)}-completed:raise ValueError('unfinished persisted attempt requires audit, refusing rerun')
    clients={}
    async def execute():
        for j in schedule:
            if j.run_id in completed:continue
            env=environment(policies,a.domain,j.task_id,j.repeat_index);apply_environment(env);key=client_key(env)
            if key not in clients:clients[key]=get_llm_client()
            client=clients[key]
            if client.model_info.model!='Qwen3.8-27B':raise ValueError('model mismatch')
            current,stale=pairs[(j.task_id,j.repeat_index)]
            with started.open('a') as h:h.write(json.dumps({'run_id':j.run_id,'attempt':1,'environment':env})+'\n');h.flush();os.fsync(h.fileno())
            await _execute(client,schedule=[j],carriers=[current,stale],runs_path=runs,errors_path=errors,completed=completed,max_attempts=1)
            if j.run_id not in completed:raise RuntimeError('terminal execution error retained: '+j.run_id)
            print(json.dumps({'completed':len(completed),'expected':len(schedule),'run_id':j.run_id}),flush=True)
    asyncio.run(execute());rows=_read_jsonl(runs);gate=audit(rows,schedule)
    (a.output_dir/'injection_gate.json').write_text(json.dumps(gate,indent=2)+'\n');(a.output_dir/'summary.json').write_text(json.dumps(summarize_rows(rows),indent=2)+'\n')
    if not gate['passed']:raise RuntimeError('injection gate failed')
    print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
