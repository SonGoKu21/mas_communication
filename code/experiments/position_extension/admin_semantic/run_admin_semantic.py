"""Isolated Admin matched-table experiment. One persisted attempt per job."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from admin_semantic import CONDITIONS,TASK_ORDER,TOPOLOGIES,digest,validate_entry


def repeat_policy(model,repeat):
    if model not in ('deepseek-v4-flash','Qwen3.5-9B') or repeat not in (1,2,3):
        raise ValueError('unsupported historical model/repeat policy')
    return {'max_steps':12 if repeat==1 else 16,'request_timeout':120 if repeat==1 else 60,
            'total_timeout':120 if repeat==1 else 180,'schedule_seed':20260815 if repeat==1 else 20260824,
            'disable_thinking':1,'max_tokens':768 if model=='Qwen3.5-9B' and repeat==1 else None}


def build_schedule(manifest,*,phase,topologies=TOPOLOGIES):
    if phase not in ('pilot','formal'):raise ValueError('unknown phase')
    if not topologies or set(topologies)-set(TOPOLOGIES):raise ValueError('invalid topology')
    task_ids=(4,107) if phase=='pilot' else TASK_ORDER
    repeats=(1,) if phase=='pilot' else (1,2,3)
    jobs=[]
    for task in task_ids:
        for topo in topologies:
            for repeat in repeats:
                matches=[e for e in manifest['entries'] if e['target']['task_id']==str(task) and e['target']['topology']==topo and e['target']['repeat_index']==repeat]
                if len(matches)!=1:raise ValueError('donor entry must resolve uniquely')
                entry=matches[0];validate_entry(entry)
                for condition in CONDITIONS:
                    jobs.append({'job_key':f'admin-table-{task}-{topo}-{condition}-r{repeat}','task_id':str(task),'topology':topo,'repeat_index':repeat,'condition':condition,'entry':entry})
    ordered=[]
    for repeat in repeats:
        group=[job for job in jobs if job['repeat_index']==repeat]
        random.Random(20260815 if repeat==1 else 20260824).shuffle(group)
        ordered.extend(group)
    return ordered


def audit_row(row,entry,condition):
    validate_entry(entry)
    errors=[];events=row.get('events',[])
    opportunities=[e for e in events if e.get('fault_parameters',{}).get('injection_opportunity_consumed')]
    applied=[e for e in events if e.get('fault_applied')]
    if len(opportunities)>1 or len(applied)>1:errors.append('multiple_exposure_opportunities')
    if applied and not opportunities:errors.append('applied_without_opportunity_provenance')
    exposure='applied' if applied else ('ineffective' if opportunities else 'not_exposed')
    for event in opportunities:
        if event.get('abstract_step')!=CONDITIONS[condition]:errors.append('wrong_injection_step')
        params=event['fault_parameters']
        for k in ('entry_sha256','source_sha256','raw_table_sha256','target_projected_sha256'):
            if params.get(k)!=entry[k]:errors.append('donor_hash_mismatch:'+k)
        if params.get('donor_run_id')!=entry['donor']['run_id']:errors.append('donor_run_mismatch')
        original=deepcopy(event.get('original_message'));delivered=event.get('delivered_messages')
        if not isinstance(original,dict) or not isinstance(delivered,list) or len(delivered)!=1:
            errors.append('invalid_delivery_shape');continue
        expected=deepcopy(original)
        if CONDITIONS[condition]==3:expected['visible_evidence']=entry['raw_table']
        else:expected['payload']['visible_evidence']=entry['target_projected_table']
        if delivered[0]!=expected:errors.append('replacement_or_unchanged_outer_field_mismatch')
        if bool(event.get('fault_applied'))!=(expected!=original):errors.append('wrong_applied_vs_noop_classification')
    for event in events:
        if event.get('abstract_step')==2 and event.get('source_agent')=='Tool Navigator' and event.get('target_agent')=='WebArena Tool Worker':
            if event.get('delivered_messages')!=[event.get('original_message')]:errors.append('task_request_changed')
        if event.get('abstract_step')==1:
            task=(event.get('original_message') or {}).get('task',{})
            if any(k in task for k in ('eval','expected_answer','reference_answers')):errors.append('reference_answer_in_agent_task')
    if row.get('topology')=='flat':
        handoffs=[e for e in events if e.get('abstract_step')==4 and e.get('source_agent')=='Evidence Worker']
        if row.get('error') in (None,'') and len(handoffs)!=2:errors.append('missing_flat_handoff')
        if len(handoffs)==2:
            if CONDITIONS[condition]==3 and handoffs[0].get('delivered_messages')!=handoffs[1].get('delivered_messages'):
                errors.append('i3_flat_branches_diverge')
            if CONDITIONS[condition]==4 and handoffs[1].get('delivered_messages')!=[handoffs[0].get('original_message')]:
                errors.append('i4_flat_direct_branch_changed')
    return {'exposure':exposure,'errors':errors,'opportunities':len(opportunities),'applied_events':len(applied)}


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _append(path,row):
    with path.open('a') as handle:
        handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())


async def execute_jobs(jobs,execute_one,out,*,required_model='deepseek-v4-flash',before_job=None):
    out.mkdir(parents=True,exist_ok=True)
    by_key={j['job_key']:j for j in jobs}
    if len(by_key)!=len(jobs):raise ValueError('duplicate schedule key')
    rows=_read(out/'main_runs.jsonl');completed=set()
    for row in rows:
        key=row['job_key']
        if key not in by_key or key in completed:raise ValueError('unknown or duplicate resumed row')
        job=by_key[key]
        if any(row.get(k)!=job[k] for k in ('task_id','topology','repeat_index','condition')) or row.get('model')!=required_model or row.get('donor_entry_sha256')!=job['entry']['entry_sha256']:
            raise ValueError('resume row identity mismatch')
        check=audit_row(row,job['entry'],job['condition'])
        if check!=row.get('table_audit') or check['exposure']!=row.get('table_exposure'):
            raise ValueError('resume audit mismatch')
        completed.add(key)
    started={r['job_key'] for r in _read(out/'started.jsonl')}
    if started-completed:raise ValueError('in-flight attempt requires audit before resuming; never silently rerun')
    for job in jobs:
        if job['job_key'] in completed:continue
        if before_job is not None:before_job(job)
        _append(out/'started.jsonl',{'job_key':job['job_key'],'timestamp':datetime.now(timezone.utc).isoformat(),'attempt':1})
        try:
            row=await execute_one(job)
            if row.get('model')!=required_model:raise ValueError('completed row model mismatch')
        except Exception as exc:
            row={'model':required_model,'events':getattr(exc,'events',[]),'error':f'{type(exc).__name__}: {exc}',
                 'termination_reason':getattr(exc,'termination_reason','execution_error'),'final_task_success':False,
                 'official_final_answer_evaluator':False,'execution_error_type':type(exc).__name__,'llm_calls':getattr(exc,'llm_calls',[])}
        row.update({k:job[k] for k in ('job_key','task_id','topology','repeat_index','condition')})
        row.update({'attempt_index':1,'donor_entry_sha256':job['entry']['entry_sha256'],'operator_version':'admin_visible_table_substitution/v1','historical_semantic_step4_equivalent':False})
        audit=audit_row(row,job['entry'],job['condition']);row['table_exposure']=audit['exposure'];row['table_audit']=audit
        _append(out/'main_runs.jsonl',row);rows.append(row);completed.add(job['job_key'])
        print(json.dumps({'job_key':job['job_key'],'completed':len(completed),'expected':len(jobs),'exposure':audit['exposure'],'error':bool(row.get('error'))}),flush=True)
    gate={'expected':len(jobs),'rows':len(rows),'completed':len(completed)==len(jobs),'exposures':dict(Counter(r['table_exposure'] for r in rows)),
          'execution_errors':sum(bool(r.get('error')) for r in rows),'audit_errors':[{'job_key':r['job_key'],'errors':r['table_audit']['errors']} for r in rows if r['table_audit']['errors']]}
    gate['passed']=bool(jobs) and gate['completed'] and not gate['execution_errors'] and not gate['audit_errors']
    (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    return gate


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','formal'],required=True)
    p.add_argument('--topology',choices=TOPOLOGIES,action='append');p.add_argument('--donor-manifest',type=Path,required=True)
    p.add_argument('--task-manifest',type=Path,required=True);p.add_argument('--config-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--required-model',default='deepseek-v4-flash')
    p.add_argument('--shopping-admin-url',default='http://192.0.2.10:7780/admin')
    p.add_argument('--dotenv',type=Path,default=Path('/opt/mas_i3_20260927/.env.local'))
    a=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(a.dotenv,override=False)
    from mas_faults import webarena_admin_confirmation as confirmation
    from mas_faults.webarena_admin_controlled import EvaluatorWorkerClient
    from mas_faults.webarena_admin_real import BrowserWorkerClient,load_admin_tasks
    from mas_faults.webarena_admin_main_matrix import MainConditionCell
    from mas_faults.llm_client import get_llm_client
    from mas_faults.deepseek_schedule import ensure_deepseek_offpeak
    from run_webarena_admin_controlled_fault_matrix import browser_environment
    from adapter import make_conditions,bind_confirmation_runner
    manifest=json.loads(a.donor_manifest.read_text());jobs=build_schedule(manifest,phase=a.phase,topologies=tuple(a.topology or TOPOLOGIES))
    if {e['donor']['model'] for e in manifest['entries']}!={a.required_model}:raise ValueError('donor source model mismatch')
    tasks={str(t['task_id']):t for t in load_admin_tasks(a.task_manifest)}
    for job in jobs:
        if tasks[job['task_id']]['task_stratum']!=job['entry']['target']['task_stratum']:raise ValueError('task stratum mismatch')
    root=Path(__file__).resolve().parent
    code_files=list(root.glob('*.py'))+list((root/'src').rglob('*.py'))
    run_manifest={'schema':'admin_matched_table_runner/v1','phase':a.phase,'model':a.required_model,'jobs':[{k:j[k] for k in ('job_key','task_id','topology','repeat_index','condition')} for j in jobs],
       'donor_manifest_sha256':hashlib.sha256(a.donor_manifest.read_bytes()).hexdigest(),'task_manifest_sha256':hashlib.sha256(a.task_manifest.read_bytes()).hexdigest(),
       'config_sha256':{tid:hashlib.sha256((a.config_dir/f'{tid}.json').read_bytes()).hexdigest() for tid in sorted({j['task_id'] for j in jobs})},
       'code_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(code_files)},
       'repeat_policy':{str(r):repeat_policy(a.required_model,r) for r in (1,2,3)},'inference_environment':{k:os.getenv(k) for k in ('LLM_PROVIDER','LLM_MODEL','LLM_BASE_URL')},
       'attempt_policy':'exactly one persisted attempt per scheduled job; no outcome-dependent retry','semantic_comparison_scope':'new same-table I3/I4 operators, not historical evidence_result replacement'}
    a.output_dir.mkdir(parents=True,exist_ok=True);mf=a.output_dir/'run_manifest.json'
    if mf.exists() and json.loads(mf.read_text())!=run_manifest:raise ValueError('resume manifest mismatch')
    mf.write_text(json.dumps(run_manifest,indent=2)+'\n')
    ensure_deepseek_offpeak(a.required_model,enabled=True)
    environment=browser_environment(a.shopping_admin_url)
    cells=make_conditions(MainConditionCell)
    async def execute_one(job):
        policy=repeat_policy(a.required_model,job['repeat_index'])
        os.environ['LLM_DISABLE_THINKING']='1'
        os.environ['LLM_REQUEST_TIMEOUT_SECONDS']=str(policy['request_timeout'])
        os.environ['LLM_TOTAL_REQUEST_TIMEOUT_SECONDS']=str(policy['total_timeout'])
        if policy['max_tokens'] is None:os.environ.pop('LLM_MAX_TOKENS',None)
        else:os.environ['LLM_MAX_TOKENS']=str(policy['max_tokens'])
        client=get_llm_client(mock_llm=False)
        if client.model_info.model!=a.required_model:raise ValueError('model mismatch')
        browser=None;evaluator=None
        try:
            browser=BrowserWorkerClient(env=environment,browser_only=True)
            evaluator=EvaluatorWorkerClient(env=environment)
            with tempfile.TemporaryDirectory(prefix='admin-table-') as temp:
                return await bind_confirmation_runner(confirmation,job['entry'])(client,browser,evaluator,tasks[job['task_id']],original_config_file=a.config_dir/f"{job['task_id']}.json",sanitized_config_dir=Path(temp),topology=job['topology'],condition_cell=cells[job['condition']],run_index=job['repeat_index'],max_steps=policy['max_steps'])
        except Exception as exc:
            exc.llm_calls=list(getattr(client,'request_log',[]))
            raise
        finally:
            if browser is not None:browser.close()
            if evaluator is not None:evaluator.close()
    gate=asyncio.run(execute_jobs(jobs,execute_one,a.output_dir,required_model=a.required_model,before_job=lambda job:ensure_deepseek_offpeak(a.required_model,enabled=True)))
    print(json.dumps(gate),flush=True)
    if not gate['passed']:raise SystemExit(2)

if __name__=='__main__':main()
