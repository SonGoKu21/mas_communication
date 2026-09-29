"""Independent 27B six-cell Admin batch, one persisted attempt per job."""
import argparse,asyncio,hashlib,json,os,random,tempfile
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from admin_semantic import CONDITIONS as TABLE_CELLS,TASK_ORDER,TOPOLOGIES,validate_entry
from qwen27_admin_audit import audit_row as audit_table
from qwen27_runtime import index_policy,environment,apply_environment
CELLS=['clean','non_delivery_step2','non_delivery_step3','non_delivery_step4','visible_table_substitution_step3','visible_table_substitution_step4']
MODEL_ERRORS={'tool request argument values must be strings','tool request arguments do not match the tool contract','tool request is not strict JSON','model output is not strict JSON','model JSON does not match the required schema','decision must be accept or reject','answer must be a string','reason must be a string'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def append(p,row):
 with p.open('a') as h:h.write(json.dumps(row,ensure_ascii=False)+'\n');h.flush();os.fsync(h.fileno())
def build_schedule(manifest,phase,topologies=TOPOLOGIES):
 tasks=(4,107) if phase=='pilot' else TASK_ORDER;repeats=(1,2) if phase=='pilot' else (1,2,3)
 lookup={(e['target']['task_id'],e['target']['topology'],e['target']['repeat_index']):e for e in manifest['entries']}
 if len(lookup)!=len(manifest['entries']):raise ValueError('duplicate donor identity')
 jobs=[]
 for repeat in repeats:
  group=[]
  for task in tasks:
   for topo in topologies:
    entry=lookup[(str(task),topo,repeat)];validate_entry(entry)
    if entry['donor']['model']!='Qwen3.8-27B':raise ValueError('wrong donor model')
    for condition in CELLS:group.append({'job_key':f'qwen27-admin-{task}-{topo}-{condition}-r{repeat}','task_id':str(task),'topology':topo,'repeat_index':repeat,'condition':condition,'entry':entry})
  random.Random(20260815 if repeat==1 else 20260824).shuffle(group);jobs.extend(group)
 return jobs

def audit_row(row,entry,condition):
 if condition in TABLE_CELLS:return audit_table(row,entry,condition)
 events=row.get('events',[]);applied=[e for e in events if e.get('fault_applied')];errors=[]
 if condition=='clean':
  if applied:errors.append('clean_injected')
 else:
  if len(applied)>1:errors.append('multiple_exposures')
  for e in applied:
   if e.get('abstract_step')!=int(condition[-1]):errors.append('wrong_injection_step')
   if e.get('delivery_count')!=0 or e.get('delivered_messages')!=[]:errors.append('omission_not_dropped')
 return {'exposure':'applied' if applied else 'not_exposed','errors':errors,'applied_events':len(applied)}

def classify(row):
 error=str(row.get('error') or '')
 if not error:return 'completed_task_success' if row.get('final_task_success') else 'completed_task_failure'
 if error.startswith('ControlledRunError: ValueError: ') and error.removeprefix('ControlledRunError: ValueError: ') in MODEL_ERRORS:return 'model_output_error'
 return 'unresolved_execution_error'

async def execute(jobs,execute_one,out):
 out.mkdir(parents=True,exist_ok=True);rows=read(out/'main_runs.jsonl');bykey={j['job_key']:j for j in jobs};done=set()
 for r in rows:
  k=r['job_key']
  if k in done or k not in bykey or r['model']!='Qwen3.8-27B':raise ValueError('resume identity mismatch')
  j=bykey[k]
  if any(r.get(x)!=j[x] for x in ['task_id','topology','repeat_index','condition']):raise ValueError('resume cell mismatch')
  if audit_row(r,j['entry'],j['condition'])!=r['injection_audit']:raise ValueError('resume audit mismatch')
  done.add(k)
 if {r['job_key'] for r in read(out/'started.jsonl')}-done:raise ValueError('unresolved started attempt; refusing rerun')
 for j in jobs:
  if j['job_key'] in done:continue
  append(out/'started.jsonl',{'job_key':j['job_key'],'attempt':1,'timestamp':datetime.now(timezone.utc).isoformat()})
  try:row=await execute_one(j)
  except Exception as e:row={'model':'Qwen3.8-27B','events':getattr(e,'events',[]),'error':f'{type(e).__name__}: {e}','termination_reason':getattr(e,'termination_reason','execution_error'),'final_task_success':False,'llm_calls':getattr(e,'llm_calls',[])}
  if row.get('model')!='Qwen3.8-27B':raise ValueError('model mismatch')
  row.update({k:j[k] for k in ['job_key','task_id','topology','repeat_index','condition']});row['attempt_index']=1
  row['donor_entry_sha256']=j['entry']['entry_sha256'];row['injection_audit']=audit_row(row,j['entry'],j['condition']);row['execution_class']=classify(row)
  append(out/'main_runs.jsonl',row);rows.append(row);done.add(j['job_key']);print(json.dumps({'completed':len(done),'expected':len(jobs),'job_key':j['job_key'],'class':row['execution_class'],'exposure':row['injection_audit']['exposure']}),flush=True)
 errors=[{'job_key':r['job_key'],'errors':r['injection_audit']['errors']} for r in rows if r['injection_audit']['errors']]
 coverage=Counter((r['topology'],r['condition'],r['repeat_index']) for r in rows if r['injection_audit']['exposure']=='applied')
 gate={'complete':len(rows)==len(jobs),'rows':len(rows),'expected':len(jobs),'audit_errors':errors,'classes':dict(Counter(r['execution_class'] for r in rows)),'coverage':{'/'.join(map(str,k)):v for k,v in coverage.items()}}
 gate['passed']=gate['complete'] and not errors and not gate['classes'].get('unresolved_execution_error',0)
 (out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n');return gate,rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','formal'],required=True);p.add_argument('--topology',choices=TOPOLOGIES,action='append');p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 here=Path(__file__).resolve().parent;repo=here.parent
 manifest=json.loads((here/'donor_manifest_qwen27.json').read_text());jobs=build_schedule(manifest,a.phase,tuple(a.topology or TOPOLOGIES));policies=index_policy(json.loads((here/'frozen_paired_inference_policy.json').read_text()))
 task_manifest=Path('/var/lib/mas/task_manifests/webarena_admin_qwen_model_comparison20_frozen_20260823_v2.json');config=Path('/var/lib/mas/task_configs/webarena_admin_qwen_model_comparison20_frozen_20260823')
 fm={'phase':a.phase,'model':'Qwen3.8-27B','jobs':[{k:j[k] for k in ['job_key','task_id','topology','repeat_index','condition']} for j in jobs],'policy_sha256':sha(here/'frozen_paired_inference_policy.json'),'donor_manifest_sha256':sha(here/'donor_manifest_qwen27.json'),'task_manifest_sha256':sha(task_manifest),'config_sha256':{t:sha(config/f'{t}.json') for t in sorted({j['task_id'] for j in jobs})},'code_sha256':{str(f):sha(f) for f in list(here.glob('*.py'))+list((repo/'src').rglob('*.py'))},'attempt_policy':'one persisted attempt per job; errors retained without success-based retry','semantic_aliases':{'semantic_corruption_step3':'visible_table_substitution_step3','semantic_corruption_step4':'visible_table_substitution_step4'},'semantic_operator_scope':'new matched-table I3/I4 operator, not historical evidence_result replacement'}
 a.output_dir.mkdir(parents=True,exist_ok=True);mf=a.output_dir/'manifest.json'
 if mf.exists() and json.loads(mf.read_text())!=fm:raise ValueError('manifest changed')
 mf.write_text(json.dumps(fm,indent=2)+'\n')
 if a.plan_only:print(json.dumps({'jobs':len(jobs),'plan_only':True}));return
 from dotenv import load_dotenv
 load_dotenv(repo/'.env.local',override=False)
 from mas_faults import webarena_admin_confirmation as confirmation
 from mas_faults.webarena_admin_controlled import EvaluatorWorkerClient
 from mas_faults.webarena_admin_real import BrowserWorkerClient,load_admin_tasks
 from mas_faults.webarena_admin_main_matrix import MainConditionCell,I3_CONDITION_BY_NAME
 from mas_faults.llm_client import get_llm_client
 from run_webarena_admin_controlled_fault_matrix import browser_environment
 from adapter import make_conditions,bind_confirmation_runner
 tasks={str(t['task_id']):t for t in load_admin_tasks(task_manifest)};cells={**I3_CONDITION_BY_NAME,**make_conditions(MainConditionCell)}
 env=browser_environment('http://192.0.2.10:7780/admin')
 async def one(job):
  task=job['task_id'];repeat=job['repeat_index'];apply_environment(environment(policies,'admin',task,repeat));client=get_llm_client();browser=None;evaluator=None
  if client.model_info.model!='Qwen3.8-27B':raise ValueError('model mismatch')
  try:
   browser=BrowserWorkerClient(env=env,browser_only=True);evaluator=EvaluatorWorkerClient(env=env)
   with tempfile.TemporaryDirectory(prefix='qwen27-admin-') as temp:
    runner=bind_confirmation_runner(confirmation,job['entry']) if job['condition'] in TABLE_CELLS else confirmation.run_admin_confirmation_task
    return await runner(client,browser,evaluator,tasks[task],original_config_file=config/f'{task}.json',sanitized_config_dir=Path(temp),topology=job['topology'],condition_cell=cells[job['condition']],run_index=repeat,max_steps=policies[('admin',task,repeat)]['budget']['max_steps'])
  except Exception as e:e.llm_calls=list(getattr(client,'request_log',[]));raise
  finally:
   if browser is not None:browser.close()
   if evaluator is not None:evaluator.close()
 gate,rows=asyncio.run(execute(jobs,one,a.output_dir))
 if a.phase=='pilot':
  missing=[]
  for topo in tuple(a.topology or TOPOLOGIES):
   for condition in CELLS[1:]:
    for repeat in (1,2):
     if not any(r['topology']==topo and r['condition']==condition and r['repeat_index']==repeat and r['injection_audit']['exposure']=='applied' for r in rows):missing.append([topo,condition,repeat])
  gate['missing_actual_exposure']=missing;gate['passed']=gate['passed'] and not missing
  (a.output_dir/'readiness.json').write_text(json.dumps(gate,indent=2)+'\n')
 print(json.dumps(gate),flush=True)
 if not gate['passed']:raise SystemExit(2)
if __name__=='__main__':main()
