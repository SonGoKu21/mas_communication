"""Prepare/run exactly 54 prespecified sequential Reddit clean jobs; no implicit admission."""
import argparse,asyncio,hashlib,importlib.util,json,os,sys,time,multiprocessing,signal
from datetime import datetime,timezone
import json as json_module
from pathlib import Path
ROOT=Path(__file__).resolve().parent
IDS=set(range(27,32))|set(range(66,70))
OUTPUT=Path(os.environ.get('MAS_REDDIT_OUTPUT', 'results/reddit_recovery'))
STAGE='reddit_sequential_clean_54'
COLLECTOR=ROOT/'webarena_reddit_stateful_real.py'
LEGACY=ROOT.parent/'evidence_checks/legacy'
WEBARENA_ROOT=Path(os.environ.get('WEBARENA_ROOT', '/opt/webarena'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def select_jobs(design):
 jobs=[j for j in design['jobs'] if j['domain']=='reddit' and j['task_id'] in IDS and j['topology']=='sequential' and j['condition']=='clean']
 if len(design['jobs'])!=720 or len(jobs)!=54 or len({j['job_key'] for j in jobs})!=54:raise ValueError('fixed720_and54_required')
 from collections import Counter
 if Counter(j['task_id'] for j in jobs)!=Counter({i:6 for i in IDS}):raise ValueError('all_nine_candidates_required')
 return jobs

def append_durable(path,row):
 with Path(path).open('a',encoding='utf-8') as f:
  f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def claim_job(directory,key):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 name=hashlib.sha256(key.encode()).hexdigest()
 try:fd=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 except FileExistsError:return False
 with os.fdopen(fd,'w') as f:f.write(key);f.flush();os.fsync(f.fileno())
 directory_fd=os.open(directory,os.O_RDONLY)
 try:os.fsync(directory_fd)
 finally:os.close(directory_fd)
 return True

def validate_admission(admission,design_path,source_root):
 if admission.get('execution_eligible') is not True or admission.get('stage')!=STAGE:raise ValueError('explicit_stage_admission_required')
 if admission.get('design_sha256')!=sha(design_path):raise ValueError('design_hash_mismatch')
 if admission.get('result_directory')!=str(OUTPUT):raise ValueError('fixed_output_required')
 expected={p.name:sha(p) for p in Path(source_root).glob('*.py') if not p.name.startswith('test_')}
 if admission.get('source_hashes')!=expected:raise ValueError('source_hash_mismatch')
 if not admission.get('legacy_client_sha256') or sha(LEGACY/'src/mas_faults/llm_client.py')!=admission['legacy_client_sha256']:raise ValueError('legacy_client_hash_mismatch')
 manifest=Path(source_root)/'legacy_dependency_hashes.json'
 if admission.get('legacy_dependency_manifest_sha256')!=sha(manifest):raise ValueError('dependency_manifest_hash_mismatch')
 actual={str(p.relative_to(LEGACY)):sha(p) for p in LEGACY.rglob('*.py')}
 if actual!=json.loads(manifest.read_text()):raise ValueError('legacy_dependency_tree_changed')
 if admission.get('collector_sha256')!=sha(COLLECTOR):raise ValueError('collector_hash_mismatch')
 if admission.get('max_http_gets')!=26 or admission.get('max_model_calls')!=2 or admission.get('trial_seconds')!=1200:raise ValueError('frozen_resource_limits_required')
 if admission.get('approved_by_root') is not True:raise ValueError('explicit_execution_approval_required')
 return select_jobs(json.loads(Path(design_path).read_text()))

def configure():
 sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(LEGACY/'src'))
 import mas_faults
 for name in ('webarena_reddit_stateful_real','webarena_reddit_official'):
  spec=importlib.util.spec_from_file_location('mas_faults.'+name,COLLECTOR if name=='webarena_reddit_stateful_real' else ROOT/(name+'.py'))
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 import run_shopping_multimechanism as common
 return common

class GuardedClient:
 def __init__(self,client,offpeak,clock=time.monotonic):
  self._client=client;self._offpeak=offpeak;self._clock=clock;self._started=clock();self._requests=0
 def __getattr__(self,name):
  if name.startswith('complete'):raise AttributeError(name)
  return getattr(self._client,name)
 def _request(self,method,*a,**kw):
  self._offpeak()
  if self._requests>=2 or self._clock()-self._started>1200:raise RuntimeError('trial_budget_exhausted')
  self._requests+=1
  return getattr(self._client,method)(*a,**kw)
 def complete(self,*a,**kw):return self._request('complete',*a,**kw)
 def complete_with_metadata(self,*a,**kw):return self._request('complete_with_metadata',*a,**kw)

def evaluator_environment():
 keys=('PLAYWRIGHT_BROWSERS_PATH','SHOPPING','SHOPPING_ADMIN','REDDIT','GITLAB','MAP','WIKIPEDIA','HOMEPAGE')
 missing=[key for key in keys if not os.environ.get(key)]
 if missing:raise ValueError('Set evaluator environment: '+', '.join(missing))
 return {key:os.environ[key] for key in keys}

def execute_one(job,task,admission,common):
 import requests
 from adapter import ReceiptSession,LegacyQueryAdapter
 from runtime import run_trial
 from mas_faults.webarena_reddit_stateful_real import RedditEvidenceWorker,RedditHTTPExecutor
 from mas_faults.webarena_admin_controlled import EvaluatorWorkerClient
 from scripts.run_multimechanism_parallel import _real_client
 common.ensure_deepseek_offpeak(common.MODEL)
 started=time.monotonic();session=requests.Session();session.trust_env=False;receipts=[]
 wrapped=ReceiptSession(session,'http://localhost:7771',receipts,max_gets=26)
 adapter=LegacyQueryAdapter(RedditEvidenceWorker(RedditHTTPExecutor('http://localhost:7771',session=wrapped)),wrapped)
 evaluator=None;client=None
 try:
  with common.local_inference_transport():
   client=_real_client(common.inference_settings())
   evaluator=EvaluatorWorkerClient(webarena_root=str(WEBARENA_ROOT),env=evaluator_environment())
   row=asyncio.run(run_trial(job,task,{'adapter':adapter,'client':GuardedClient(client,lambda:common.ensure_deepseek_offpeak(common.MODEL)),'evaluator':evaluator,'evaluator_config_file':str(WEBARENA_ROOT/'config_files'/f'{task["task_id"]}.json')}))
   row['execution_eligible']=True;row['terminal_status']='completed'
   rb=row.get('p2',{}).get('readback',{})
   if rb.get('attempted') and not rb.get('acquired'):
    row['terminal_status']='error';row['error_type']=rb.get('error_type');row['final_task_success']=None;row['error_preserved']=True
   row['model_requests']=client.request_log
   row['execution_class']='model_output_error' if row.get('final_answer',{}).get('reason')=='invalid_coordinator_json' else 'completed'
   if row['execution_class']=='model_output_error':row['final_task_success']=False
   return row
 except Exception as exc:
  model_invalid=False # Exception origin is not proved; do not infer it from class/call count.
  return dict(job,terminal_status='error',execution_class='model_output_error' if model_invalid else 'infrastructure_or_setup_unknown',error_type=type(exc).__name__,final_task_success=False if model_invalid else None,http_receipts=receipts,model_requests=getattr(client,'request_log',[]),error_preserved=True)
 finally:
  session.close()
  if evaluator is not None:evaluator.close()

def _child(connection,job,task,admission):
 os.setsid()
 try:connection.send(execute_one(job,task,admission,configure()))
 except BaseException as exc:connection.send(dict(job,terminal_status='error',error_type=type(exc).__name__,final_task_success=None,error_preserved=True))
 finally:connection.close()

def run_isolated(job,task,admission):
 context=multiprocessing.get_context('spawn');parent,child=context.Pipe(duplex=False)
 process=context.Process(target=_child,args=(child,job,task,admission));process.start();child.close()
 try:
  if parent.poll(admission['trial_seconds']):
   try:result=parent.recv()
   except EOFError:result=dict(job,terminal_status='error',error_type='worker_exited_without_result',final_task_success=None,error_preserved=True)
  else:result=dict(job,terminal_status='error',error_type='hard_trial_timeout',final_task_success=None,model_usage_unknown=True,error_preserved=True)
 finally:
  if process.is_alive():
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:process.terminate()
  process.join(5)
  if process.is_alive():
   try:os.killpg(process.pid,signal.SIGKILL)
   except ProcessLookupError:process.kill()
   process.join()
  parent.close()
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--design',type=Path,default=ROOT/'design_candidate.json');p.add_argument('--admission',type=Path,default=ROOT/'admission_candidate.json');p.add_argument('--run',action='store_true');p.add_argument('--max-jobs',type=int,default=2);a=p.parse_args()
 if a.max_jobs<1 or a.max_jobs>54:raise ValueError('max_jobs_must_be_1_to_54')
 design=json.loads(a.design.read_text());jobs=select_jobs(design)
 if not a.run:
  print(json.dumps({'jobs':len(jobs),'task_ids':sorted(IDS),'model_calls':0,'execution_eligible':False,'design_sha256':sha(a.design)}));return
 admission=json.loads(a.admission.read_text());jobs=validate_admission(admission,a.design,ROOT)
 common=configure();common.inference_settings();common.ensure_deepseek_offpeak(common.MODEL)
 OUTPUT.mkdir(parents=True,exist_ok=True)
 import fcntl
 with (OUTPUT/'execution.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  tasks={t['task_id']:t for t in design['tasks'] if t['domain']=='reddit'}
  count=0
  for job in jobs:
   if count>=a.max_jobs:break
   from mas_faults.deepseek_schedule import is_deepseek_offpeak
   if not is_deepseek_offpeak():break
   if not claim_job(OUTPUT/'permanent_claims',job['job_key']):continue
   append_durable(OUTPUT/'started.jsonl',dict(job,status='started',at=datetime.now(timezone.utc).isoformat(),design_sha256=sha(a.design),source_hashes=admission['source_hashes']))
   row=run_isolated(job,tasks[job['task_id']],admission)
   append_durable(OUTPUT/'results.jsonl',row)
   count+=1
   if row['terminal_status']=='error':break
if __name__=='__main__':main()
