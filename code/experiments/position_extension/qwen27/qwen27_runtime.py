"""Pure policy and reference lookup for the frozen 27B paired batch."""
import json,os
BASE={'LLM_PROVIDER':'modelscope_local','LLM_MODEL':'Qwen3.8-27B','LLM_BASE_URL':'http://127.0.0.1:18002','LLM_DISABLE_THINKING':'1'}
OLD={'swe':{'astropy__astropy-14309','django__django-10097','django__django-10914','pytest-dev__pytest-5631','scikit-learn__scikit-learn-10297'},'tac':{'ds-format-excel-sheets','qa-escalate-emergency','sde-change-license-easy','sde-close-an-issue','sde-install-go'}}
def index_references(refs):
 out={}
 for r in refs:
  key=(r['benchmark'],str(r['task_id']),int(r['repeat_index']))
  if key in out:raise ValueError('duplicate reference identity')
  if r['source_model']!='Qwen3.8-27B':raise ValueError('reference model mismatch')
  out[key]=r
 return out

def index_policy(policy):
 if policy['status']!='FROZEN_FOR_NEW_CONTEMPORANEOUS_PAIRED_BATCH':raise ValueError('policy not frozen')
 out={}
 for r in policy['proposed']:
  key=(r['domain'],str(r['task_id']),int(r['repeat_index']))
  if key in out:raise ValueError('duplicate policy identity')
  out[key]=r
 return out

def environment(policies,domain,task,repeat):
 b=policies[(domain,str(task),repeat)]['budget']
 return {**BASE,'LLM_MAX_TOKENS':None if b['max_tokens'] is None else str(b['max_tokens']),'LLM_REQUEST_TIMEOUT_SECONDS':str(b['request_timeout_seconds']),'LLM_TOTAL_REQUEST_TIMEOUT_SECONDS':str(b['total_timeout_seconds'])}

def apply_environment(env):
 for key,value in env.items():
  if value is None:os.environ.pop(key,None)
  else:os.environ[key]=value

def client_key(env):return json.dumps(env,sort_keys=True)

def pilot_representatives(domain,refs,policies):
 groups={}
 for r in sorted(refs,key=lambda r:(r['task_id'],r['repeat_index'])):
  task=str(r['task_id']);repeat=r['repeat_index']
  cohort='old' if task in OLD.get(domain,set()) else 'new'
  if domain=='reddit':cohort='736' if task=='reddit-736' else 'other'
  # astropy-14309 is the old SWE task with a changed clean payload.
  if domain=='swe' and task=='astropy__astropy-14309':cohort='old_changed_clean'
  key=(cohort,repeat,client_key(environment(policies,domain,task,repeat)))
  groups.setdefault(key,(task,repeat))
 return set(groups.values())
