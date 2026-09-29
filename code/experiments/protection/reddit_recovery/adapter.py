"""Minimal wrapper: frozen legacy collector, receiver-only contract, audited fresh GET."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import threading
import uuid
from urllib.parse import urljoin,urlsplit,urlencode,parse_qsl
import requests

QUERY_IDS=frozenset([27,28,29,30,31,66,67,68,69])
PUBLIC=('task_id','intent','task_stratum','instantiation_dict','sites','require_login','start_url')
def public_task(task):
 return {k:deepcopy(task[k]) for k in PUBLIC if k in task}
class HTTPBudgetExhausted(RuntimeError):pass
class OnceBudget:
 def __init__(self):self.used=False;self._lock=threading.Lock()
 def reserve(self):
  with self._lock:
   if self.used:return False
   self.used=True;return True
class ReceiptSession:
 """Caller owns authenticated zero-retry session; no login/mutation methods exposed."""
 def __init__(self,session,origin,receipts,*,max_gets):
  self.session=session;self.origin=origin.rstrip('/');self.receipts=receipts;self.max_gets=max_gets
  self.phase='initial';self.collection_id=None
  if hasattr(session,'adapters') and any(x.max_retries.total not in (0,False) for x in session.adapters.values()):raise ValueError('hidden_retries_forbidden')
 def _check(self,url):
  p=urlsplit(url);o=urlsplit(self.origin)
  if (p.scheme,p.netloc)!=(o.scheme,o.netloc) or p.username or p.password:raise ValueError('off_origin')
  parts=p.path.strip('/').split('/')
  if any(x in {'edit','delete','vote','login','login_check','logout','submit','subscribe','unsubscribe'} for x in parts):raise ValueError('mutation_or_auth_path')
  if not (p.path=='/search' or (parts[0]=='f' and len(parts)>=2) or (parts[0]=='user' and len(parts)==3 and parts[2] in {'comments','submissions'})):raise ValueError('unapproved_path')
  if any(k not in {'q','t','next[id]','next[timestamp]','next[ranking]'} for k,v in parse_qsl(p.query)):raise ValueError('unapproved_query')
 def get(self,url,**kwargs):
  params=kwargs.pop('params',None)
  if params:url+=('&' if '?' in url else '?')+urlencode(params)
  kwargs.pop('allow_redirects',None)
  for hop in range(6):
   self._check(url)
   if len(self.receipts)>=self.max_gets:raise HTTPBudgetExhausted('http_budget_exhausted')
   receipt={'request_id':uuid.uuid4().hex,'collection_id':self.collection_id,'phase':self.phase,'method':'GET','url':url,'started_utc':datetime.now(timezone.utc).isoformat(),'status':None,'response_sha256':None,'error_type':None}
   self.receipts.append(receipt)
   try:
    response=self.session.get(url,allow_redirects=False,**kwargs)
    receipt.update(status=response.status_code,response_sha256=hashlib.sha256(response.content).hexdigest(),byte_length=len(response.content))
   except Exception as exc:
    receipt['error_type']=type(exc).__name__;raise
   finally:receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
   if response.status_code in {301,302,303,307,308}:
    url=urljoin(url,response.headers['Location']);continue
   self._check(response.url or url)
   return response
  raise RuntimeError('redirect_limit')
class LegacyQueryAdapter:
 def __init__(self,worker,transport):self.worker=worker;self.transport=transport
 def collect(self,task,*,phase='initial'):
  if int(task['task_id']) not in QUERY_IDS:raise ValueError('731_action_requires_isolation_and_edit_permission_gate')
  self.transport.phase=phase;self.transport.collection_id=uuid.uuid4().hex
  return self.worker.collect(public_task(task))
 def reacquire(self,task,budget):
  if not budget.reserve():return {'attempted':False,'acquired':False,'new_gets':0,'error_type':'budget_exhausted'}
  before=len(self.transport.receipts)
  try:
   evidence=self.collect(task,phase='readback');new=len(self.transport.receipts)-before
   if new==0:raise RuntimeError('cached_collection_is_not_reacquisition')
   return {'attempted':True,'acquired':True,'new_gets':new,'evidence':evidence,'contract':check_evidence(task,evidence),'error_type':None}
  except Exception as exc:return {'attempted':True,'acquired':False,'new_gets':len(self.transport.receipts)-before,'error_type':type(exc).__name__}

def check_evidence(task,evidence):
 """No original carrier, evaluator, receipt or fault labels are arguments."""
 task=public_task(task);reasons=[];limits=[]
 if not isinstance(evidence,dict):return {'accepted':False,'reasons':['evidence_not_object'],'limitations':[]}
 p=task.get('instantiation_dict',{});forum=str(p.get('forum') or p.get('subreddit') or '')
 if str(evidence.get('task_id'))!=str(task['task_id']):reasons.append('task_identity_mismatch')
 if evidence.get('query_type')!=task.get('task_stratum'):reasons.append('query_type_mismatch')
 if evidence.get('requested_forum')!=forum or str(evidence.get('resolved_forum','')).casefold()!=forum.casefold():reasons.append('forum_binding_mismatch')
 if int(task['task_id']) in range(27,32):
  limits=['single_comment_page','missing_score_legacy_defaults_zero','full_comment_scope_unverified']
  latest=evidence.get('latest_submission');rows=evidence.get('downvoted_comments');n=evidence.get('downvoted_comment_count');total=evidence.get('comments_total')
  if not isinstance(latest,dict) or not latest.get('author') or latest.get('forum','').casefold()!=forum.casefold():reasons.append('latest_submission_binding_missing')
  if not isinstance(rows,list):reasons.append('required_records_missing')
  else:
   if type(n) is not int or n!=len(rows):reasons.append('count_record_mismatch')
   if any(not isinstance(r,dict) or type(r.get('score')) is not int or r['score']>=0 or not r.get('comment_id') for r in rows):reasons.append('invalid_negative_record')
  if type(total) is not int or total<0 or (type(n) is int and not 0<=n<=total):reasons.append('invalid_count_relation')
 else:
  limits=['sort_not_observable_in_legacy_payload','default_hot_listing','body_detail_fallback_only','historical_answer_guidance_preserved']
  posts=evidence.get('posts');count=p.get('number',10)
  if evidence.get('requested_count')!=count:reasons.append('requested_count_mismatch')
  if not isinstance(posts,list) or not posts:reasons.append('required_posts_missing')
  elif len(posts)>int(count) or any(not isinstance(x,dict) or x.get('rank')!=i+1 or str(x.get('forum','')).casefold()!=forum.casefold() or not x.get('title') or not x.get('url') for i,x in enumerate(posts)):reasons.append('invalid_post_records')
 return {'accepted':not reasons,'reasons':reasons,'limitations':limits}
