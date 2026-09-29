import importlib.util
from pathlib import Path
import sys
import pytest
import requests
sys.path.insert(0,str(Path(__file__).parent))
from adapter import LegacyQueryAdapter, ReceiptSession, check_evidence, OnceBudget
SOURCE=Path(__file__).parent/'webarena_reddit_stateful_real.py'
spec=importlib.util.spec_from_file_location('legacy_reddit_test',SOURCE)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
TASK={'task_id':27,'intent':'count','task_stratum':'comment_state_query','instantiation_dict':{'forum':'space'},'eval':{'poison':1},'fault_type':'poison'}
class Session:
 def __init__(self): self.calls=[]
 def get(self,url,**kwargs):
  self.calls.append(url);r=requests.Response();r.status_code=200;r.url=url
  r._content=(b'<article class="submission"><a class="submission__title" href="/f/space/1/title">Title</a><a class="submission__submitter">Alice</a></article>' if url.endswith('/new') else b'<div class="comment" id="comment_1"><span class="vote__net-score">-1</span></div>');return r

def make():
 session=Session();receipts=[];wrapped=ReceiptSession(session,'http://example.test',receipts,max_gets=20)
 worker=legacy.RedditEvidenceWorker(legacy.RedditHTTPExecutor('http://example.test',session=wrapped))
 return LegacyQueryAdapter(worker,wrapped),session,receipts

def test_collect_preserves_exact_historical_payload_and_requests():
 a,s,r=make();first=a.collect(TASK); assert first['downvoted_comment_count']==1
 assert s.calls==['http://example.test/f/space','http://example.test/f/space/new','http://example.test/user/Alice/comments']
 assert len(r)==3 and all(x['phase']=='initial' for x in r)

def test_once_reacquire_same_bytes_new_requests_and_no_oracle():
 a,s,r=make();original=a.collect(TASK);budget=OnceBudget();result=a.reacquire(TASK,budget)
 assert result['evidence']==original and result['new_gets']==3 and result['attempted']
 assert len({x['request_id'] for x in r})==6
 assert not a.reacquire(TASK,budget)['attempted']
 assert check_evidence(TASK,original)['accepted']

def test_wrong_forum_and_partial_list_reject():
 a,_,_=make();e=a.collect(TASK);e['resolved_forum']='WorcesterMA';assert not check_evidence(TASK,e)['accepted']
 e=a.collect(TASK);e['downvoted_comments']=[];assert 'count_record_mismatch' in check_evidence(TASK,e)['reasons']

def test_consistent_wrong_text_can_pass():
 a,_,_=make();e=a.collect(TASK);e['downvoted_comments'][0]['text']='plausible wrong text';assert check_evidence(TASK,e)['accepted']

def test_block_off_origin_mutation_and_credentials():
 _,s,_=make();wrapper=ReceiptSession(s,'http://example.test',[],max_gets=3)
 for url in ['http://evil.test/f/books','http://example.test/f/books/1/-/edit','http://example.test/login','http://user:pass@example.test/f/books']:
  with pytest.raises(ValueError):wrapper.get(url)
 assert not s.calls

def test_timeout_consumes_once_and_emits_receipt():
 a,s,r=make()
 def fail(*args,**kwargs):raise requests.Timeout('sensitive')
 s.get=fail;b=OnceBudget();out=a.reacquire(TASK,b)
 assert out['attempted'] and not out['acquired'] and b.used
 assert r[-1]['error_type']=='Timeout' and 'sensitive' not in str(r)

def test_731_not_silently_query():
 a,_,_=make()
 with pytest.raises(ValueError,match='731'):a.collect({'task_id':731})

def test_books_keeps_default_list_and_does_not_claim_top():
 task={'task_id':66,'task_stratum':'top_post_semantic_query','instantiation_dict':{'forum':'books','number':10}}
 a,s,r=make();e=a.collect(task)
 assert s.calls==['http://example.test/f/books','http://example.test/f/books']
 assert 'sort_not_observable_in_legacy_payload' in check_evidence(task,e)['limitations']

def test_get_budget_is_distinct_resource_error():
 from adapter import HTTPBudgetExhausted
 a,s,r=make();a.transport.max_gets=1
 with pytest.raises(HTTPBudgetExhausted):a.collect(TASK)
 assert len(r)==1
