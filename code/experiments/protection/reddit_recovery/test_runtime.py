import asyncio,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'i3_launch_20260927/server_source/src'))
from test_adapter import make,TASK
from runtime import run_trial
class Client:
 call_count=0;prompt_tokens=0;completion_tokens=0
 model_info=type('Info',(),{'model':'offline','provider':'test'})()
 def complete(self,prompt,**kw):
  self.call_count+=1;self.prompt_tokens+=1;self.completion_tokens+=1
  return '{"decision":"accept","answer":"1","reason":"visible"}' if kw.get('json_mode') else 'Read visible records'
class Eval:
 def evaluate(self,*args):return {'score':1,'evaluator_mode':'offline_fixture'}
def test_full_graph_protection_reacquires_once():
 a,s,r=make();task=dict(TASK,eval={'reference_answers':{'exact_match':'1'}})
 job={'domain':'reddit','task_id':27,'topology':'sequential','repeat_index':0,'condition':'valid_partial','variant':'semantic_only','job_key':'j','pair_key':'p'}
 row=asyncio.run(run_trial(job,task,{'adapter':a,'client':Client(),'evaluator':Eval(),'evaluator_config_file':'fixture'}))
 assert row['api_call_count']==2 and row['p2']['readback']['new_gets']==3
 assert row['p2']['replacement_used'] and row['fault_applied'] and len(r)==6
 assert row['p2']['contract_initial']['accepted'] is False
 assert row['final_task_success'] is True

def test_baseline_no_reacquisition():
 a,s,r=make();task=dict(TASK,eval={'reference_answers':{'exact_match':'1'}})
 job={'domain':'reddit','task_id':27,'topology':'sequential','repeat_index':0,'condition':'semantic_corruption','variant':'baseline'}
 row=asyncio.run(run_trial(job,task,{'adapter':a,'client':Client(),'evaluator':Eval(),'evaluator_config_file':'fixture'}))
 assert len(r)==3 and row['evidence']['downvoted_comment_count']==2
 assert not row['p2']['replacement_used']

def test_unresolved_contract_does_not_leak_bad_evidence_to_coordinator():
 a,s,r=make();task=dict(TASK,instantiation_dict={'forum':'Worcester'},eval={'reference_answers':{'exact_match':'1'}})
 # Simulate the historically valid resolver alias; fresh acquisition repeats it.
 a.worker.executor.resolve_forum=lambda forum:'space'
 job={'task_id':27,'topology':'sequential','repeat_index':0,'condition':'clean','variant':'semantic_only'}
 row=asyncio.run(run_trial(job,task,{'adapter':a,'client':Client(),'evaluator':Eval(),'evaluator_config_file':'fixture'}))
 assert row['evidence']=={} and row['final_task_success'] is False
 assert row['p2']['task_evaluator_success'] is True

def test_books_operator_is_binding_not_requested_count():
 from runtime import apply_fault
 from adapter import check_evidence
 task={'task_id':66,'task_stratum':'top_post_semantic_query','instantiation_dict':{'forum':'books','number':10}}
 e={'task_id':66,'query_type':'top_post_semantic_query','requested_forum':'books','resolved_forum':'books','requested_count':10,'posts':[{'rank':1,'forum':'books','title':'Book','url':'http://example.test/f/books/1/book'}]}
 damaged,applied,operator=apply_fault(e,'semantic_corruption',task)
 assert applied and operator=='p2_reddit_forum_binding_corruption_v1'
 assert damaged['requested_count']==10 and not check_evidence(task,damaged)['accepted']
 assert apply_fault(dict(e,posts=[]),'semantic_corruption',task)[1] is False
