"""Sequential historical AutoGen graph with one new evidence boundary.

All historical agent prompts/evaluator calls are invoked directly, unchanged.
Flat requires separately frozen structure adaptation and is not admitted here.
"""
from copy import deepcopy
from adapter import OnceBudget,check_evidence,public_task

def apply_fault(evidence,condition,task):
 delivered=deepcopy(evidence);field='downvoted_comments' if int(task['task_id'])<32 else 'posts'
 if condition=='clean':return delivered,False,None
 if condition=='valid_partial':
  if field not in delivered:return delivered,False,'required_field_absent_before_fault'
  del delivered[field];return delivered,True,'p2_reddit_drop_primary_collection_v1'
 if condition=='semantic_corruption':
  if int(task['task_id'])<32 and type(delivered.get('downvoted_comment_count')) is int:
   delivered['downvoted_comment_count']+=1;return delivered,True,'p2_reddit_observed_count_plus_one_v1'
  if isinstance(delivered.get('posts'),list) and delivered['posts']:
   delivered['posts'][0]['forum']='p2_wrong_forum';return delivered,True,'p2_reddit_forum_binding_corruption_v1'
  return delivered,False,'empty_post_collection_not_exposed'
 raise ValueError('unknown_condition')

async def run_trial(job,task,config):
 """config: adapter/client/evaluator/evaluator_config_file, no implicit network setup."""
 if job['topology']!='sequential':raise ValueError('flat_query_structure_not_admitted')
 if int(task['task_id'])==731:raise ValueError('731_permission_and_isolation_gate_pending')
 if job['variant'] not in {'baseline','semantic_only'}:raise ValueError('unknown_variant')
 from webarena_reddit_official import run_clean_reddit_evidence_task
 adapter=config['adapter'];budget=OnceBudget();audit={'replacement_used':False,'readback':{'attempted':False},'fault_applied':False}
 class BoundaryWorker:
  def collect(self,public):
   original=adapter.collect(public);delivered,applied,operator=apply_fault(original,job['condition'],public)
   audit.update(original_evidence=deepcopy(original),delivered_evidence=deepcopy(delivered),fault_applied=applied,operator=operator)
   audit['contract_initial']=check_evidence(public,delivered)
   selected=delivered
   if job['variant']=='semantic_only' and not audit['contract_initial']['accepted']:
    recovered=adapter.reacquire(public,budget);audit['readback']=recovered
    if recovered.get('acquired') and recovered['contract']['accepted']:
     selected=recovered['evidence'];audit['replacement_used']=True
   audit['contract_final']=check_evidence(public,selected)
   return {} if job['variant']=='semantic_only' and not audit['contract_final']['accepted'] else selected
 # The historical agents inspect only task intent/instantiation fields; evaluator
 # retains the full original task through its original evaluation implementation.
 agent_task=public_task(task)
 # Routing stub contains no reference answer; only evaluator subprocess reads original config.
 agent_task['eval']={'reference_answers':{'exact_match':''}}
 row=await run_clean_reddit_evidence_task(config['client'],BoundaryWorker(),config.get('evaluator'),agent_task,evaluator_config_file=config['evaluator_config_file'],run_index=job['repeat_index'])
 row.update(deepcopy(job));row['p2']=audit;row['fault_applied']=audit['fault_applied']
 row['http_receipts']=deepcopy(adapter.transport.receipts)
 row['p2']['query_correctness_verified']=False
 row['p2']['readback_budget_used']=int(budget.used)
 row['p2']['source_runtime']='historical_run_clean_reddit_evidence_task_unmodified'
 row['p2']['task_evaluator_success']=row['final_task_success']
 row['p2']['task_evaluator_score']=row['task_score']
 if job['variant']=='semantic_only' and not audit['contract_final']['accepted']:
  row['final_task_success']=False
  row['p2']['acceptance_blocked']=True
 row['execution_eligible']=False
 return row
