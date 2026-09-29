import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from fault_launcher import select_jobs
from runtime import apply_fault

def test_fixed_fault_subset_not_success_selected():
 d=json.loads((Path(__file__).parent/'design_candidate.json').read_text());j=select_jobs(d)
 assert len(j)==108 and len({x['job_key'] for x in j})==108
 assert {x['task_id'] for x in j}==set(range(27,32))|set(range(66,70))
 assert {x['variant'] for x in j}=={'baseline','semantic_only'}
 assert {x['condition'] for x in j}=={'valid_partial','semantic_corruption'}
 assert all(x['topology']=='sequential' for x in j)
def test_operators_remain_exact_prior_domain_definitions():
 t={'task_id':27};e={'downvoted_comment_count':0,'downvoted_comments':[]}
 assert apply_fault(e,'semantic_corruption',t)[0]['downvoted_comment_count']==1
 assert 'downvoted_comments' not in apply_fault(e,'valid_partial',t)[0]
 t={'task_id':66};e={'posts':[{'forum':'books'}],'requested_count':10}
 changed,applied,op=apply_fault(e,'semantic_corruption',t)
 assert applied and op=='p2_reddit_forum_binding_corruption_v1' and changed['requested_count']==10
