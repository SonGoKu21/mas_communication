from pathlib import Path
import json,pytest
from launcher import select_jobs,claim_job,append_durable,validate_admission
DESIGN=Path(__file__).resolve().parent.parent/'runner/design_candidate.json'
if not DESIGN.exists():DESIGN=Path(__file__).parent/'design_candidate.json'
def test_subset_is_54_and_retains_each_id():
 design=json.loads(DESIGN.read_text());jobs=select_jobs(design)
 assert len(jobs)==54 and {j['task_id'] for j in jobs}==set(range(27,32))|set(range(66,70))
 assert all(j['condition']=='clean' and j['topology']=='sequential' for j in jobs)
def test_claim_is_permanent_across_output_choices(tmp_path):
 assert claim_job(tmp_path,'a') is True
 assert claim_job(tmp_path,'a') is False
 assert claim_job(tmp_path,'b') is True
def test_admission_failclosed_before_any_output(tmp_path):
 with pytest.raises(ValueError):validate_admission({},DESIGN,Path(__file__).parent)
 assert not list(tmp_path.iterdir())
def test_durable_log(tmp_path):
 p=tmp_path/'log.jsonl';append_durable(p,{'job_key':'a','status':'started'});append_durable(p,{'job_key':'b','status':'done'})
 assert len(p.read_text().splitlines())==2

def test_metadata_model_entrypoint_shares_call_guard():
 from launcher import GuardedClient
 class Client:
  call_count=0
  def complete(self,*a,**k):self.call_count+=1;return 'ok'
  def complete_with_metadata(self,*a,**k):self.call_count+=1;return 'ok'
 checks=[];client=GuardedClient(Client(),lambda:checks.append(1),clock=lambda:0)
 assert client.complete('a')=='ok' and client.complete_with_metadata('b')=='ok'
 with pytest.raises(RuntimeError,match='budget'):client.complete_with_metadata('c')
 assert len(checks)==3
