"""Historical silent residual sensitivity only. No P0 normalized-data merge."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'artifacts/tables'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ratio(k,n):return k/n if n else None
def summary(rs):
 known=[r for r in rs if type(r['success']) is bool];fails=[r for r in rs if r['success'] is False]
 eligible=[r for r in rs if r['eligible']];ef=[r for r in fails if r['eligible']]
 silent=sum(r['silent'] for r in ef)
 return dict(applied_n=len(rs),known_outcome_n=len(known),unknown_outcome_n=len(rs)-len(known),original_detection_eligible_n=len(eligible),original_eligible_silent=sum(r['silent'] for r in eligible),original_silent_per_eligible_applied=ratio(sum(r['silent'] for r in eligible),len(eligible)),all_known_failure_n=len(fails),eligible_failure_n=len(ef),silent_n=silent,silent_per_eligible_failure=ratio(silent,len(ef)),failure_detection_missing_n=len(fails)-len(ef),failure_detection_coverage=ratio(len(ef),len(fails)),detected_failure_n=sum(r['detected'] is True for r in ef),undetected_failure_n=sum(r['detected'] is False for r in ef),observed_silent_per_all_known_failures_lower_bound=ratio(silent,len(fails)),all_known_failures_silent_partial_identification_upper_bound=ratio(silent+len(fails)-len(ef),len(fails)),eligible_failure_tasks=len({(r['domain'],r['task']) for r in ef}),all_failure_tasks=len({(r['domain'],r['task']) for r in fails}),recovery_flag_nonboolean_failure_n=sum(type(r['raw_recovery_detected']) is not bool for r in fails))
def concentration(rs):
 # Universe is every task with an eligible failure, including zero silent cases.
 tasks=collections.defaultdict(lambda:{'eligible_failure_n':0,'silent_n':0})
 for r in rs:
  if r['success'] is False and r['eligible']:
   t=tasks[(r['domain'],r['task'])];t['eligible_failure_n']+=1;t['silent_n']+=int(r['silent'])
 ordered=sorted((dict(domain=k[0],task=k[1],**v) for k,v in tasks.items()),key=lambda x:(-x['silent_n'],x['domain'],x['task']))
 n=len(ordered);k=math.ceil(.2*n);total=sum(t['silent_n'] for t in ordered)
 if not n:return {'task_n':0,'rows':[]}
 threshold=ordered[k-1]['silent_n'];higher=[t for t in ordered if t['silent_n']>threshold];ties=[t for t in ordered if t['silent_n']==threshold]
 exact=sum(t['silent_n'] for t in higher)+(k-len(higher))*threshold
 inclusive=sum(t['silent_n'] for t in higher+ties)
 return dict(universe='all domain-task identities with >=1 detection-eligible known failure; includes zero-silent tasks',task_n=n,top20_slots=k,cutoff_silent_n=threshold,strictly_above_cutoff_n=len(higher),cutoff_tie_n=len(ties),cutoff_tie_fraction=(k-len(higher))/len(ties),silent_total=total,top20_fractional_tie_contribution=ratio(exact,total),top20_inclusive_tie_task_n=len(higher)+len(ties),top20_inclusive_tie_contribution=ratio(inclusive,total),rows=ordered)
def main():
 rows=json.loads((SOURCE/'confirmation_records.json').read_text())
 provenance=[];checked=len(rows)
 for r in rows:
  assert r['silent']==(r['success'] is False and r['detected'] is False and r['raw_recovery_detected'] is False and r['decision']=='accept' and not r['has_fault_detection_evidence'])
 clean={}
 for r in rows:
  key=tuple(r[k] for k in ('domain','model','task','topology','rep'))
  if r['condition']=='clean':assert key not in clean;clean[key]=r['success']
 for r in rows:assert r['clean_success']==clean[tuple(r[k] for k in ('domain','model','task','topology','rep'))]
 fault=[r for r in rows if r['condition']!='clean' and r['applied']]
 base=summary(fault);assert (base['original_detection_eligible_n'],base['silent_n'],base['eligible_failure_n'])==(11022,1722,8335)
 scenarios={'all_original_applied':fault,'exclude_Admin':[r for r in fault if r['domain']!='WebArena Admin'],'exclude_Shopping':[r for r in fault if r['domain']!='WebArena Shopping'],'exclude_forum_task28':[r for r in fault if not(r['domain']=='WebArena Reddit' and r['task']=='28')],'clean_success_only':[r for r in fault if r['clean_success'] is True],'all_known_failure_scope':[r for r in fault if r['success'] is False],'all_known_failure_eligible':[r for r in fault if r['success'] is False and r['eligible']]}
 domains=sorted({r['domain'] for r in fault})
 out={'inputs':{n:sha(SOURCE/n) for n in ('confirmation_records.json',)},'raw_manifest_verification':provenance,'raw_rows_recomputed':checked,'scope':'Only historical 18900 source rows; excludes P0 normalized and all new P1/P2/P3 runs','definition':'known failed outcome AND receiver_detected is exactly False AND recovery_detected is exactly False AND final decision accept AND no truthy fault_detection_evidence; original applied fault eligibility is receiver_detected boolean','original_definition_note':'Historical code uses not final_task_success; all raw outcomes are audited for boolean coverage rather than treating missing outcome as failure. B in derived data coerces missing recovery flags; silent is recomputed from raw exact-false instead.','excluded_forum_task28_rows':sum(r['domain']=='WebArena Reddit' and r['task']=='28' for r in rows),'scenario_removed_applied_n':{k:len(fault)-len(v) for k,v in scenarios.items()},'scenarios':{k:summary(v) for k,v in scenarios.items()},'by_domain':{d:summary([r for r in fault if r['domain']==d]) for d in domains},'leave_one_domain_out':{d:summary([r for r in fault if r['domain']!=d]) for d in domains},'concentration':concentration(fault),'concentration_by_domain':{d:concentration([r for r in fault if r['domain']==d]) for d in domains},'concentration_by_scenario':{k:concentration(v) for k,v in scenarios.items()}}
 (ROOT/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');report(out)
def report(o):
 print('Analysis complete; results.json contains the reproduced statistics.')
if __name__=='__main__':main()
