"""Offline source-bound within-experiment pairing and six-product cluster bootstrap."""
import collections,hashlib,json,math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
SOURCES={'P1':('artifacts/tables/p1_analysis_rows.jsonl','artifacts/tables/p1_audit.json',432),'P3':('artifacts/tables/p3_analysis_rows.jsonl','artifacts/tables/p3_audit.json',288)}
LABELS=['success_without_recorded_error','success_with_incorrect_acceptance','failure_with_incorrect_acceptance','failure_without_recorded_error']
B=10000;SEED=20260928

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(row):
 success=row.get('final_task_success');errors=row.get('evidence_acceptance_errors')
 if type(success) is not bool:return 'unknown_outcome'
 if type(errors) is not int or errors<0:return 'unknown_acceptance_record'
 return LABELS[(0 if errors==0 else 1) if success else (3 if errors==0 else 2)]
def scope(exp,row):return row['condition'] if exp=='P1' else row['exposure_scheme']
def key(exp,row):return (row['task_id'],row['topology'],row['repeat_index'],row['condition'],scope(exp,row))
def percentile(xs,p):
 if not xs:return None
 xs=sorted(xs);v=(len(xs)-1)*p;l=int(v);return xs[l]+(xs[min(l+1,len(xs)-1)]-xs[l])*(v-l)
def ci(xs):return [percentile(xs,.025),percentile(xs,.975)] if xs else None

def pair_summary(pairs,bootstrap=B,seed=SEED):
 known=[(a,b) for a,b in pairs if type(a.get('final_task_success')) is bool and type(b.get('final_task_success')) is bool]
 unknown=len(pairs)-len(known)
 def vector(a,b):
  sa=int(a['final_task_success']);sb=int(b['final_task_success'])
  return [1,sb-sa,int(not sa and sb),int(sa and not sb),b['http_receipt_count']-a['http_receipt_count'],b['model_calls']-a['model_calls']]
 clusters=collections.defaultdict(lambda:[0]*6)
 for a,b in known:
  assert a['product_cluster']==b['product_cluster']
  clusters[a['product_cluster']]=[x+y for x,y in zip(clusters[a['product_cluster']],vector(a,b))]
 total=[sum(v[i] for v in clusters.values()) for i in range(6)]
 rng=random.Random(seed);values=list(clusters.values());draws=[]
 for _ in range(bootstrap):
  chosen=rng.choices(values,k=len(values)) if values else []
  draws.append([sum(v[i] for v in chosen) for i in range(6)])
 n,net,rescued,harmed,http,model=total
 metrics={'n_paired':len(pairs),'n_known_pairs':n,'n_unknown_outcome_pairs':unknown,'product_clusters':len(clusters),'net_additional_successes':net,'rescued':rescued,'harmed':harmed,'success_rate_difference':net/n if n else None,'success_rate_difference_cluster_ci95':ci([d[1]/d[0] for d in draws if d[0]]),'total_http_difference':http,'total_model_call_difference':model,'http_difference_per_pair':http/n if n else None,'model_difference_per_pair':model/n if n else None,'http_difference_per_pair_cluster_ci95':ci([d[4]/d[0] for d in draws if d[0]]),'model_difference_per_pair_cluster_ci95':ci([d[5]/d[0] for d in draws if d[0]]),'baseline_total_http':sum(a['http_receipt_count'] for a,b in known),'arm_total_http':sum(b['http_receipt_count'] for a,b in known),'baseline_total_models':sum(a['model_calls'] for a,b in known),'arm_total_models':sum(b['model_calls'] for a,b in known)}
 for denominator,idx in [('net_additional_success',1),('rescued',2)]:
  den=total[idx]
  for cost,costidx in [('http',4),('model_calls',5)]:
   valid=[d[costidx]/d[idx] for d in draws if d[idx]>0]
   metrics[f'{cost}_per_{denominator}']={'numerator_total_increment':total[costidx],'denominator':den,'estimate':total[costidx]/den if den>0 else None,'status':'defined' if den>0 else ('undefined_zero_denominator' if den==0 else 'undefined_negative_denominator'),'cluster_ci95':ci(valid) if den>0 and len(valid)==bootstrap else None,'bootstrap_positive_denominator_fraction':len(valid)/bootstrap,'ci_status':'percentile_all_resamples_positive' if den>0 and len(valid)==bootstrap else 'not_reported_nonpositive_denominator','interpretation':'incremental resource units; negative numerator means fewer calls; never a dollar cost'}
 transitions=collections.Counter((state(a),state(b)) for a,b in pairs)
 metrics['joint_state_transitions']=[{'baseline':a,'arm':b,'n':n} for (a,b),n in sorted(transitions.items())]
 assert rescued-harmed==net
 assert sum(x['n'] for x in metrics['joint_state_transitions'])==len(pairs)
 return metrics

def group_state(exp,rows,sc,top,arm):
 selected=[r for r in rows if scope(exp,r)==sc and r['variant']==arm and (top=='pooled' or r['topology']==top)]
 counts=collections.Counter(state(r) for r in selected)
 clusters=collections.defaultdict(list)
 for r in selected:clusters[r['product_cluster']].append(r)
 values=list(clusters.values());rng=random.Random(SEED+int(hashlib.sha256(f'state|{exp}|{sc}|{top}|{arm}'.encode()).hexdigest()[:8],16));samples={label:[] for label in LABELS}
 vectors=[[len(v)]+[sum(state(r)==label for r in v) for label in LABELS] for v in values]
 for _ in range(B):
  draw=rng.choices(vectors,k=len(vectors));n=sum(v[0] for v in draw)
  for i,label in enumerate(LABELS):samples[label].append(sum(v[i+1] for v in draw)/n)
 return {'state_rate_cluster_ci95':{label:ci(samples[label]) for label in LABELS},'experiment':exp,'scope':sc,'topology':top,'variant':arm,'n':len(selected),'counts':{k:counts[k] for k in LABELS},'outcome_unknown':counts['unknown_outcome'],'error_acceptance_unknown':counts['unknown_acceptance_record'],'recorded_error_acceptance_events':sum(r['evidence_acceptance_errors'] for r in selected if type(r.get('evidence_acceptance_errors')) is int and r['evidence_acceptance_errors']>=0),'runs_with_recorded_detection_events':sum(bool(r.get('detection_events')) for r in selected),'detection_field_missing':sum('detection_events' not in r for r in selected),'runs_with_recorded_common_recovery':sum(any(e.get('common') is True for e in r.get('recovery_events',[])) for r in selected),'runs_with_recorded_dedicated_recovery':sum(any(e.get('common') is False for e in r.get('recovery_events',[])) for r in selected)}

def main():
 all_groups=[];comparisons=[];binding={}
 for exp,(rel,arel,expected) in SOURCES.items():
  source=ROOT/rel;audit_path=ROOT/arel;audit=json.loads(audit_path.read_text());assert sha(source)==audit['main_sha256'];rows=[json.loads(x) for x in source.read_text().splitlines()];assert len(rows)==expected==audit['rows'];assert len({r['job_key'] for r in rows})==expected
  binding[exp]={'main_path':rel,'main_sha256':sha(source),'audit_path':arel,'audit_sha256':sha(audit_path),'rows':expected}
  for r in rows:
   assert r['http_receipt_count']==r['whole_budget']['used']['http']+r['whole_budget']['evaluation_http_calls']
   assert r['model_calls']==r['model_request_count']==r['whole_budget']['used']['model']
  arms=['baseline','check_only','always_readback','guarded_readback'] if exp=='P1' else ['baseline','combined']
  lookup={}
  for r in rows:
   identity=(key(exp,r),r['variant']);assert identity not in lookup;lookup[identity]=r
  scopes=sorted({scope(exp,r) for r in rows},key=lambda x:(x!='clean',x))
  for sc in scopes:
   for top in ['sequential','flat','pooled']:
    for arm in arms:all_groups.append(group_state(exp,rows,sc,top,arm))
    baseline=[r for r in rows if r['variant']=='baseline' and scope(exp,r)==sc and (top=='pooled' or r['topology']==top)]
    for arm in arms[1:]:
     pairs=[(a,lookup[(key(exp,a),arm)]) for a in baseline]
     result=pair_summary(pairs,seed=SEED+int(hashlib.sha256(f'{exp}|{sc}|{top}|{arm}'.encode()).hexdigest()[:8],16))
     result['baseline_evaluation_http']=sum(a['whole_budget']['evaluation_http_calls'] for a,b in pairs)
     result['arm_evaluation_http']=sum(b['whole_budget']['evaluation_http_calls'] for a,b in pairs)
     result['evaluation_http_difference']=result['arm_evaluation_http']-result['baseline_evaluation_http']
     result['workflow_only_http_difference']=sum(b['whole_budget']['used']['http']-a['whole_budget']['used']['http'] for a,b in pairs)
     assert result['workflow_only_http_difference']+result['evaluation_http_difference']==result['total_http_difference']
     comparisons.append({'experiment':exp,'scope':sc,'topology':top,'baseline':'baseline','arm':arm,**result})
 out={'sources':binding,'state_labels':LABELS,'definitions':{'wrong_acceptance':'evidence_acceptance_errors > 0; zero means no recorded error, missing/invalid remains unknown','pair_identity':['task_id','topology','repeat_index','condition','exposure_scheme_for_P3'],'HTTP_cost':'all recorded physical HTTP receipts including independent evaluation; separately checked workflow+evaluation accounting','model_cost':'recorded model requests/calls; excludes tokens/dollar pricing','ratio_numerator':'total arm minus baseline resources over all known paired outcomes, not just rescued cases','rescued':'baseline failure and protected success, observed paired classification not individual causal effect','net':'rescued minus harmed','confidence_interval':f'{B} paired product-cluster percentile bootstrap, 6 product clusters; resample whole product across its tasks/repeats in the group; no p-values','undefined_ratio':'point estimate null if denominator <= 0; CI withheld if any bootstrap denominator <= 0; no conditional-positive CI','inference_limits':'exploratory, six clusters, zero-width intervals not population certainty; no cross-P1/P3 randomized comparison'},'bootstrap':{'replicates':B,'seed':SEED},'states':all_groups,'comparisons':comparisons,'self_checks':{'unique_source_jobs':True,'audit_sha_bound':True,'all_pairs_complete':True,'cost_accounting_reconciled':True,'rescued_minus_harmed_equals_net':True,'transition_counts_cover_pairs':True}}
 (OUT/'protection_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({'groups':len(all_groups),'comparisons':len(comparisons),'sources':binding}))
if __name__=='__main__':main()
