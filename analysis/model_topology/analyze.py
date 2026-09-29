"""Offline nine-configuration matched rescue and failure-overlap analysis."""
import argparse,collections,hashlib,itertools,json,platform
from pathlib import Path
import numpy as np
from core import *
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=HERE.parents[1]/'artifacts/tables/position_records.json');p.add_argument('--output',type=Path,default=HERE);p.add_argument('--draws',type=int,default=10000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
raw=a.input.read_bytes();source=json.loads(raw);records=source['records'];index={};domains=sorted({r['domain'] for r in records})
fields=('domain','task_id','repeat_index','condition','model','topology')
for r in records:
 key=tuple(r[k] for k in fields)
 assert key not in index,'duplicate identity';index[key]=r
 assert r['condition'] in ('clean',*FAULTS) and (r['model'],r['topology']) in CONFIGS
 assert type(r['success']) is bool or r['success'] is None
 assert type(r['applied']) is bool or r['applied'] is None
 assert type(r['repeat_index']) is int and r['repeat_index'] in (1,2,3)
 if r['domain']=='admin' and r['condition'].startswith('semantic'):
  assert r['operator_family']=='matched_visible_table_substitution' and r['historical_semantic_step4_equivalent'] is False
  assert r['original_condition']==r['condition'].replace('semantic_corruption','visible_table_substitution')
tasks={d:sorted({r['task_id'] for r in records if r['domain']==d}) for d in domains}
assert len(records)==16200 and sum(r['success'] is None for r in records)==3
for d in domains:
 for m,t in CONFIGS:
  assert {r['task_id'] for r in records if r['domain']==d and r['model']==m and r['topology']==t}==set(tasks[d])
keys=list(itertools.chain.from_iterable(((d,t,r,f) for t in tasks[d] for r in (1,2,3) for f in FAULTS) for d in domains))
units={};audit=[];selected={c:[] for c in COHORTS}
for key in keys:
 d,t,r,f=key
 fault={(m,top):index[(d,t,r,f,m,top)] for m,top in CONFIGS if (d,t,r,f,m,top) in index}
 clean={(m,top):index[(d,t,r,'clean',m,top)] for m,top in CONFIGS if (d,t,r,'clean',m,top) in index}
 # Present explicit operator identities must agree within this exact domain/condition unit.
 operator={x.get('operator_family') for x in fault.values() if x.get('operator_family') is not None};assert len(operator)<=1
 units[key]=fault
 reasons={c:select_unit(fault,clean,c) for c in COHORTS}
 for c,reason in reasons.items():
  if reason=='included':selected[c].append(key)
 audit.append(dict(unit=list(key),selection=reasons))
assert len(keys)==1500
clusters=[(d,t) for d in domains for t in tasks[d]];cluster_index={k:i for i,k in enumerate(clusters)}
# Resample scheduled tasks within each domain; keep all faults/repeats/configurations together.
seed=2026092807;rng=np.random.default_rng(seed);weights=np.zeros((a.draws,len(clusters)),dtype=np.int16)
for d in domains:
 cols=[cluster_index[(d,t)] for t in tasks[d]];n=len(cols)
 weights[:,cols]=rng.multinomial(n,np.full(n,1/n),size=a.draws)
 assert np.all(weights[:,cols].sum(axis=1)==n), 'domain stratum task count changed'
def bootstrap(group,bm,bt,other):
 # Sufficient counts by intact (domain,task) cluster.
 counts=np.zeros((len(clusters),9),dtype=float)
 for k in group:
  r=units[k];b=r[(bm,bt)]['success'];f=r[(bm,'flat')]['success'];m=r[(other,bt)]['success']
  counts[cluster_index[k[:2]]]+=[1,not b,b,(not b and f),(b and not f),(not b and m),(b and not m),((not b) and f and m),((not b) and (f or m))]
 sim=weights@counts
 formulas={
 'flat_net_pp':(sim[:,3]-sim[:,4],sim[:,0],100),
 'model_net_pp':(sim[:,5]-sim[:,6],sim[:,0],100),
 'flat_minus_model_net_pp':(sim[:,3]-sim[:,4]-sim[:,5]+sim[:,6],sim[:,0],100),
 'flat_rescue_pct':(sim[:,3],sim[:,1],100),
 'model_rescue_pct':(sim[:,5],sim[:,1],100),
 'flat_minus_model_rescue_pp':(sim[:,3]-sim[:,5],sim[:,1],100),
 'flat_regression_pct':(sim[:,4],sim[:,2],100),
 'model_regression_pct':(sim[:,6],sim[:,2],100),
 'rescue_jaccard':(sim[:,7],sim[:,8],1)}
 out={}
 for name,(num,den,factor) in formulas.items():
  valid=den>0;v=factor*num[valid]/den[valid];ci=np.quantile(v,[.025,.975]).tolist() if len(v) else None
  out[name]=dict(ci95=ci,valid_draws=int(valid.sum()),draws=a.draws,degenerate=ci is not None and ci[0]==ci[1])
 return out
def bootstrap_failure(group,top):
 counts=np.zeros((len(clusters),8),dtype=float)
 pairs=list(itertools.combinations(MODELS,2))
 for k in group:
  fail={m:not units[k][(m,top)]['success'] for m in MODELS}
  values=[]
  for left,right in pairs:values.extend([fail[left] and fail[right],fail[left] or fail[right]])
  values.extend([all(fail.values()),any(fail.values())]);counts[cluster_index[k[:2]]]+=values
 sim=weights@counts;out={}
 for i,name in enumerate([x+'/'+y for x,y in pairs]+['three_models']):
  valid=sim[:,2*i+1]>0;v=sim[valid,2*i]/sim[valid,2*i+1];ci=np.quantile(v,[.025,.975]).tolist() if len(v) else None
  out[name]=dict(ci95=ci,valid_draws=int(valid.sum()),draws=a.draws,degenerate=ci is not None and ci[0]==ci[1])
 return out
contrasts=[];overlaps=[];selections=[]
for cohort in COHORTS:
 for domain in ('ALL_DOMAINS',*domains):
  for condition in ('ALL_FIVE_FAULTS',*FAULTS):
   scheduled=[k for k in keys if (domain=='ALL_DOMAINS' or k[0]==domain) and (condition=='ALL_FIVE_FAULTS' or k[3]==condition)]
   group=[k for k in selected[cohort] if (domain=='ALL_DOMAINS' or k[0]==domain) and (condition=='ALL_FIVE_FAULTS' or k[3]==condition)]
   reasons=collections.Counter(x['selection'][cohort] for x in audit if tuple(x['unit']) in set(scheduled))
   selections.append(dict(cohort=cohort,domain=domain,condition=condition,scheduled=len(scheduled),included=len(group),exclusion_counts=dict(reasons)))
   for bm,bt,other in itertools.product(MODELS,('sequential','hierarchical'),MODELS):
    if bm==other:continue
    b=[units[k][(bm,bt)]['success'] for k in group];f=[units[k][(bm,'flat')]['success'] for k in group];m=[units[k][(other,bt)]['success'] for k in group]
    flat=paired(b,f);model=paired(b,m);o=overlap(b,f,m)
    out=dict(cohort=cohort,domain=domain,condition=condition,baseline_model=bm,baseline_topology=bt,alternative_model=other,flat_change=flat,model_change=model,rescue_overlap=o,flat_minus_model_net_pp=(flat['net_pp']-model['net_pp']) if group else None,flat_minus_model_rescue_pp=pct(flat['rescued']-model['rescued'],flat['baseline_failures']))
    if domain=='ALL_DOMAINS' and condition=='ALL_FIVE_FAULTS':out['bootstrap']=bootstrap(group,bm,bt,other)
    assert sum(o[x] for x in ('both','flat_only','model_only','neither'))==flat['baseline_failures']
    contrasts.append(out)
   for top in TOPS:
    fail={m:{k for k in group if not units[k][(m,top)]['success']} for m in MODELS}
    union=set.union(*fail.values());inter=set.intersection(*fail.values())
    overlaps.append(dict(cohort=cohort,domain=domain,condition=condition,topology=top,n=len(group),model_failure_counts={m:len(s) for m,s in fail.items()},pairwise=[dict(left=x,right=y,**jaccard(fail[x],fail[y])) for x,y in itertools.combinations(MODELS,2)],three_model_intersection=len(inter),three_model_union=len(union),three_model_jaccard=len(inter)/len(union) if union else None))
    if domain=='ALL_DOMAINS' and condition=='ALL_FIVE_FAULTS':overlaps[-1]['bootstrap']=bootstrap_failure(group,top)
model_aliases={m:dict(collections.Counter(str(r.get('actual_model')) for r in records if r['model']==m)) for m in MODELS}
result=dict(input_file='artifacts/tables/position_records.json',input_sha256=hashlib.sha256(raw).hexdigest(),analysis_status='exploratory_matched_descriptive_not_causal_not_multiplicity_adjusted',audit=dict(records=len(records),unique_identities=len(index),scheduled_fault_units=len(keys),task_counts= {d:len(t) for d,t in tasks.items()},known_unknown_outcomes=dict(known=sum(type(r['success']) is bool for r in records),unknown=sum(r['success'] is None for r in records)),missing_fault_configuration_units=sum(x['selection']['all_known']=='missing_fault_configuration' for x in audit),model_aliases=model_aliases),definitions=dict(unit=['domain','task_id','repeat_index','same_fault_condition'],configurations=[list(c) for c in CONFIGS],cohorts={'common_applied':'All nine exact same-fault configuration outcomes known and applied==true.','all_known':'All nine same-fault outcomes known, without requiring applied.','clean_success_common_applied':'Primary common_applied cohort plus all nine same-task/repeat clean outcomes known and successful.'},four_states='First digit baseline success, second digit candidate success: 00 persistent failure,01 rescue,10 regression,11 retained success.',rescue_overlap='Within the same baseline-failure set: both Flat change and alternative-model change rescue, Flat only, model only, neither.',net_pp='100*(rescued-regressed)/all included units',empty_union='Jaccard=null/NA, never zero',pooled_weighting='Unit-weighted observed five-fault mixture; fault identity is always retained in joins. Not a common semantic operator across domains.'),bootstrap=dict(draws=a.draws,seed=seed,method='Domain-stratified task-cluster percentile bootstrap; sample the original number of scheduled tasks with replacement within each domain; retain all eligible repeats, conditions and nine configurations of each sampled task together.',scope='Only all-domain/all-five-fault major summaries; no per-cell significance tests or multiplicity claims.',scheduled_clusters=len(clusters),retain_zero_eligible_task_clusters=True),inherited_limitations=source['limitations'],selection=selections,unit_selection=audit,contrasts=contrasts,failure_overlap=overlaps,software=dict(python=platform.python_version(),numpy=np.__version__),source_hashes=source['source_hashes'],input_hashes=source.get('input_hashes'),handoff_hashes=source.get('handoff_hashes'))
(a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(contrasts=len(contrasts),failure_overlap_groups=len(overlaps),bootstrap_contrast_summaries=sum('bootstrap' in r for r in contrasts),bootstrap_failure_overlap_summaries=sum('bootstrap' in r for r in overlaps),cohort_counts={c:len(selected[c]) for c in COHORTS}),ensure_ascii=False))
