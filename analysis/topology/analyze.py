import json,collections,hashlib,itertools
from pathlib import Path

def transition(pairs):
 c=collections.Counter((int(a),int(b)) for a,b in pairs);n=sum(c.values());f=c[0,0]+c[0,1]
 return dict(n=n,ss=c[1,1],rescued=c[0,1],regression=c[1,0],ff=c[0,0],baseline_failures=f,rescue_given_failure_pct=100*c[0,1]/f if f else None,net_pp=100*(c[0,1]-c[1,0])/n if n else None)

def ci(records):
 import numpy as np
 if not records:return None
 domains=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in records:domains[r['domain']][r['task']].append(int(r['right'])-int(r['left']))
 rng=np.random.default_rng(20260928);num=np.zeros(10000);den=np.zeros(10000)
 for tasks in domains.values():
  vals=np.array([[sum(v),len(v)] for v in tasks.values()]);idx=rng.integers(0,len(vals),(10000,len(vals)))
  sums=vals[idx].sum(axis=1);num+=sums[:,0];den+=sums[:,1]
 return list(np.quantile(100*num/den,[.025,.975]))

def main():
 P=Path(__file__).resolve().parent;src=P.parents[1]/'artifacts/tables/position_records.json'
 raw=src.read_bytes();rows=json.loads(raw)['records'];idx={}
 for r in rows:
  k=(r['domain'],str(r['task_id']),r['repeat_index'],r['model'],r['topology'],r['condition']);assert k not in idx;idx[k]=r
 assert len(idx)==16200
 pairs=[]
 for r in rows:
  if r['condition']=='clean' or r['topology'] not in ('sequential','hierarchical'):continue
  k=(r['domain'],str(r['task_id']),r['repeat_index'],r['model']);b=idx[(*k,'flat',r['condition'])]
  ca=idx[(*k,r['topology'],'clean')];cb=idx[(*k,'flat','clean')]
  if type(r['success']) is not bool or type(b['success']) is not bool:continue
  pairs.append(dict(domain=r['domain'],task=str(r['task_id']),repeat=r['repeat_index'],model=r['model'],left_topology=r['topology'],condition=r['condition'],left=r['success'],right=b['success'],both_applied=r['applied'] is True and b['applied'] is True,both_clean_success=ca['success'] is True and cb['success'] is True))
 primary=[p for p in pairs if p['both_applied']]
 variants={'matched_applied':primary,'all_scheduled_known':pairs,'clean_success_only':[p for p in primary if p['both_clean_success']]}
 for domain in sorted({p['domain'] for p in pairs}):variants['exclude_'+domain]=[p for p in primary if p['domain']!=domain]
 variants['exclude_forum_binding_task28']=[p for p in primary if not(p['domain']=='reddit' and p['task']=='28')]
 allout=[]
 conditions=sorted({p['condition'] for p in pairs})
 for name,subset in variants.items():
  for top in ['sequential','hierarchical']:
   for cond in ['all_faults']+conditions:
    rs=[p for p in subset if p['left_topology']==top and (cond=='all_faults' or p['condition']==cond)]
    allout.append(dict(variant=name,left_topology=top,right_topology='flat',condition=cond,**transition([(p['left'],p['right']) for p in rs]),ci95_net_pp=ci(rs),tasks=len({(p['domain'],p['task']) for p in rs})))
 detailed=[]
 for dom,model,top,cond in itertools.product(sorted({p['domain'] for p in pairs}),['deepseek','qwen9','qwen27'],['sequential','hierarchical'],conditions):
  rs=[p for p in primary if (p['domain'],p['model'],p['left_topology'],p['condition'])==(dom,model,top,cond)]
  detailed.append(dict(domain=dom,model=model,left_topology=top,condition=cond,**transition([(p['left'],p['right']) for p in rs])))
 # Hold all compared non-delivery positions eligible for the same model/task/repeat/topology pair.
 pos=['non_delivery_step'+str(i) for i in (2,3,4)];group=collections.defaultdict(dict)
 for p in primary:
  if p['condition'] in pos:group[(p['domain'],p['task'],p['repeat'],p['model'],p['left_topology'])][p['condition']]=p
 common=[v[c] for v in group.values() if set(v)==set(pos) for c in pos];commonout=[]
 for top,c in itertools.product(['sequential','hierarchical'],pos):
  rs=[p for p in common if p['left_topology']==top and p['condition']==c]
  commonout.append(dict(left_topology=top,condition=c,**transition([(p['left'],p['right']) for p in rs]),ci95_net_pp=ci(rs)))
 result=dict(source_sha256=hashlib.sha256(raw).hexdigest(),primary_definition='Both task outcomes known, both faults applied; same domain/task/repeat/fault/model; clean success not required',unknown_rows=sum(r['success'] is None for r in rows),sensitivity=allout,domain_model_cells=detailed,common_nondelivery_positions=commonout)
 (P/'results.json').write_text(json.dumps(result,indent=2));(P/'matched_pairs.json').write_text(json.dumps(pairs))
 print('Analysis complete; results.json contains the reproduced statistics.')
if __name__=='__main__':main()
