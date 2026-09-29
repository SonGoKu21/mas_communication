"""Independent raw-input recount of every exported contrast and failure overlap."""
import collections,hashlib,itertools,json
from pathlib import Path
H=Path(__file__).resolve().parent;d=json.loads((H/'results.json').read_text());p=H.parents[1]/d['input_file'];raw=p.read_bytes();source=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==d['input_sha256']
R=source['records'];idx={(x['domain'],x['task_id'],x['repeat_index'],x['condition'],x['model'],x['topology']):x for x in R};assert len(idx)==16200
cfg=list(itertools.product(('deepseek','qwen9','qwen27'),('sequential','hierarchical','flat')))
unitkeys=sorted({(x['domain'],x['task_id'],x['repeat_index'],x['condition']) for x in R if x['condition']!='clean'})
selected={c:[] for c in ('common_applied','all_known','clean_success_common_applied')}
for u in unitkeys:
 r=[idx[(*u,m,t)] for m,t in cfg]
 if not all(type(x['success']) is bool for x in r):continue
 selected['all_known'].append(u)
 if not all(x['applied'] is True for x in r):continue
 selected['common_applied'].append(u)
 clean=[idx[(*u[:3],'clean',m,t)] for m,t in cfg]
 if all(x['success'] is True for x in clean):selected['clean_success_common_applied'].append(u)
assert {c:len(u) for c,u in selected.items()}==dict(common_applied=1408,all_known=1497,clean_success_common_applied=1053)
def group(x):return [u for u in selected[x['cohort']] if (x['domain']=='ALL_DOMAINS' or u[0]==x['domain']) and (x['condition']=='ALL_FIVE_FAULTS' or u[3]==x['condition'])]
for x in d['contrasts']:
 g=group(x);bm=x['baseline_model'];bt=x['baseline_topology'];om=x['alternative_model']
 states={kind:collections.Counter() for kind in ('flat_change','model_change')};over=collections.Counter()
 for u in g:
  b=idx[(*u,bm,bt)]['success'];f=idx[(*u,bm,'flat')]['success'];m=idx[(*u,om,bt)]['success']
  states['flat_change'][str(int(b))+str(int(f))]+=1;states['model_change'][str(int(b))+str(int(m))]+=1
  if not b:over['both' if f and m else 'flat_only' if f else 'model_only' if m else 'neither']+=1
 for name,s in states.items():
  v=x[name];assert v['n']==len(g) and all(v['four_states'][k]==s[k] for k in ('00','01','10','11'))
  assert v['rescued']==s['01'] and v['regressed']==s['10'];assert v['baseline_failures']==s['00']+s['01']
  if g:assert abs(v['net_pp']-100*(s['01']-s['10'])/len(g))<1e-10
 assert all(x['rescue_overlap'][k]==over[k] for k in ('both','flat_only','model_only','neither'))
 if 'bootstrap' in x:
  assert x['domain']=='ALL_DOMAINS' and x['condition']=='ALL_FIVE_FAULTS'
  for val in x['bootstrap'].values():
   assert val['draws']==10000 and 0<=val['valid_draws']<=10000
   if val['ci95'] is not None:assert val['ci95'][0]<=val['ci95'][1]
for x in d['failure_overlap']:
 g=group(x);sets={m:{u for u in g if idx[(*u,m,x['topology'])]['success'] is False} for m in ('deepseek','qwen9','qwen27')}
 for z in x['pairwise']:
  a,b=sets[z['left']],sets[z['right']];union=a|b
  assert z['intersection']==len(a&b) and z['union']==len(union)
  assert z['jaccard']==(len(a&b)/len(union) if union else None)
 assert x['three_model_intersection']==len(set.intersection(*sets.values()))
 assert x['three_model_union']==len(set.union(*sets.values()))
report=dict(passed=True,input_sha256=d['input_sha256'],input_rows=16200,unique_rows=16200,unknown_rows=sum(x['success'] is None for x in R),unit_count=1500,contrasts_independently_recounted=len(d['contrasts']),failure_overlap_groups_independently_recounted=len(d['failure_overlap']),cohort_sizes={k:len(v) for k,v in selected.items()},model_or_server_calls=0)
(H/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
