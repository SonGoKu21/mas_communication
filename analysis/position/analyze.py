"""Recompute matched I2/I3/I4 success rates within each model/domain/topology."""
import collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
rows=json.loads((ROOT/'artifacts/tables/position_records.json').read_text())['records']
idx={(r['domain'],r['model'],str(r['task_id']),r['repeat_index'],r['topology'],r['condition']):r for r in rows}
assert len(idx)==len(rows)
conditions=['non_delivery_step2','non_delivery_step3','non_delivery_step4']
groups=collections.defaultdict(list)
for r in rows:
 if r['condition']!='clean':continue
 key=(r['domain'],r['model'],str(r['task_id']),r['repeat_index'],r['topology'])
 faults=[idx.get((*key,c)) for c in conditions]
 if r['success'] is True and all(x is not None and x['applied'] is True and type(x['success']) is bool for x in faults):groups[(r['domain'],r['model'],r['topology'])].append(faults)
result=[]
for (domain,model,topology),units in sorted(groups.items()):
 for i,condition in enumerate(conditions):
  k=sum(x[i]['success'] for x in units);result.append(dict(domain=domain,model=model,topology=topology,condition=condition,n=len(units),successes=k,success_pct=100*k/len(units)))
Path(__file__).with_name('results.json').write_text(json.dumps({'definition':'Matched clean-success units with all three faults applied and known outcomes.','cells':result},indent=2))
print('Recomputed',len(result),'matched position cells.')
