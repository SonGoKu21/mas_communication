"""Use the original Shopping runner under a frozen per-repeat matched budget."""
import argparse,json,os,subprocess,itertools,hashlib
from pathlib import Path
from qwen27_runtime import index_policy,environment
P=argparse.ArgumentParser();P.add_argument('--phase',choices=['pilot','formal'],required=True);P.add_argument('--shard',type=int,default=0);a=P.parse_args()
HERE=Path(__file__).resolve().parent;REPO=HERE.parent;ROOT=Path('/var/lib/mas/results/i3_supplement_20260927/qwen27');PY='python3'
source=json.loads((HERE/'shopping_source_manifest.json').read_text());tasks=source['tasks']
assert len(tasks)==30 and len({t['task_id'] for t in tasks})==30
selected=tasks[:2] if a.phase=='pilot' else tasks[a.shard::3]
assert a.shard in range(3)
policy=index_policy(json.loads((HERE/'frozen_paired_inference_policy.json').read_text()))
out=ROOT/('shopping_pilot' if a.phase=='pilot' else f'shopping_formal_shard{a.shard}');out.mkdir(parents=True,exist_ok=True)
mf={**source,'tasks':selected};(out/'task_manifest.json').write_text(json.dumps(mf,ensure_ascii=False,indent=2)+'\n')
repeats=[1,2] if a.phase=='pilot' else [1,2,3]
cells=[('none',3),('omission',3),('message_corruption',3),('omission',2),('omission',4),('message_corruption',4)]
expected={(t['task_id'],top,f,s,r) for t,top,(f,s),r in itertools.product(selected,['sequential','flat','hierarchical'],cells,repeats)}
metadata={'phase':a.phase,'shard':a.shard,'expected_jobs':len(expected),'policy_sha256':hashlib.sha256((HERE/'frozen_paired_inference_policy.json').read_bytes()).hexdigest(),'task_manifest_sha256':hashlib.sha256((out/'task_manifest.json').read_bytes()).hexdigest(),'original_runner_sha256':hashlib.sha256((REPO/'run_webarena_architecture_rq2.py').read_bytes()).hexdigest(),'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'attempts':1,'stages':[]}
for start,count in [(1,1),(2,1 if a.phase=='pilot' else 2)]:
 env=os.environ.copy();env.update(PYTHONPATH=str(REPO/'src'),LLM_API_KEY='local-qwen')
 setting=environment(policy,'shopping',selected[0]['task_id'],start)
 assert all(environment(policy,'shopping',t['task_id'],start)==setting for t in selected)
 for k,v in setting.items():
  if v is None:env.pop(k,None)
  else:env[k]=v
 common=[PY,'-u',str(REPO/'run_webarena_architecture_rq2.py'),'--base-url','http://localhost:7770','--tasks',str(len(selected)),'--task-manifest',str(out/'task_manifest.json'),'--topologies','sequential,flat,hierarchical','--required-model','Qwen3.8-27B','--llm-run-attempts','1','--resume','--output-dir',str(out),'--run-index-start',str(start),'--runs-per-condition',str(count)]
 for faults,steps in [('none,omission,message_corruption','3'),('omission','2'),('omission,message_corruption','4')]:
  command=common+['--faults',faults,'--steps',steps];record={'environment':setting,'command':command};metadata['stages'].append(record);(out/'paired_run_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
  result=subprocess.run(command,cwd=REPO,env=env);record['returncode']=result.returncode;(out/'paired_run_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
  if result.returncode:raise RuntimeError('Shopping stage failed; outputs retained and no automatic retry')
rows=[json.loads(l) for l in (out/'llm_communication_full.checkpoint.jsonl').read_text().splitlines() if l.strip()]
actual=[(r['task_id'],r['topology'],r['fault_type'],r['injection_step'],r['repeat_index']) for r in rows]
errors=[]
if len(actual)!=len(set(actual)) or set(actual)!=expected:errors.append('coverage/duplicate mismatch')
for row in rows:
 if row['model']!='Qwen3.8-27B':errors.append('model mismatch')
 applied=[e for e in row['events'] if e['fault_applied']]
 if row['fault_type']=='none':
  if applied:errors.append('clean injected')
  continue
 if len(applied)!=1 or applied[0]['step_index']!=row['injection_step']:errors.append('invalid exposure');continue
 event=applied[0]
 if row['fault_type']=='omission':
  if event['delivered_message']!=[]:errors.append('omission delivered')
 else:
  expected_message=dict(event['original_message']);expected_message['observed_quantity']+=1
  if event['delivered_message']!=[expected_message]:errors.append('semantic mutation mismatch')
gate={'passed':not errors,'rows':len(rows),'expected':len(expected),'errors':errors,'model_output_errors':sum(bool(r.get('error')) for r in rows)}
(out/'injection_gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate))
if not gate['passed']:raise RuntimeError('Shopping gate failed')
