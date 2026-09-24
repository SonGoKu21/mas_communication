import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'rq123_six_condition_inputs_20260916_v2/analysis_config.json'
groups = collections.defaultdict(lambda: [0, 0])
strata = collections.defaultdict(lambda: [0, 0, set()])
matched = []
for exp in json.loads(CONFIG.read_text())['experiments']:
    rows = []
    for path in exp['paths']:
        with open(path) as source:
            for line, raw in enumerate(source, 1):
                r = json.loads(raw)
                answer = r.get('final_answer') or {}
                sem = r.get('semantic_evaluator_evidence') or {}
                rows.append(dict(task=str(r['task_id']), graph=r['topology'],
                                 rep=r['repeat_index'], condition=r['condition'],
                                 applied=r.get('fault_applied'), success=r['final_task_success'],
                                 stratum=r.get('task_stratum'), answer=answer,
                                 expected=r.get('expected_answer'), missing=sem.get('missing_required_fields'),
                                 run_id=r['run_id'], line=line, file=path))
    clean = {(r['task'], r['graph'], r['rep']): r['success'] for r in rows if r['condition'] == 'clean'}
    for r in rows:
        if r['condition'] == 'clean' or not r['applied'] or not clean.get((r['task'], r['graph'], r['rep'])):
            continue
        r.update(domain=exp['dataset'], model=exp['label'])
        matched.append(r)
        for key in [(r['domain'], r['graph'], r['condition']),
                    (r['domain'], r['model'], r['graph'], r['condition'])]:
            groups[key][0] += 1
            groups[key][1] += not r['success']
        if r['stratum']:
            key = (r['model'], r['graph'], r['stratum'], r['condition'])
            v = strata[key]
            v[0] += 1
            v[1] += not r['success']
            v[2].add(r['task'])

result = dict(scope='Applied fault runs with successful task/model/topology/repetition-matched clean; descriptive only.',
              groups=[dict(key=k, n=v[0], failures=v[1]) for k, v in sorted(groups.items())],
              admin_strata=[dict(key=k, n=v[0], failures=v[1], tasks=sorted(v[2])) for k, v in sorted(strata.items())])
out = ROOT / 'domain_pattern_audit.json'
out.write_text(json.dumps(result, indent=2))
print('MATCHED',len(matched),'OUTPUT',out)
for g in ['flat','sequential','hierarchical']:
    print('\nGRAPH',g)
    for k,v in sorted(groups.items()):
        if len(k)==3 and k[1]==g:
            print(k[0],k[2],f'{v[1]}/{v[0]} ({100*v[1]/v[0]:.1f}%)')
print('\nMODEL RANGE FLAT')
for domain in sorted({r['domain'] for r in matched}):
    for cond in sorted({r['condition'] for r in matched}):
        vals=[(k[1],round(100*v[1]/v[0],1),v) for k,v in groups.items() if len(k)==4 and k[0]==domain and k[2]=='flat' and k[3]==cond]
        print(domain,cond,vals)
print('\nADMIN FLASH FLAT STRATA: SEMANTIC AND I2')
for k,v in sorted(strata.items()):
    if k[0]=='DeepSeek-V4-Flash' and k[1]=='flat' and k[3] in ['semantic_corruption_step4','non_delivery_step2']:
        print(k[2:],v[:2],sorted(v[2]))
print('\nPARTIAL: FAILED BUT ALL MUST_INCLUDE ANSWER STRINGS PRESENT')
for domain in ['WebArena Reddit','WebArena Admin']:
    rows=[r for r in matched if r['domain']==domain and r['condition']=='valid_partial_message_step4']
    known=[r for r in rows if isinstance(r['expected'],dict) and r['expected'].get('must_include') and isinstance(r['answer'],dict)]
    hits=[r for r in known if not r['success'] and all(str(s).casefold() in str(r['answer'].get('answer','')).casefold() for s in r['expected']['must_include'])]
    print(domain,'total',len(rows),'must_include_evaluable',len(known),'failed_but_strings_present',len(hits))
