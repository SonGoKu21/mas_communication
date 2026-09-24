import json
import re
from pathlib import Path

site=Path(__file__).resolve().parents[1]/'site'
m=json.loads((site/'data/manifest.json').read_text())
rows=[r for f in m['catalogs'] for r in json.loads((site/f).read_text())]
assert len(rows)==26613==m['total']
assert len({r['id'] for r in rows})==len(rows)
assert sum(bool(r['trace_path']) for r in rows)==m['detailed']
ids={r['id'] for r in rows}
for f in json.loads((site/'data/findings.json').read_text()):
    assert f['count'] and f['examples'],f['id']
    assert set(f['examples'])<=ids
    for case in f.get('cases',[]):
        matching=[r for r in rows if all(r.get(k)==v for k,v in case['query'].items())]
        assert len(matching)==1,case['label']
        assert matching[0]['id']==case['run_id'] and matching[0]['trace_path'],case['label']
for r in rows:
    if r['trace_path']:
        path=site/r['trace_path'];d=json.loads(path.read_text())
        assert d['metadata']['id']==r['id']
        assert isinstance(d['events'],list)
for path in (site/'data').rglob('*.json'):
    text=path.read_text()
    assert not re.search(r'/Users/|/home/hqn|/data[23]/|\bsk-[A-Za-z0-9_-]{12,}|Bearer\s+[a-zA-Z0-9]|10\.102\.35\.120|202\.117\.43\.5',text),path
print('PASS: counts, unique IDs, trace references, finding references, source privacy patterns.')
