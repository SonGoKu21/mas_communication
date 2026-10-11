"""Generate opaque-origin compatible data scripts without changing evidence."""
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'site'

def build():
    count = 0
    for source in sorted((ROOT / 'data').rglob('*.json')):
        raw = source.read_bytes()
        json.loads(raw)  # Reject malformed source evidence.
        key = source.relative_to(ROOT).as_posix()
        payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii')
        target = ROOT / 'archive' / source.relative_to(ROOT / 'data').with_suffix('.js')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'MASArchive.receive({json.dumps(key)},{json.dumps(payload)});\n')
        assert gzip.decompress(base64.b64decode(payload)) == raw
        count += 1
    # One startup request avoids anonymous-host rate limits across catalog shards.
    names = ['manifest.json', 'findings.json', 'task-map.json']
    names += [Path(n).name for n in json.loads((ROOT / 'data/manifest.json').read_text())['catalogs']]
    bootstrap = {'data/' + n: json.loads((ROOT / 'data' / n).read_text()) for n in names}
    raw = json.dumps(bootstrap, separators=(',', ':')).encode()
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode('ascii')
    (ROOT / 'archive/bootstrap.js').write_text(f'MASArchive.receive("data/bootstrap.json",{json.dumps(payload)});\n')
    assert json.loads(gzip.decompress(base64.b64decode(payload))) == bootstrap
    print(f'Generated and byte-verified {count} archive scripts plus startup bundle')

if __name__ == '__main__':
    build()
