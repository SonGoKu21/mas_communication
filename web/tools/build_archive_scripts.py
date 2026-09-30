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
    print(f'Generated and byte-verified {count} archive scripts')

if __name__ == '__main__':
    build()
