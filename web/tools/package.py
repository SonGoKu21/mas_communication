import json
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
site=root/'site'
manifest=json.loads((site/'data/manifest.json').read_text())
upload=root.parent/'mas_fault_static_upload.zip'
with zipfile.ZipFile(upload,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(site.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(site))
    z.write(root/'README.md','DEPLOYMENT.md')
source=root.parent/'mas_fault_static_source.zip'
with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for directory in ['tools','tests']:
        for p in sorted((root/directory).glob('*.py')): z.write(p,p.relative_to(root))
    for p in site.iterdir():
        if p.is_file(): z.write(p,p.relative_to(root))
    for name in ['README.md','LEDGER.md','VERIFICATION.md','paper-findings.json']: z.write(root/name,name)
    z.write(site/'data/paper-alignment.json','site/data/paper-alignment.json')
with zipfile.ZipFile(upload) as z:
    assert z.testzip() is None
    assert 'index.html' in z.namelist()
    assert not any('.venv' in n or '__pycache__' in n for n in z.namelist())
print(f'Upload: {upload.name}, {upload.stat().st_size/1024**2:.1f} MiB')
print(f'Source: {source.name}, {source.stat().st_size/1024:.1f} KiB')
