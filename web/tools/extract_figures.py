"""Rebuild web figures from the locally supplied manuscript; never copy the PDF."""
import argparse
import hashlib
import json
from pathlib import Path
import pymupdf

parser=argparse.ArgumentParser()
parser.add_argument('pdf',type=Path)
args=parser.parse_args()
out=Path(__file__).resolve().parents[1]/'site/figures'
spec=json.loads((out/'source.json').read_text())
if hashlib.sha256(args.pdf.read_bytes()).hexdigest()!=spec['paper_sha256']:
    raise SystemExit('PDF digest differs from the reviewed manuscript revision.')
with pymupdf.open(args.pdf) as doc:
    for number,entry in spec['figures'].items():
        page=doc[entry['pdf_page']-1]
        if entry.get('exclude_text_points'):
            # Remove only a neighboring paragraph fragment outside the diagram.
            page.add_redact_annot(pymupdf.Rect(entry['exclude_text_points']),fill=(1,1,1))
            page.apply_redactions(images=0,graphics=0)
        page.get_pixmap(matrix=pymupdf.Matrix(5,5),clip=pymupdf.Rect(entry['clip_points']),alpha=False).save(out/f'fig-{number}.png')
print('Extracted reviewed figure regions; manuscript PDF not copied.')
