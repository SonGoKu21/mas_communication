"""Rebuild web figures from the locally supplied manuscript; never copy the PDF."""
import argparse
import hashlib
import json
from pathlib import Path
import pymupdf

parser=argparse.ArgumentParser()
parser.add_argument('pdf',type=Path)
parser.add_argument('--figure-2-pdf',type=Path,help='Author-supplied replacement layered_propagation.pdf')
args=parser.parse_args()
out=Path(__file__).resolve().parents[1]/'site/figures'
spec=json.loads((out/'source.json').read_text())
if hashlib.sha256(args.pdf.read_bytes()).hexdigest()!=spec['paper_sha256']:
    raise SystemExit('PDF digest differs from the reviewed manuscript revision.')
replacement=spec['figures']['2'].get('external_source')
if replacement and (not args.figure_2_pdf or hashlib.sha256(args.figure_2_pdf.read_bytes()).hexdigest()!=spec['figures']['2']['source_sha256']):
    raise SystemExit('Supply the reviewed replacement with --figure-2-pdf; refusing to restore the old Figure 2.')
with pymupdf.open(args.pdf) as doc:
    for number,entry in spec['figures'].items():
        if entry.get('external_source'):
            with pymupdf.open(args.figure_2_pdf) as replacement_doc:
                replacement_doc[entry['pdf_page']-1].get_pixmap(matrix=pymupdf.Matrix(entry['scale'],entry['scale']),clip=pymupdf.Rect(entry['clip_points']),alpha=False).save(out/entry.get('output_file',f'fig-{number}.png'))
            continue
        page=doc[entry['pdf_page']-1]
        if entry.get('exclude_text_points'):
            # Remove only a neighboring paragraph fragment outside the diagram.
            page.add_redact_annot(pymupdf.Rect(entry['exclude_text_points']),fill=(1,1,1))
            page.apply_redactions(images=0,graphics=0)
        page.get_pixmap(matrix=pymupdf.Matrix(5,5),clip=pymupdf.Rect(entry['clip_points']),alpha=False).save(out/entry.get('output_file',f'fig-{number}.png'))
print('Extracted reviewed figure regions; manuscript PDF not copied.')
