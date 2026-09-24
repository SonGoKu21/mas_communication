"""Shared task identity survives filtering, detail loading and evidence downloads."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE=os.environ.get('REPLAY_URL','http://127.0.0.1:8767/')
with sync_playwright() as p:
    b=p.chromium.launch();page=b.new_page(viewport={'width':1440,'height':1000});page.goto(BASE)
    expect(page.locator('.finding')).to_have_count(10,timeout=30000)
    page.locator('[data-view=runs]').click()
    expect(page.locator('.run strong').first).to_have_text('Task 001')
    page.locator('#search').fill('Task 001')
    expect(page.locator('.run strong').first).to_have_text('Task 001')
    assert set(page.locator('.run strong').all_text_contents())=={'Task 001'}
    page.locator('.run').first.click()
    expect(page.locator('#detail h2')).to_have_text('Task 001')
    expect(page.locator('#detail')).to_contain_text('Original ID: shopping-001-q1')
    with page.expect_download() as d:page.get_by_role('button',name='Download sanitized evidence').click()
    data=json.loads(Path(d.value.path()).read_text());assert data['metadata']['task_id']=='shopping-001-q1'
    assert data['metadata']['display_id']=='Task 001'
    page.locator('#search').fill('051')
    expect(page.locator('.run strong').first).to_have_text('Task 051')
    assert set(page.locator('.run strong').all_text_contents())=={'Task 051'}
    page.locator('#search').fill('208')
    expect(page.locator('#resultCount')).not_to_have_text('0 matching records')
    assert any('Original ID: 208' in x for x in page.locator('.run').all_text_contents())
    page.locator('#search').fill('xarray-2905')
    expect(page.locator('#resultCount')).not_to_have_text('0 matching records')
    assert all('pydata__xarray-2905' in x for x in page.locator('.run').all_text_contents())
    mapping=page.evaluate('all.map(r=>({study:r.study,domain:r.domain,task:r.task_id,label:r.display_id}))')
    main=[r for r in mapping if r['study']=='rq123'];assert len({r['label'] for r in main})==100
    identity={}
    for r in main:
        k=(r['domain'],r['task']);assert k not in identity or identity[k]==r['label'];identity[k]=r['label']
    for r in mapping:
        if r['study']=='extension':assert r['label']==identity[(r['domain'],r['task'])]
        elif r['study'].startswith('rq4-'):assert r['label'].startswith('RQ4 Task ')
        elif r['study']=='bridge':assert r['label'].startswith('Fixture ')
    b.close()
print('PASS: stable task identities, ordering, old/new search and preserved source IDs.')
