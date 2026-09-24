"""End-to-end checks against a local static server; no model calls."""
import os
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

BASE=os.environ.get('REPLAY_URL','http://127.0.0.1:8767/')
OUT=Path(__file__).parents[1]/'verification'
OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[];external=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:external.append(r.url) if not r.url.startswith(BASE) else None)
    page.goto(BASE)
    expect(page.locator('.finding')).to_have_count(10,timeout=30000)
    expect(page.locator('#coverage')).to_contain_text('26,613')
    page.screenshot(path=str(OUT/'desktop-findings.png'),full_page=True)
    for i in range(10):
        page.locator('[data-view=findings]').click()
        page.locator('.finding').nth(i).get_by_role('button').first.click()
        expect(page.locator('#detail')).to_be_visible()
        expect(page.locator('#resultCount')).not_to_have_text('0 matching records')
    page.locator('[data-view=findings]').click()
    page.locator('.finding').nth(7).get_by_role('button').first.click()
    expect(page.locator('#detail')).to_be_visible()
    expect(page.locator('#detail')).to_contain_text('shopping-001-q1to2')
    page.screenshot(path=str(OUT/'desktop-replay.png'),full_page=True)
    page.get_by_role('button',name='Next event',exact=True).click()
    expect(page.locator('.playbar')).to_contain_text('2 /')
    page.get_by_role('button',name='Previous event',exact=True).click()
    expect(page.locator('.playbar')).to_contain_text('1 /')
    page.get_by_role('button',name='Pin for comparison',exact=True).click()
    page.locator('.run').nth(1).click()
    expect(page.get_by_role('button',name='Compare with pinned',exact=True)).to_be_visible()
    page.get_by_role('button',name='Compare with pinned',exact=True).click()
    expect(page.locator('#compareBody article')).to_have_count(2)
    page.screenshot(path=str(OUT/'desktop-comparison.png'),full_page=True)
    page.get_by_role('button',name='Close comparison').click()
    with page.expect_download() as download:
        page.get_by_role('button',name='Download sanitized evidence').click()
    data=json.loads(Path(download.value.path()).read_text())
    assert data['metadata']['id'] in page.url
    page.locator('#reset').click()
    page.locator('#search').fill('DOES_NOT_EXIST_123456')
    expect(page.locator('#resultCount')).to_contain_text('0 matching')
    expect(page.locator('#detail')).to_be_hidden()
    page.locator('#reset').click()
    page.locator('#f-result').select_option('Failure')
    assert page.locator('.run .badge').all_text_contents()==['Failure']*15
    page.locator('#next').click()
    expect(page.locator('#pageLabel')).to_contain_text('2 /')
    page.locator('#previous').click()
    page.locator('#tracesOnly').check()
    assert all('summary only' not in t for t in page.locator('.run').all_text_contents())
    page.set_viewport_size({'width':390,'height':844})
    page.locator('[data-view=findings]').click()
    page.screenshot(path=str(OUT/'mobile-findings.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('.finding').nth(3).get_by_role('button').first.click()
    expect(page.locator('#detail')).to_be_visible()
    expect(page.locator('.messages pre')).to_have_count(2)
    page.screenshot(path=str(OUT/'mobile-replay.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    assert not external,external
    browser.close()
print('PASS: catalog, findings, playback, comparison, download, filters, pagination, mobile, no external calls.')
