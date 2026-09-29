import os
import threading
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1280,'height':900})
    page.goto(os.environ.get('REPLAY_URL','http://127.0.0.1:8768/site/'))
    expect(page.locator('.finding')).to_have_count(10,timeout=30000)
    page.locator('[data-view=findings]').click()
    page.locator('.finding').nth(7).get_by_role('button').first.click()
    expect(page.locator('#detail')).to_be_visible()
    first=page.locator('#detail .muted').first.inner_text()
    page.route('**/data/runs/*.json',lambda route:route.fulfill(status=404,body='Missing'))
    page.locator('.run').nth(1).click()
    expect(page.locator('#detailEmpty')).to_contain_text('404')
    expect(page.locator('.run')).to_have_count(15)
    page.unroute('**/data/runs/*.json')
    page.locator('.run').nth(1).click()
    expect(page.locator('#detail')).to_be_visible()
    def delayed(route):
        # Start another selection while the first fetch remains outstanding.
        page.locator('.run').nth(2).click()
        route.continue_()
    page.route('**/data/runs/*.json',delayed,times=1)
    page.locator('.run').nth(0).click()
    expect(page.locator('#detail')).to_be_visible()
    expect(page.locator('.run.selected')).to_have_count(1)
    selected=page.locator('#detail .muted').first.inner_text()
    assert selected!=first
    page.set_viewport_size({'width':390,'height':844})
    page.get_by_role('button',name='Pin for comparison',exact=True).click()
    page.locator('.run').nth(3).click()
    expect(page.get_by_role('button',name='Compare with pinned',exact=True)).to_be_visible()
    page.get_by_role('button',name='Compare with pinned',exact=True).click()
    expect(page.locator('#compareBody article')).to_have_count(2)
    assert page.locator('#comparison').evaluate('(x)=>x.scrollWidth<=x.clientWidth')
    page.get_by_role('button',name='Close comparison').click()
    page.evaluate("compare(all.find(r=>r.source_run_id==='clean-0-350-89-0'), all.find(r=>r.source_run_id==='loss-3-5000-89-0'))")
    expect(page.locator('#compareBody article')).to_have_count(2)
    expect(page.locator('#matchNote')).to_contain_text('Illustrative comparison')
    expect(page.locator('#matchNote')).to_contain_text('deadline ms')
    browser.close()
print('PASS: subdirectory hosting, failed trace/retry, obsolete-fetch guard, mobile comparison.')
