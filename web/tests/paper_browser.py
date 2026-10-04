"""Catch wrong evidence selection and mismatched RQ grouping in the published UI."""
import os
from playwright.sync_api import sync_playwright, expect
BASE=os.environ.get('REPLAY_URL','http://127.0.0.1:8767/')
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE+"results.html")
    expect(page.locator('.finding')).to_have_count(7,timeout=30000)
    page.locator('[data-view=findings]').click()
    expected={1:[1,2],2:[3,4],3:[5,6],4:[7]}
    for rq,nums in expected.items():
        group=page.locator(f'[data-rq="{rq}"]')
        expect(group.locator('.finding')).to_have_count(len(nums))
    page.get_by_role('button',name='Fig. 6 · I2 request loss',exact=True).click()
    expect(page.locator('#detail')).to_contain_text('WebArena-Reddit-67-flat-non_delivery_step2-r1')
    expect(page.locator('.run.selected')).to_contain_text('non delivery step2')
    page.locator('[data-view=findings]').click()
    page.get_by_role('button',name='Fig. 9 · Sequential',exact=True).click()
    expect(page.locator('#detail')).to_contain_text('DeepSeek-V4-Flash')
    expect(page.locator('#detail')).to_contain_text('WebArena-Reddit-67-sequential-non_delivery_step4-r2')
    assert not errors,errors
    browser.close()
print('PASS: four RQs, seven findings, exact figure-case evidence links.')
