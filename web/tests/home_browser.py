import os
from playwright.sync_api import sync_playwright,expect
BASE=os.environ.get('REPLAY_URL','http://127.0.0.1:8767/')
with sync_playwright() as p:
 b=p.chromium.launch();page=b.new_page(viewport={'width':1440,'height':1000});page.goto(BASE)
 expect(page.locator('#homeView')).to_be_visible()
 expect(page.locator('#homeView h1')).to_contain_text('When Agents Miscommunicate')
 expect(page.locator('#homeView .home-finding')).to_have_count(10,timeout=30000)
 assert page.locator('#homeView img').count()>=7
 expect(page.locator('.case-explanation')).to_have_count(4)
 expect(page.locator('img[src*="fig-2-layered-"]')).to_have_count(1)
 expect(page.locator('[aria-label="Figure 10 case explanation"]')).to_contain_text('exact paired runs are not bundled')
 for d in page.locator('#homeView details').all():d.evaluate('(d)=>d.open=true')
 for img in page.locator('#homeView img').all():
  img.scroll_into_view_if_needed();expect(img).to_be_visible();assert img.evaluate('(i)=>i.complete && i.naturalWidth>0')
 page.get_by_role('button',name='Explore RQ2 evidence',exact=True).click();expect(page.locator('#findingsView')).to_be_visible()
 page.locator('[data-view=home]').click();page.locator('#homeView .figure-open').first.click();expect(page.locator('#figureDialog')).to_be_visible();page.get_by_role('button',name='Close figure',exact=True).click()
 page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.screenshot(path='web/verification/home-mobile.png',full_page=True)
 page.set_viewport_size({'width':1440,'height':1000});page.screenshot(path='web/verification/home-desktop.png',full_page=True)
 slow=b.new_page();slow.route('**/data/catalog-00.json',lambda route:None);slow.goto(BASE,wait_until='domcontentloaded');expect(slow.get_by_role('button',name='Explore RQ2 evidence',exact=True)).to_be_disabled();slow.close()
 b.close()
print('PASS: home default, ten findings, figures, evidence navigation, zoom and mobile layout.')
