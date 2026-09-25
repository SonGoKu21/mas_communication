"""Check extended-method navigation, evidence links and narrow-screen reading."""
import os
from playwright.sync_api import sync_playwright, expect
BASE=os.environ.get('REPLAY_URL','http://127.0.0.1:8767/')
with sync_playwright() as p:
 b=p.chromium.launch();page=b.new_page(viewport={'width':1440,'height':1000});errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)));page.goto(BASE)
 page.get_by_role('link',name='Methodology',exact=True).click()
 expect(page.locator('h1')).to_have_text('Extended methodology')
 for section in ['taxonomy','boundaries','operators','evaluation','walkthrough','protocol']:
  expect(page.locator('#'+section)).to_be_visible()
 assert page.locator('#taxonomy tbody tr').count()==29
 expect(page.locator('#protocol')).to_contain_text('18,900')
 expect(page.locator('#walkthrough')).to_contain_text('hollister, Joust Bag')
 page.get_by_role('link',name='Open fault run',exact=True).click()
 expect(page.locator('#detail')).to_contain_text('admin-main-sequential-41-r131-044af0c8',timeout=30000)
 page.goto(BASE+'methodology.html');page.set_viewport_size({'width':390,'height':844})
 assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
 page.screenshot(path='web/verification/methodology-mobile.png',full_page=True)
 page.set_viewport_size({'width':1440,'height':1000});page.screenshot(path='web/verification/methodology-desktop.png',full_page=True)
 assert not errors,errors
 b.close()
print('PASS: methodology sections, taxonomy, exact evidence link and mobile layout.')
