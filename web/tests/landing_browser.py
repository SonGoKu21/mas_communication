"""Verify the concise home and preservation of the complete results page."""
import os,re,subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE=os.environ.get('REPLAY_URL','http://127.0.0.1:8767/')
previous=subprocess.check_output(['git','show','HEAD:web/site/index.html']).decode()
results=Path('web/site/results.html').read_text()
for section in re.findall(r'<section class="home-section home-rq".*?</section>',previous):
 assert section in results,'An RQ section lost content during relocation'
with sync_playwright() as p:
 b=p.chromium.launch();page=b.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(BASE);expect(page.locator('#homeView h1')).to_have_text('Broken Telephone');expect(page.locator('#homeView img')).to_have_count(1);expect(page.locator('#homeView .home-finding')).to_have_count(0)
 page.get_by_role('link',name='Explore results and cases →').click();expect(page.locator('#homeView .home-finding')).to_have_count(7,timeout=30000);expect(page.locator('#homeView img')).to_have_count(10)
 page.get_by_role('link',name='Home',exact=True).click();page.get_by_role('button',name='Inspect archived runs →').click();expect(page.locator('#runsView')).to_be_visible()
 page.goto(BASE+'#home-rq-3');expect(page).to_have_url(re.compile('results.html#home-rq-3'))
 page.goto(BASE);page.screenshot(path='web/verification/landing-desktop.png',full_page=True);page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path='web/verification/landing-mobile.png',full_page=True)
 assert not errors,errors;b.close()
print('PASS: concise home, full RQ content preservation, result navigation, old anchors and mobile layout')
