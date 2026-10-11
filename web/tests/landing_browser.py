"""Check the paper overview and preservation of detailed evidence."""
import os,re,json,subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
BASE=os.environ.get('REPLAY_URL','http://127.0.0.1:8767/')
results=Path('web/site/results.html').read_text()
assert results.count('class="home-section home-rq"') == 4
for n in range(1,5):
 assert 'id="home-rq-'+str(n)+'"' in results
assert results.count('class="case-explanation"') == 4
findings=json.loads(Path('web/paper-findings.json').read_text())
with sync_playwright() as p:
 b=p.chromium.launch();page=b.new_page(viewport={'width':1440,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(BASE);expect(page.locator('#homeView h1')).to_have_text('Broken Telephone');expect(page.locator('#homeView img')).to_have_count(0)
 expect(page.locator('#abstract')).to_contain_text('20.7%');expect(page.locator('.overview-rq')).to_have_count(4);expect(page.locator('.overview-finding')).to_have_count(7)
 for f in findings:expect(page.locator('#overview-rq-'+str(f['rq']))).to_contain_text(f['title'])
 page.get_by_role('link',name='Experimental details for RQ3 →',exact=True).click();expect(page).to_have_url(re.compile('results.html#home-rq-3'));expect(page.locator('#homeView .home-finding')).to_have_count(7,timeout=30000);expect(page.locator('#homeView img')).to_have_count(11)
 page.get_by_role('link',name='Home',exact=True).click();page.locator('[data-home-view="runs"]').click();expect(page.locator('#runsView')).to_be_visible()
 page.goto(BASE+'#home-rq-3');expect(page).to_have_url(re.compile('results.html#home-rq-3'))
 page.goto(BASE);page.screenshot(path='web/verification/landing-desktop.png',full_page=True);page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth');page.screenshot(path='web/verification/landing-mobile.png',full_page=True)
 # Overview does not depend on loading the large archive.
 plain=b.new_page();plain.route('**/data/*.json',lambda route:route.abort());plain.goto(BASE);expect(plain.locator('.overview-finding')).to_have_count(7);expect(plain.locator('#abstract')).to_be_visible();plain.close()
 assert not errors,errors;b.close()
print('PASS: abstract, four RQs, seven findings, full detail preservation, links and mobile layout')
