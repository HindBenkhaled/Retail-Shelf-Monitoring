"""Local Edge/Playwright integration check; saves real rendered screenshots."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
from scripts.common import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:5000')
    p.add_argument('--image', default='data/sku110k/images/test/test_00000.jpg')
    p.add_argument('--expected-mode', choices=['hybrid','classical_fallback'], required=True)
    args = p.parse_args()
    dest = Path('docs/screenshots'); dest.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width':1440,'height':1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.locator('#health')).to_contain_text('ready')
        page.screenshot(path=str(dest/'upload.png'), full_page=True)
        page.locator('#image-input').set_input_files(args.image)
        expect(page.locator('#preview')).to_be_visible()
        with page.expect_response('**/api/analyze', timeout=120000) as response:
            page.get_by_role('button', name='Analyze shelf').click()
        data = response.value.json()
        assert response.value.status == 200, data
        assert data['mode'] == args.expected_mode, data
        expect(page.locator('#output')).to_be_visible()
        page.locator('#output').evaluate('(img) => img.decode()')
        expect(page.locator('#gaps')).to_have_text(str(data['gap_count']))
        assert page.request.get(args.url + data['annotated_image_url']).status == 200
        page.screenshot(path=str(dest/f'{args.expected_mode}-local.png'), full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(dest/f'{args.expected_mode}-mobile-local.png'), full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile overflow'
        assert not errors, errors
        save_json(f'results/browser_{args.expected_mode}.json', {'status':'passed', 'browser':browser.version,
            'checks':['upload','preview','analysis POST','mode','statistics','annotated retrieval','mobile overflow','no JS errors'],
            'analysis': data})
        browser.close()
    print(json.dumps({'status':'passed','mode':data['mode'],'gaps':data['gap_count'],'products':data['products_detected']}))


if __name__ == '__main__':
    main()
