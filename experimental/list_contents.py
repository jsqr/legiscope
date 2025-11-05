from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://codelibrary.amlegal.com")

    # Extract all links inside the 'browse-columns' div
    links = page.evaluate('''() => {
        const div = document.querySelector('div.browse-columns.roboto');
        if (!div) return [];
        const anchors = div.querySelectorAll('a.browse-link.roboto');
        return Array.from(anchors).map(a => ({
            text: a.innerText,
            href: a.href
        }));
    }''')

    print(links)  # List of dictionaries: [{'text': 'Alaska', 'href': '/regions/ak'}, ...]
    browser.close()
