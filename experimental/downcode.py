import os
import time

from playwright.sync_api import sync_playwright

browser = None
page = None
playwright = None
download_dir = "/Users/jj/tmp"


def init_browser(headless=False):
    """Initialize the browser and page globally."""
    os.makedirs(download_dir, exist_ok=True)
    global browser, page, playwright
    
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=headless, downloads_path=download_dir)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    return page


def close_browser():
    """Close the browser and cleanup."""
    global browser, page, playwright
    if page:
        page.close()
    if browser:
        browser.close()
    if playwright:
        playwright.stop()


def extract_links(url):
    """Extract all links from the 'browse-columns' div."""
    page.goto(url)
    links = page.evaluate('''() => {
        const div = document.querySelector('div.browse-columns.roboto');
        if (!div) return [];
        const anchors = div.querySelectorAll('a.browse-link.roboto');
        return Array.from(anchors).map(a => ({
            text: a.innerText,
            href: a.href
        }));
    }''')
    return links


def follow_link(links, link_text):
    """Follow the link with the given text."""
    target_link = next((link for link in links if link['text'] == link_text), None)
    if not target_link:
        raise ValueError(f"No link found with text: {link_text}")
    page.goto(target_link['href'])
    return page.url

def click_download():
    # 1. Click the download button that opens the modal
    page.locator('button.btn.btn-white-circle[title="Download"]').click()
    time.sleep(1)
    
    # 2. Wait for the popup to appear and select all checkboxes
    # (Assuming checkboxes are inside the popup; adjust selector as needed)
    page.wait_for_selector('input[type="checkbox"]', state='visible', timeout=5000)
    #page.evaluate('''() => {
    #    document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
    #}''')
    page.locator('input[type="checkbox"]').first.click()
    time.sleep(1)

    # 3. Click download button in the modal
    try:
        page.wait_for_selector('#download-button:not([disabled])', timeout=5000)
        page.locator('#download-button').click()
    except Exception as e:
        print("Download button is still disabled. Checking for errors...")
        # Log the button's state and surrounding HTML for debugging
        print(page.locator('#download-button').get_attribute('disabled'))
        print(page.locator('#download-button').get_attribute('class'))
        print(page.content())  # Log the modal's HTML
        raise e
    
    # 4. Click "Save Word" button
    page.get_by_role('button', name='Save Word').click()

    # Wait for the download link and click it
    with page.expect_download() as download_info:
        page.get_by_role('link', name='Opens exported or printable').click(timeout=120000)
    download = download_info.value
    download_path = os.path.join(download_dir, download.suggested_filename)
    download.save_as(download_path)
    #download.save_as("/Users/jj/tmp/foo")
    print(f"Downloaded to: {download_path}")
    return download


def select_link_by_number(links):
    if not links:
        print("No links available.")
        return None

    print("Available links:")
    for i, link in enumerate(links, start=1):
        print(f"{i}. {link['text']}")

    while True:
        try:
            selection = input("Enter the number of the link you want to select: ")
            index = int(selection) - 1
            if 0 <= index < len(links):
                return links[index]['text']
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":
    try:
        init_browser(headless=False)
        base_url = "https://codelibrary.amlegal.com"
        state_links = extract_links(base_url)
        
        state = select_link_by_number(state_links)
        # state = "Alaska"

        state_url = follow_link(state_links, state)
        print(f"Followed '{state}' to: {state_url}")
        town_links = extract_links(state_url)

        town = select_link_by_number(town_links)    
        #town = "King Cove"
        
        town_url = follow_link(town_links, town)
        print(f"Followed '{town}' to: {town_url}")

        click_download()
    finally:
        close_browser()
