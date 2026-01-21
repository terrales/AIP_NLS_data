"""
Convert HTML to PNG using playwright
"""
import asyncio
import sys

async def html_to_png(html_file, output_file, width=1400, height=800):
    """Convert HTML file to PNG using playwright."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: playwright not installed. Trying alternative method...")
        return False

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': width, 'height': height})

        # Load HTML file
        with open(html_file, 'r') as f:
            html_content = f.read()
        await page.set_content(html_content)

        # Wait for content to load
        await asyncio.sleep(1)

        # Take screenshot
        await page.screenshot(path=output_file, full_page=True)
        await browser.close()

        print(f"✓ Converted {html_file} to {output_file}")
        return True

if __name__ == '__main__':
    html_file = 'elo_comparison_report.html'
    output_file = 'elo_comparison_report.png'

    success = asyncio.run(html_to_png(html_file, output_file))

    if not success:
        print("\nAlternative: You can open the HTML file in a browser and take a screenshot")
        print(f"File location: {html_file}")
        sys.exit(1)
