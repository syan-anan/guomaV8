import os, sys, io, base64, json
os.chdir("H:/qinglong/syandaV8")
os.environ["XDG_CACHE_HOME"] = "H:/qinglong/syandaV8/__cache"
sys.path.insert(0, ".")
import asyncio
from playwright.async_api import async_playwright
from solver.ocr import solve_ocr
import httpx

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page()
        
        # 访问 Geetest adaptive captcha demo
        await page.goto("https://www.geetest.com/en/adaptive-captcha-demo", wait_until="networkidle", timeout=30000)
        print("Page loaded:", await page.title())
        
        # 截图整个页面
        screenshot = await page.screenshot(full_page=True)
        path = "H:/qinglong/syandaV8/__cache/test_samples/geetest_full.png"
        with open(path, "wb") as f:
            f.write(screenshot)
        print("Screenshot saved:", len(screenshot), "bytes")
        
        # 尝试找 captcha 图片元素
        elements = await page.query_selector_all("img")
        print("Image elements found:", len(elements))
        for el in elements[:5]:
            src = await el.get_attribute("src")
            if src:
                print("  src:", src[:100])
        
        # 尝试点击 captcha 触发
        buttons = await page.query_selector_all("button")
        for btn in buttons:
            text = await btn.inner_text()
            if "captcha" in text.lower() or "demo" in text.lower() or "start" in text.lower():
                print("Clicking:", text)
                await btn.click()
                break
        
        await page.wait_for_timeout(3000)
        
        # 再次截图
        screenshot2 = await page.screenshot(full_page=True)
        path2 = "H:/qinglong/syandaV8/__cache/test_samples/geetest_after.png"
        with open(path2, "wb") as f:
            f.write(screenshot2)
        print("After click screenshot saved")
        
        await browser.close()

asyncio.run(main())
