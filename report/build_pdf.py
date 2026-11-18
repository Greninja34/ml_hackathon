"""Render report/report.html to report/PathSnap_Technical_Report.pdf with headless Chrome (Playwright)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

here = Path(__file__).resolve().parent
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page()
    pg.goto((here / "report.html").as_uri(), wait_until="networkidle")
    pg.pdf(path=str(here / "PathSnap_Technical_Report.pdf"), format="A4", print_background=True, display_header_footer=True,
           header_template="<span></span>",
           footer_template='<div style="width:100%;font-size:8px;color:#888;text-align:center;font-family:Arial">PathSnap technical report · page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
           margin=dict(top="16mm", bottom="18mm", left="16mm", right="16mm"))
    b.close()
print("written", here / "PathSnap_Technical_Report.pdf")
