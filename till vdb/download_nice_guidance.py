import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from tqdm import tqdm

BASE_URL = "https://www.nice.org.uk"
START_URL = (
    "https://www.nice.org.uk/guidance/conditions-and-diseases/"
    "eye-conditions/products?ProductType=Guidance&Status=Published"
)

OUTPUT_DIR = "nice_eye_condition_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; NICE-PDF-downloader/1.0)"
})


def get_all_guidance_links():
    """Follow pagination and collect all guidance links."""
    links = set()
    next_url = START_URL

    while next_url:
        resp = session.get(next_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # capture ALL guidance types
        for a in soup.select("a[href^='/guidance/']"):
            href = a.get("href")
            if any(href.startswith(f"/guidance/{p}") for p in ("ta", "ng", "ipg", "mtg", "dg", "hst")):
                links.add(urljoin(BASE_URL, href))

        # find pagination link
        next_link = soup.select_one("a[rel='next']")
        next_url = urljoin(BASE_URL, next_link["href"]) if next_link else None

    return sorted(links)


def get_pdf_link(guidance_url):
    """Extract the PDF download link from a guidance page."""
    resp = session.get(guidance_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pdf_button = soup.find("a", string=lambda s: s and "Download guidance" in s)
    if not pdf_button:
        return None

    return urljoin(BASE_URL, pdf_button["href"])


def download_pdf(pdf_url):
    filename = pdf_url.split("/")[-1] + ".pdf"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        return

    with session.get(pdf_url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


def main():
    guidance_links = get_all_guidance_links()
    print(f"Found {len(guidance_links)} guidance pages")

    for url in tqdm(guidance_links, desc="Downloading PDFs"):
        try:
            pdf_url = get_pdf_link(url)
            if pdf_url:
                download_pdf(pdf_url)
        except Exception as e:
            print(f"Failed: {url} -> {e}")


if __name__ == "__main__":
    main()
