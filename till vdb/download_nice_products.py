import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://www.nice.org.uk"
START_URL = (
    "https://www.nice.org.uk/guidance/conditions-and-diseases/"
    "eye-conditions/products?ProductType=Advice&Status=Published"
)

OUTPUT_DIR = "nice_products"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; NICE-Advice-PDF-downloader/1.0)"
})


def get_advice_links():
    """Collect all advice product links (should be 9)."""
    resp = session.get(START_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = set()
    for a in soup.select("a[href^='/advice/']"):
        links.add(urljoin(BASE_URL, a["href"]))

    return sorted(links)


def get_pdf_link(advice_url):
    """Extract PDF download link from advice page."""
    resp = session.get(advice_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    pdf_link = soup.find("a", string=lambda s: s and "PDF" in s)
    if not pdf_link:
        return None

    return urljoin(BASE_URL, pdf_link["href"])


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
    advice_links = get_advice_links()
    print(f"Found {len(advice_links)} advice pages")

    for url in tqdm(advice_links, desc="Downloading Advice PDFs"):
        try:
            pdf_url = get_pdf_link(url)
            if pdf_url:
                download_pdf(pdf_url)
        except Exception as e:
            print(f"Failed: {url} -> {e}")


if __name__ == "__main__":
    main()
