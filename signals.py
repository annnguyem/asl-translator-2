# signals.py
import os
import re
import string
from functools import lru_cache
from urllib.parse import urljoin

import requests

_SIGNASL_BASES = ("https://www.signasl.org/", "https://signasl.org/")
_USE_BROWSER = os.getenv("USE_BROWSER", "0").lower() in ("1", "true", "yes")

def _strip_punct(t: str) -> str:
    return (t or "").translate(str.maketrans("", "", string.punctuation)).lower().strip()

def _browser_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.signasl.org/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    return s

def _fetch_signasl_urls_http(token: str) -> list[str]:
    token = _strip_punct(token)
    if not token:
        return []
    sess = _browser_session()
    found: list[str] = []

    # JSON API (if exposed)
    for base in _SIGNASL_BASES:
        api = urljoin(base, f"api/sign/{token}")
        try:
            rj = sess.get(api, timeout=8, allow_redirects=True)
            if rj.ok:
                data = rj.json()
                if isinstance(data, list):
                    for item in data:
                        u = (item or {}).get("video_url")
                        if u:
                            found.append(u)
        except Exception:
            pass

    # HTML scrape (mp4/webm/m3u8)
    attr_re = re.compile(
        r'(?:src|data-src|srcset|data-video|data-hls)=["\']([^"\']+?\.(?:mp4|webm|m3u8)(?:\?[^"\']*)?)["\']',
        re.IGNORECASE,
    )
    abs_re = re.compile(r'https?://[^\s"\'<>]+?\.(?:mp4|webm|m3u8)\b', re.IGNORECASE)

    for base in _SIGNASL_BASES:
        page = urljoin(base, f"sign/{token}")
        try:
            rh = sess.get(page, timeout=12, allow_redirects=True)
            if not rh.ok:
                continue
            html = rh.text
            for m in attr_re.findall(html):
                found.append(urljoin(base, m))
            for m in abs_re.findall(html):
                found.append(m)
        except Exception:
            pass

    # de-dupe
    seen, out = set(), []
    for u in found:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def _fetch_signasl_urls_browser(token: str) -> list[str]:
    if not _USE_BROWSER:
        return []
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return []

    token = _strip_punct(token)
    if not token:
        return []

    hits: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            page = ctx.new_page()

            # Sniff media responses (mp4/webm/m3u8, or video content-types)
            def on_response(resp):
                try:
                    url = resp.url
                    if url.startswith("blob:"):
                        return
                    ct = (resp.headers or {}).get("content-type", "").lower()
                    if (
                        any(ext in url.lower() for ext in (".mp4", ".webm", ".m3u8"))
                        or "video/" in ct
                        or "vnd.apple.mpegurl" in ct
                        or "mpegurl" in ct
                    ):
                        hits.append(url)
                except Exception:
                    pass

            page.on("response", on_response)

            for base in _SIGNASL_BASES:
                url = urljoin(base, f"sign/{token}")
                try:
                    page.goto(url, wait_until="networkidle", timeout=20000)
                    page.wait_for_timeout(800)

                    # DOM sources as backup
                    dom_urls = page.eval_on_selector_all(
                        "video, video source, source",
