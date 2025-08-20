# signals.py
"""
SignASL clip finder.

- fetch_signasl_urls(token)  -> list[str] of direct media URLs (mp4/webm/m3u8)
- lookup_sign_urls_for_word(word) -> 1-2 good URLs for a word, else fingerspelling
- translate_text_to_sign(sentence) -> URLs for a whole sentence

Enable headless browser fallback by setting env: USE_BROWSER=1
(Requires Playwright + Chromium installed in your deploy.)
"""
from __future__ import annotations

import os
import re
import string
from functools import lru_cache
from urllib.parse import urljoin

import requests


# ------------------------- config -------------------------
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


# ------------------------- HTTP scrape (fast path) -------------------------
def _fetch_signasl_urls_http(token: str) -> list[str]:
    token = _strip_punct(token)
    if not token:
        return []

    sess = _browser_session()
    found: list[str] = []

    # 1) JSON API (some tokens return a list of {video_url: ...})
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
            pass  # fall through to HTML

    # 2) HTML: look for mp4/webm/m3u8 declared in attributes or inline URLs
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
            # relative paths from attributes
            for m in attr_re.findall(html):
                found.append(urljoin(base, m))
            # absolute URLs
            for m in abs_re.findall(html):
                found.append(m)
        except Exception:
            pass

    # de-dupe preserve order
    seen, out = set(), []
    for u in found:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# ------------------------- Browser scrape (fallback) -------------------------
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

    out: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
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
            for base in _SIGNASL_BASES:
                url = urljoin(base, f"sign/{token}")
                try:
                    page.goto(url, wait_until="networkidle", timeout=20000)
                    page.wait_for_timeout(600)  # give their JS a moment

                    # Collect from <video> and <source>
                    urls = page.eval_on_selector_all(
                        "video, video source, source",
                        """els => els.map(e => e.currentSrc || e.src || e.getAttribute('src') || e.getAttribute('data-src') || '')"""
                    )
                    for u in urls or []:
                        if u:
                            out.append(urljoin(base, u))
                except Exception:
                    continue
            browser.close()
    except Exception:
        return []

    # keep only media we can handle
    media_re = re.compile(r'\.(mp4|webm|m3u8)(?:\?|$)', re.IGNORECASE)
    out = [u for u in out if media_re.search(u)]

    # de-dupe
    seen, dedup = set(), []
    for u in out:
        if u not in seen:
            dedup.append(u); seen.add(u)
    return dedup


# ------------------------- Public helpers -------------------------
@lru_cache(maxsize=4096)
def fetch_signasl_urls(token: str) -> list[str]:
    """
    Try HTTP scrape first; if none, try headless browser (when enabled).
    """
    urls = _fetch_signasl_urls_http(token)
    if urls:
        return urls
    return _fetch_signasl_urls_browser(token)


def lookup_sign_urls_for_word(word: str) -> list[str]:
    """
    Return 1–2 clips for a word. If none, fall back to letters (max ~6 clips).
    """
    urls = fetch_signasl_urls(word)
    if urls:
        return urls[:2]

    out: list[str] = []
    for ch in _strip_punct(word):
        u = fetch_signasl_urls(ch)
        if u:
            out.append(u[0])
        if len(out) >= 6:
            break
    return out


def translate_text_to_sign(sentence: str) -> list[str]:
    """
    Convenience: return a flat list of URLs for the whole sentence.
    (Your worker still builds a per-word timing plan.)
    """
    tokens = _strip_punct(sentence).split()
    urls: list[str] = []
    for t in tokens:
        urls += lookup_sign_urls_for_word(t)
    return urls


# -------------- tiny CLI for manual tests --------------
if __name__ == "__main__":
    for tok in ["you", "love", "i"]:
        hits = fetch_signasl_urls(tok)
        print(tok, len(hits), hits[:3])
