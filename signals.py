# --- SignASL dynamic extractor (needs playwright) ----------------------------
from functools import lru_cache
from playwright.sync_api import sync_playwright
import re, time

def _strip_punct(t: str) -> str:
    import string
    return (t or "").translate(str.maketrans("", "", string.punctuation)).lower().strip()

@lru_cache(maxsize=4096)
def fetch_signasl_urls(token: str) -> list[str]:
    """
    Loads https://www.signasl.org/sign/<token> in a headless browser and returns
    a de-duped list of direct video URLs (mp4/webm/hls).
    """
    token = _strip_punct(token)
    if not token:
        return []

    url = f"https://www.signasl.org/sign/{token}"
    vids: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(700)  # give their JS a hair to attach sources

        # 1) <video><source src> nodes
        srcs = page.eval_on_selector_all("video source", "els => els.map(e => e.src || e.getAttribute('src'))")
        # 2) <video src> / currentSrc as a fallback
        vsrcs = page.eval_on_selector_all("video", "els => els.map(e => e.currentSrc || e.src || '')")
        vids = [s for s in (srcs or []) + (vsrcs or []) if s]

        # Clean & de-dupe (prefer mp4/webm over HLS if both exist)
        def score(u: str) -> int:
            u = u.lower()
            if u.endswith(".mp4"): return 3
            if u.endswith(".webm"): return 2
            if ".m3u8" in u: return 1
            return 0
        seen = set()
        ordered = sorted((u for u in vids if u), key=score, reverse=True)
        final = []
        for u in ordered:
            if u not in seen:
                seen.add(u)
                final.append(u)

        browser.close()
        return final

def lookup_sign_urls_for_word(word: str) -> list[str]:
    """Return 1–3 best clips for a word, else fall back to fingerspelling per-letter."""
    urls = fetch_signasl_urls(word)
    if urls:
        # take a couple to avoid super long videos
        return urls[:2]
    # Fingerspelling fallback (keeps you off the caption-only path)
    out = []
    for ch in _strip_punct(word):
        out += fetch_signasl_urls(ch)[:1]
        if len(out) >= 6:  # cap
            break
    return out
