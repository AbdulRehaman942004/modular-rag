"""
web_search.py  –  Web Search tool for the Modular-RAG orchestrator.

How it works:
1. Search    – DuckDuckGo (duckduckgo-search) finds the top 5 result URLs + snippets.
2. Scrape    – requests fetches the HTML of the top 3 pages;
               BeautifulSoup strips tags and pulls clean paragraph text.
               Falls back to the DDG snippet if a page cannot be scraped.
3. Synthesize – raw content is passed to Groq, which produces a concise answer.

No API key required for web search.
"""

import sys
import os
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from bs4 import BeautifulSoup
from ddgs import DDGS

# Make the project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LLM import call_groq


# ── Configuration ─────────────────────────────────────────────────────────────

MAX_SEARCH_RESULTS  = 5     # DuckDuckGo results to request
MAX_PAGES_TO_SCRAPE = 3     # Of those, how many pages to actually scrape
MAX_CHARS_PER_PAGE  = 3000  # Character cap per scraped page
REQUEST_TIMEOUT     = 8     # Seconds before giving up on a page HTTP request
DDG_TIMEOUT         = 15    # Seconds before giving up on the DDG search itself

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


# ── Step 1: Search ────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """Return top DuckDuckGo results as a list of {title, href, body} dicts.
    
    Wrapped in a thread with a hard timeout so it never hangs indefinitely.
    """
    def _fetch():
        with DDGS() as ddgs:
            return list(ddgs.text(
                query,
                max_results=max_results,
                region="us-en",       # force English results
                safesearch="off",
            ))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch)
        try:
            return future.result(timeout=DDG_TIMEOUT)
        except FuturesTimeout:
            raise TimeoutError(
                f"DuckDuckGo search timed out after {DDG_TIMEOUT}s. "
                "This usually means DDG is rate-limiting. Try again in a few seconds."
            )


# ── Step 2: Scrape ────────────────────────────────────────────────────────────

def _scrape_page(url: str) -> str:
    """Fetch a URL and return clean paragraph text (no HTML tags)."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Remove noisy elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Collect meaningful paragraph text
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paragraphs if len(p) > 40)
        return text[:MAX_CHARS_PER_PAGE]

    except Exception as e:
        return f"[Could not scrape {url}: {e}]"


# ── Step 3: Synthesize ────────────────────────────────────────────────────────

def _synthesize(query: str, scraped_results: list[dict]) -> str:
    """Ask Groq to produce a grounded answer from the scraped web content."""
    sources_text = ""
    for i, r in enumerate(scraped_results, 1):
        sources_text += (
            f"\n--- Source {i}: {r['title']} ({r['url']}) ---\n"
            f"{r['content']}\n"
        )

    prompt = f"""You are "Now Assist", an expert AI assistant for the ServiceNow platform.
You have been given live web search results to answer the user's question.

USER QUERY:
{query}

WEB SEARCH RESULTS:
{sources_text}

INSTRUCTIONS:
- Answer the query using ONLY the information in the WEB SEARCH RESULTS.
- Be concise, factual, and well-structured.
- Use bullet points or numbered lists where appropriate.
- Where relevant, mention which source the information is from (e.g. "According to Source 1...").
- If the results do not contain enough information to answer, say so clearly.
- Do NOT hallucinate or add facts not present in the results.

RESPONSE:"""

    return call_groq(prompt)


# ── Public API ────────────────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """Search the web and return a synthesized, grounded answer.

    Pipeline:
        DuckDuckGo search → scrape top pages → Groq synthesis → answer string

    Args:
        query: The user's question requiring live or external information.

    Returns:
        A grounded answer string to be used as context by the orchestrator.
    """
    print(f"[web_search] Searching: {query!r}")

    # Step 1 – Search (with hard timeout)
    try:
        results = _ddg_search(query)
    except TimeoutError as e:
        return f"[web_search] {e}"
    except Exception as e:
        return f"[web_search] Search failed: {e}"

    if not results:
        return "[web_search] No search results found for this query."

    print(f"[web_search] Found {len(results)} result(s). Scraping top {MAX_PAGES_TO_SCRAPE}...")

    # Step 2 – Scrape top pages, fall back to snippet if scraping fails
    scraped = []
    for result in results[:MAX_PAGES_TO_SCRAPE]:
        url     = result.get("href", "")
        title   = result.get("title", "No title")
        snippet = result.get("body", "")

        content = _scrape_page(url)

        if content.startswith("[Could not scrape"):
            print(f"  [!] Falling back to snippet for: {url}")
            content = snippet

        scraped.append({"title": title, "url": url, "content": content})
        print(f"  ✓ {title[:70]}")

    # Step 3 – Synthesize
    print("[web_search] Synthesizing answer with Groq...")
    return _synthesize(query, scraped)