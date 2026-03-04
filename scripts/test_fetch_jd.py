"""Quick smoke tests for the safe fetcher and Fetch JD regex."""
import asyncio
import re
import sys

# ── Regex detection tests ──────────────────────────────────────────────
pattern = r"^\s*fetch\s+jd\s*$"
cases = [
    ("Fetch JD", True),
    ("fetch jd", True),
    ("  Fetch  JD  ", True),
    ("FETCH JD", True),
    ("Fetch JD please", False),
    ("go fetch jd now", False),
    ("search for jobs", False),
]
print("=== Regex tests ===")
for text, expect in cases:
    result = bool(re.match(pattern, text, re.I))
    status = "PASS" if result == expect else "FAIL"
    print(f"  {status}: \"{text}\" -> {result} (expected {expect})")
    if result != expect:
        sys.exit(1)

# ── Safe fetcher tests ─────────────────────────────────────────────────
from job_agent.tools import RankingTools  # noqa: E402


async def test_fetcher():
    rt = RankingTools(store=None, ranking_service=None)

    # 1. Block private/loopback
    r = await rt.fetch_job_description_from_url("http://localhost:8080/test")
    print(f"\n=== SSRF block ===\n  {r}")
    assert not r["ok"], "Expected SSRF block"

    # 2. Bad scheme
    r = await rt.fetch_job_description_from_url("ftp://example.com/file")
    print(f"\n=== Scheme block ===\n  {r}")
    assert not r["ok"], "Expected scheme block"

    # 3. Public HTML (httpbin)
    r = await rt.fetch_job_description_from_url("https://httpbin.org/html", timeout=10)
    print(f"\n=== Public HTML ===\n  ok={r['ok']}  text_len={len(r.get('text',''))}")
    assert r["ok"], "Expected successful fetch"

    print("\nAll tests passed!")


asyncio.run(test_fetcher())
