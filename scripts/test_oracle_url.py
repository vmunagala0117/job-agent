"""Test the safe fetcher against the Oracle career page URL."""
import asyncio
import time
from job_agent.tools import RankingTools

async def main():
    rt = RankingTools(None, None)
    url = "https://ebwh.fa.us2.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1001/job/84335"
    start = time.monotonic()
    r = await rt.fetch_job_description_from_url(url, timeout=10)
    elapsed = round((time.monotonic() - start) * 1000)
    print(f"ok={r['ok']}")
    print(f"error={r.get('error', '')}")
    print(f"text_len={len(r.get('text', ''))}")
    print(f"elapsed={elapsed}ms")
    if r.get("text"):
        print(f"\nFirst 300 chars:\n{r['text'][:300]}")

asyncio.run(main())
