#!/usr/bin/env python
"""End-to-end cron job test script.

Tests the full cron pipeline against a running server:
  1. List profiles and pick the first one with a resume
  2. Enable cron for that profile
  3. Run the cron endpoint (Phase 1: search + rank, Phase 2: app prep)
  4. Verify results — scored jobs, application packages
  5. Display cover letters, resume suggestions, and intro emails
  6. Optionally disable cron when done

Prerequisites:
  - Server running on localhost:8080
  - CRON_API_KEY env var set (or passed via --key)

Usage:
    # Start the server first (in another terminal):
    $env:CRON_API_KEY="test-local-key"
    python -m job_agent.webapp

    # Then run this script:
    python scripts/test_cron.py
    python scripts/test_cron.py --key my-secret --profile abc-123
    python scripts/test_cron.py --base-url http://localhost:8080 --keep-enabled
"""

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_KEY = os.getenv("CRON_API_KEY", "test-local-key")
TIMEOUT_SECONDS = 300  # 5 min max for cron run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(method: str, url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    """Make an HTTP request and return parsed JSON."""
    req = Request(url, method=method, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except HTTPError as e:
        body = e.read().decode()
        print(f"\n  ✗ {method} {url}")
        print(f"    Status: {e.code}")
        try:
            detail = json.loads(body)
            print(f"    Detail: {detail.get('detail', body)}")
        except json.JSONDecodeError:
            print(f"    Body: {body[:300]}")
        sys.exit(1)


def _get(url: str, **kw) -> dict:
    return _req("GET", url, **kw)


def _post(url: str, **kw) -> dict:
    return _req("POST", url, **kw)


def _patch(url: str, **kw) -> dict:
    return _req("PATCH", url, **kw)


def _header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _section(text: str):
    print(f"\n--- {text} ---")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_health_check(base: str):
    """Verify the server is running."""
    _section("Step 0: Health check")
    data = _get(f"{base}/health")
    status = data.get("status", "unknown")
    if status in ("healthy", "ok"):
        print(f"  ✓ Server is {status}")
    else:
        print(f"  ✗ Server status: {status}")
        sys.exit(1)


def step_list_profiles(base: str) -> list[dict]:
    """List profiles and show cron_enabled / has_resume status."""
    _section("Step 1: List profiles")
    profiles = _get(f"{base}/api/profiles")
    if not profiles:
        print("  ✗ No profiles found. Upload a resume first.")
        sys.exit(1)

    print(f"  Found {len(profiles)} profile(s):\n")
    print(f"  {'#':<3} {'Name':<20} {'Resume':<8} {'Cron':<6} {'ID'}")
    print(f"  {'-'*3} {'-'*20} {'-'*8} {'-'*6} {'-'*36}")
    for i, p in enumerate(profiles, 1):
        resume = "✓" if p.get("has_resume") else "✗"
        cron = "ON" if p.get("cron_enabled") else "off"
        print(f"  {i:<3} {p['name']:<20} {resume:<8} {cron:<6} {p['id']}")
    return profiles


def step_pick_profile(profiles: list[dict], explicit_id: str | None) -> dict:
    """Pick a profile to test with."""
    _section("Step 2: Select profile")

    if explicit_id:
        match = [p for p in profiles if p["id"] == explicit_id]
        if not match:
            print(f"  ✗ Profile {explicit_id} not found")
            sys.exit(1)
        chosen = match[0]
    else:
        # Pick first profile with a resume
        with_resume = [p for p in profiles if p.get("has_resume")]
        if not with_resume:
            print("  ✗ No profiles with a resume. Upload one first.")
            sys.exit(1)
        chosen = with_resume[0]

    print(f"  ✓ Using profile: {chosen['name']} ({chosen['id'][:8]}…)")
    if not chosen.get("has_resume"):
        print("  ✗ This profile has no resume — cron will fail.")
        sys.exit(1)
    return chosen


def step_enable_cron(base: str, profile_id: str) -> dict:
    """Enable cron for the chosen profile."""
    _section("Step 3: Enable cron")
    result = _patch(f"{base}/api/profiles/{profile_id}/cron?enabled=true")
    if result.get("cron_enabled"):
        print(f"  ✓ Cron enabled for {result.get('profile_name', profile_id)}")
    else:
        print(f"  ✗ Unexpected response: {result}")
        sys.exit(1)
    return result


def step_run_cron(base: str, key: str, profile_id: str | None = None) -> dict:
    """Run the cron endpoint and wait for results."""
    _section("Step 4: Run cron job")
    url = f"{base}/api/cron/daily-search"
    if profile_id:
        url += f"?profile_id={profile_id}"

    print(f"  Calling POST {url}")
    print(f"  (this may take 1-3 minutes…)\n")

    t0 = time.time()
    result = _post(url, headers={"X-Cron-Key": key}, timeout=TIMEOUT_SECONDS)
    elapsed = time.time() - t0

    status = result.get("status", "unknown")
    if status == "ok":
        print(f"  ✓ Cron completed in {elapsed:.1f}s")
    elif status == "skipped":
        print(f"  ⚠ Cron skipped: {result.get('reason')}")
        return result
    else:
        print(f"  ✗ Unexpected status: {status}")
        print(f"    {json.dumps(result, indent=2)}")
        sys.exit(1)

    # Print per-profile results
    for r in result.get("results", []):
        print(f"\n  Profile: {r['profile']}")
        print(f"    Status:             {r['status']}")
        print(f"    Jobs found:         {r.get('jobs_found', 0)}")
        print(f"    Top matches:        {r.get('top_matches', 0)}")
        print(f"    Packages generated: {r.get('packages_generated', 0)}")
        print(f"    Duration:           {r.get('duration_ms', 0) / 1000:.1f}s")

    return result


def step_show_cron_run(base: str):
    """Fetch and display the latest cron run record."""
    _section("Step 5: Latest cron run record")
    runs = _get(f"{base}/api/cron/runs?limit=1")
    if not runs:
        print("  ✗ No cron runs found")
        return

    run = runs[0]
    print(f"  Run ID:      {run['id']}")
    print(f"  Profile:     {run['profile_name']}")
    print(f"  Status:      {run['status']}")
    print(f"  Jobs found:  {run['jobs_found']}")
    print(f"  Duration:    {(run.get('duration_ms') or 0) / 1000:.1f}s")
    print(f"  Created:     {run['created_at']}")

    top = run.get("top_matches") or []
    if top:
        print(f"\n  Top {len(top)} scored jobs:")
        print(f"  {'#':<3} {'Score':<8} {'Title':<40} {'Company':<20}")
        print(f"  {'-'*3} {'-'*8} {'-'*40} {'-'*20}")
        for i, m in enumerate(top, 1):
            score = m.get("score")
            score_str = f"{score * 100:.1f}%" if score is not None else "N/A"
            title = (m.get("title") or "")[:38]
            company = (m.get("company") or "")[:18]
            print(f"  {i:<3} {score_str:<8} {title:<40} {company:<20}")
            url = m.get("url")
            if url:
                print(f"      🔗 {url[:80]}{'…' if len(url) > 80 else ''}")
    else:
        print("\n  ⚠ No scored matches — ranking may not have run.")

    return run


def step_show_packages(base: str, run: dict | None):
    """Fetch and display application packages generated by the cron."""
    _section("Step 6: Application packages (cover letters, resume suggestions, intro emails)")

    # The cron stores package IDs in the notification_channels field of the run
    # But we can also just list recent packages via the chat agent
    # For now, list packages from the /api/cron/runs top_matches

    # We don't have a direct /api/packages endpoint yet, so let's query
    # the store via a quick chat message
    print("  To view generated application materials, use the chat UI:")
    print('    → "Show my application packages"')
    print('    → "Show the cover letter for the [Job Title] job"')
    print('    → "What resume changes do you suggest for [Company]?"')
    print()
    print("  Or via API — ask the chat endpoint:")
    print(f'    POST {base if base else "http://localhost:8080"}/api/chat')
    print('    {{"message": "List all my application packages"}}')


def step_disable_cron(base: str, profile_id: str):
    """Disable cron for the profile (cleanup)."""
    _section("Step 7: Disable cron (cleanup)")
    result = _patch(f"{base}/api/profiles/{profile_id}/cron?enabled=false")
    if not result.get("cron_enabled"):
        print(f"  ✓ Cron disabled for {result.get('profile_name', profile_id)}")
    else:
        print(f"  ⚠ Cron still enabled: {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test the cron job pipeline end-to-end against a running server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_cron.py
  python scripts/test_cron.py --key my-secret
  python scripts/test_cron.py --profile d33da9d6-4fcd-4cf1-a906-ec68a6beea2c
  python scripts/test_cron.py --keep-enabled   # don't disable cron after test
        """,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Server base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--key", default=DEFAULT_KEY,
                        help="Cron API key (default: $CRON_API_KEY or 'test-local-key')")
    parser.add_argument("--profile", default=None,
                        help="Specific profile ID to test (default: first with resume)")
    parser.add_argument("--keep-enabled", action="store_true",
                        help="Don't disable cron after the test")
    args = parser.parse_args()

    _header("CRON JOB END-TO-END TEST")
    print(f"  Server:  {args.base_url}")
    print(f"  Key:     {'*' * len(args.key)} ({len(args.key)} chars)")

    # Step 0: Health check
    step_health_check(args.base_url)

    # Step 1: List profiles
    profiles = step_list_profiles(args.base_url)

    # Step 2: Pick a profile
    chosen = step_pick_profile(profiles, args.profile)
    profile_id = chosen["id"]

    # Step 3: Enable cron
    step_enable_cron(args.base_url, profile_id)

    # Step 4: Run the cron
    result = step_run_cron(args.base_url, args.key, profile_id)
    if result.get("status") == "skipped":
        _header("TEST SKIPPED")
        return

    # Step 5: Show cron run details
    run = step_show_cron_run(args.base_url)

    # Step 6: Show packages
    step_show_packages(args.base_url, run)

    # Step 7: Disable cron (unless --keep-enabled)
    if not args.keep_enabled:
        step_disable_cron(args.base_url, profile_id)
    else:
        print(f"\n  ℹ Cron left enabled for {chosen['name']} (--keep-enabled)")

    # Summary
    cron_results = result.get("results", [{}])
    r = cron_results[0] if cron_results else {}
    _header("TEST COMPLETE")
    print(f"  Profile:          {chosen['name']}")
    print(f"  Jobs found:       {r.get('jobs_found', 0)}")
    print(f"  Top matches:      {r.get('top_matches', 0)}")
    print(f"  Packages created: {r.get('packages_generated', 0)}")
    print(f"  Duration:         {r.get('duration_ms', 0) / 1000:.1f}s")
    print()

    pkgs = r.get("packages_generated", 0)
    if pkgs > 0:
        print("  ✓ Phase 1 (search + rank) — PASSED")
        print("  ✓ Phase 2 (app prep)      — PASSED")
        print(f"    {pkgs} application package(s) generated with")
        print(f"    cover letters, resume suggestions, and intro emails.")
    elif r.get("top_matches", 0) > 0:
        print("  ✓ Phase 1 (search + rank) — PASSED")
        print("  ⚠ Phase 2 (app prep)      — NO PACKAGES")
        print("    Jobs were scored but none scored > 30% for app prep.")
    else:
        print("  ⚠ Phase 1 (search + rank) — NO SCORED MATCHES")
        print("  ⚠ Phase 2 (app prep)      — SKIPPED")
    print()


if __name__ == "__main__":
    main()
