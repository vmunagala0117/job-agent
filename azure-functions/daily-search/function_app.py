"""Azure Functions Timer Trigger for daily job search.

Calls the Job Agent's ``POST /api/cron/daily-search`` endpoint on a schedule.
Default: every day at 06:00 UTC.

Environment variables:
  CRON_APP_URL   — Base URL of the running Job Agent webapp (e.g. https://job-agent.azurewebsites.net)
  CRON_API_KEY   — Shared secret sent as X-Cron-Key header
  CRON_SCHEDULE  — (optional) Override the timer schedule (NCRONTAB, default "0 0 6 * * *")
"""

import datetime
import json
import logging
import os

import azure.functions as func
import httpx

app = func.FunctionApp()

# Azure Functions uses NCRONTAB (6 fields including seconds):
# sec min hour day month day-of-week
SCHEDULE = os.getenv("CRON_SCHEDULE", "0 0 6 * * *")


@app.timer_trigger(
    schedule=SCHEDULE,
    arg_name="timer",
    run_on_startup=False,
)
def daily_job_search(timer: func.TimerRequest) -> None:
    """Trigger daily job search via the webapp's cron endpoint."""
    logging.info("Daily job search triggered at %s", datetime.datetime.utcnow().isoformat())

    if timer.past_due:
        logging.warning("Timer is past due — running anyway")

    app_url = os.getenv("CRON_APP_URL", "http://localhost:8080").rstrip("/")
    api_key = os.getenv("CRON_API_KEY", "")

    if not api_key:
        logging.error("CRON_API_KEY not set — aborting")
        return

    url = f"{app_url}/api/cron/daily-search"
    headers = {"X-Cron-Key": api_key, "Content-Type": "application/json"}

    try:
        # Use httpx for async-compatible HTTP client (sync context in Azure Functions v1)
        with httpx.Client(timeout=300) as client:
            resp = client.post(url, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        logging.info(
            "Daily search completed — profiles processed: %s",
            result.get("profiles_processed", "?"),
        )
        for entry in result.get("results", []):
            logging.info(
                "  Profile '%s': %s (jobs_found=%s, duration=%sms)",
                entry.get("profile", "?"),
                entry.get("status", "?"),
                entry.get("jobs_found", "?"),
                entry.get("duration_ms", "?"),
            )

    except httpx.HTTPStatusError as exc:
        logging.error("HTTP error %s: %s", exc.response.status_code, exc.response.text[:500])
    except Exception as exc:
        logging.exception("Daily search failed: %s", exc)
