"""FastAPI web application for the Job Agent.

Serves a clean chat UI and exposes a REST API for the multi-agent workflow.
Designed for deployment to Azure Web App or Azure Container App.

Usage:
    uvicorn job_agent.webapp:app --host 0.0.0.0 --port 8080

Or via the CLI entry point:
    python -m job_agent.webapp
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_framework import ChatMessage, Role, TextContent

from .clients import build_azure_openai_client
from .config import AppConfig
from .models import ResponseFeedback, ResponseRating
from .workflows import create_agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace buffer — captures [TRACE] logs per session for the UI panel
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    """A single trace log line."""
    timestamp: float
    level: str
    message: str


class SessionTraceHandler(logging.Handler):
    """Logging handler that captures [TRACE] lines into per-session buffers."""

    def __init__(self, max_entries: int = 200):
        super().__init__()
        self._buffers: dict[str, deque[TraceEntry]] = {}
        self._max = max_entries
        self._active_session: str | None = None

    def set_session(self, session_id: str) -> None:
        self._active_session = session_id
        if session_id not in self._buffers:
            self._buffers[session_id] = deque(maxlen=self._max)

    def clear_session(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)

    def get_traces(self, session_id: str) -> list[dict]:
        buf = self._buffers.get(session_id, [])
        return [
            {"timestamp": e.timestamp, "level": e.level, "message": e.message}
            for e in buf
        ]

    def emit(self, record: logging.LogRecord) -> None:
        if self._active_session is None:
            return
        msg = self.format(record)
        entry = TraceEntry(
            timestamp=record.created,
            level=record.levelname,
            message=msg,
        )
        buf = self._buffers.get(self._active_session)
        if buf is not None:
            buf.append(entry)


# Global trace handler — installed once at startup
_trace_handler = SessionTraceHandler()


# ---------------------------------------------------------------------------
# Session management (in-memory — swap for Redis on multi-instance Azure)
# ---------------------------------------------------------------------------

@dataclass
class ChatSession:
    """Maintains conversation history for one chat session."""
    id: str
    messages: list[ChatMessage] = field(default_factory=list)
    resume_filename: str | None = None


_sessions: dict[str, ChatSession] = {}
_agent: Any = None  # Populated during lifespan startup
_services: dict[str, Any] = {}  # Shared services for resume upload


def _get_or_create_session(session_id: str | None) -> ChatSession:
    """Return existing session or create a new one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    session = ChatSession(id=new_id)
    _sessions[new_id] = session
    return session


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    traces: list[dict] = []
    usage: dict | None = None  # {input_tokens, output_tokens, total_tokens}
    classifier_confidence: float | None = None
    elapsed_ms: int | None = None
    agent: str | None = None
    tool_count: int | None = None


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool


class SessionInfo(BaseModel):
    session_id: str
    message_count: int


class UploadResponse(BaseModel):
    status: str
    filename: str
    summary: str
    session_id: str


# ---------------------------------------------------------------------------
# Application lifespan — initialise agent once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start up: configure telemetry, build the multi-agent workflow. Shut down: clean up."""
    global _agent

    # --- Install trace handler for UI panel ---
    wf_logger = logging.getLogger("job_agent.workflows")
    _trace_handler.setFormatter(logging.Formatter("%(message)s"))
    wf_logger.addHandler(_trace_handler)
    wf_logger.setLevel(logging.INFO)

    # --- OpenTelemetry ---
    from agent_framework.observability import configure_otel_providers

    # Build list of exporters — add Azure Monitor if connection string is set
    exporters = []
    ai_conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if ai_conn_str:
        try:
            from azure.monitor.opentelemetry.exporter import (
                AzureMonitorLogExporter,
                AzureMonitorMetricExporter,
                AzureMonitorTraceExporter,
            )
            exporters.append(AzureMonitorTraceExporter(connection_string=ai_conn_str))
            exporters.append(AzureMonitorMetricExporter(connection_string=ai_conn_str))
            exporters.append(AzureMonitorLogExporter(connection_string=ai_conn_str))
            logger.info("Azure Application Insights exporters configured")
        except Exception as exc:
            logger.warning("Could not configure App Insights exporters: %s", exc)

    configure_otel_providers(exporters=exporters if exporters else None)
    logger.info("OpenTelemetry configured")

    logger.info("Initialising Job Agent...")
    config = AppConfig.load()
    client = build_azure_openai_client(config.azure_openai)
    use_db = bool(config.database and config.database.is_configured)

    # create_agent now returns (agent, store, ranking_svc) — single set of services
    _agent, store, ranking_svc = await create_agent(client, use_database=use_db)
    _services["store"] = store
    _services["ranking"] = ranking_svc

    # --- Auto-load default profile from database ---
    from .tools import set_current_profile
    try:
        default_profile = await store.get_default_profile()
        if default_profile:
            set_current_profile(default_profile)
            logger.info("Auto-loaded profile: %s", default_profile.name)
        else:
            logger.info("No saved profiles found — upload one to get started")
    except Exception as exc:
        logger.warning("Could not auto-load profile: %s", exc)

    logger.info("Job Agent ready ✓")
    yield
    logger.info("Shutting down Job Agent")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Job Agent",
    description="AI-powered job search and application preparation assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static assets (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def index():
    """Serve the chat UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for Azure load balancer / readiness probe."""
    return HealthResponse(status="ok", agent_ready=_agent is not None)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to the agent and get a response."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session = _get_or_create_session(req.session_id)

    # Point the trace handler at this session
    _trace_handler.set_session(session.id)

    # Append user message
    session.messages.append(
        ChatMessage(role=Role.USER, contents=[TextContent(text=req.message)])
    )

    try:
        response = await _agent.run(session.messages)
        raw_text = response.text or "(No response)"

        # Parse metadata suffix injected by the workflow
        metadata = {}
        _META_RE = re.compile(r"\n<!--METADATA:(\{.*?\})-->$", re.DOTALL)
        match = _META_RE.search(raw_text)
        if match:
            try:
                metadata = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
            assistant_text = raw_text[: match.start()]
        else:
            assistant_text = raw_text

        # Append assistant message
        session.messages.append(
            ChatMessage(role=Role.ASSISTANT, contents=[TextContent(text=assistant_text)])
        )

        # Collect traces captured during this request
        traces = _trace_handler.get_traces(session.id)

        return ChatResponse(
            response=assistant_text,
            session_id=session.id,
            traces=traces,
            usage=metadata.get("usage"),
            classifier_confidence=metadata.get("classifier_confidence"),
            elapsed_ms=metadata.get("elapsed_ms"),
            agent=metadata.get("agent"),
            tool_count=metadata.get("tool_count"),
        )

    except Exception as exc:
        logger.exception("Agent error")
        session.messages.pop()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload-resume", response_model=UploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    session_id: str | None = None,
):
    """Upload and parse a resume file (PDF, DOCX, TXT).

    Creates a user profile with embeddings for job matching.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    # Validate file type
    filename = file.filename or "resume"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "docx", "txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Use PDF, DOCX, or TXT.",
        )

    session = _get_or_create_session(session_id)
    _trace_handler.set_session(session.id)

    try:
        # Read and base64-encode for the parser
        file_bytes = await file.read()
        file_b64 = base64.b64encode(file_bytes).decode("ascii")

        # Use ResumeParser directly (same path as the agent's upload_resume tool)
        from .resume_parser import ResumeParser
        from .tools import set_current_profile

        parser = ResumeParser()
        parsed = await parser.parse_and_extract(
            file_data=file_b64,
            file_type=ext,
            use_llm=False,
        )

        profile = parsed.to_user_profile()

        # Embed the profile for ranking
        ranking_svc = _services.get("ranking")
        if ranking_svc:
            profile = await ranking_svc.embed_user_profile(profile)

        # Set as current profile and persist
        set_current_profile(profile)
        store = _services.get("store")
        if store:
            await store.save_profile(profile)

        session.resume_filename = filename

        # Inject a system note into the conversation so the agent knows
        summary_parts = []
        if parsed.name:
            summary_parts.append(f"Name: {parsed.name}")
        if parsed.current_title:
            summary_parts.append(f"Title: {parsed.current_title}")
        if parsed.years_experience:
            summary_parts.append(f"Experience: {parsed.years_experience} years")
        skills = ", ".join(parsed.skills[:12])
        if parsed.skills:
            summary_parts.append(f"Skills: {skills}")

        summary = "\n".join(summary_parts)

        # Add to conversation so agent has context
        session.messages.append(
            ChatMessage(
                role=Role.USER,
                contents=[TextContent(text=f"[System: User uploaded resume '{filename}'. Parsed profile:\n{summary}]")],
            )
        )
        session.messages.append(
            ChatMessage(
                role=Role.ASSISTANT,
                contents=[TextContent(text=f"Resume parsed successfully! I've loaded your profile. {summary}")],
            )
        )

        return UploadResponse(
            status="ok",
            filename=filename,
            summary=summary,
            session_id=session.id,
        )

    except Exception as exc:
        logger.exception("Resume upload error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/traces")
async def get_traces(session_id: str):
    """Return trace log entries for a session."""
    return _trace_handler.get_traces(session_id)


@app.get("/api/profiles")
async def list_profiles():
    """List all saved user profiles for the dropdown selector."""
    store = _services.get("store")
    if not store:
        return []
    try:
        profiles = await store.list_profiles()
        from .tools import get_current_profile
        current = get_current_profile()
        current_id = current.id if current else None
        return [
            {
                "id": p.id,
                "name": p.name or "Unnamed",
                "title": p.current_title or "",
                "skills_count": len(p.skills),
                "updated_at": p.updated_at.isoformat() if p.updated_at else "",
                "is_active": p.id == current_id,
                "cron_enabled": p.cron_enabled,
                "has_resume": bool(p.resume_text),
            }
            for p in profiles
        ]
    except Exception as exc:
        logger.exception("Failed to list profiles")
        return []


@app.post("/api/profiles/select")
async def select_profile(profile_id: str):
    """Activate a saved profile by ID."""
    store = _services.get("store")
    if not store:
        raise HTTPException(status_code=503, detail="Store not available")

    profile = await store.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    from .tools import set_current_profile
    set_current_profile(profile)
    logger.info("Switched to profile: %s", profile.name)

    return {
        "status": "ok",
        "name": profile.name,
        "title": profile.current_title or "",
        "skills": profile.skills[:12],
    }


@app.patch("/api/profiles/{profile_id}/cron")
async def toggle_profile_cron(profile_id: str, enabled: bool = True):
    """Enable or disable the daily cron job for a specific profile.

    Only profiles with a resume and cron_enabled=True will be processed
    by the automated daily search pipeline.

    Query params:
      enabled — True to opt-in, False to opt-out (default: True)
    """
    store = _services.get("store")
    if not store:
        raise HTTPException(status_code=503, detail="Store not available")

    profile = await store.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if enabled and not profile.resume_text:
        raise HTTPException(
            status_code=400,
            detail="Cannot enable cron for a profile without a resume. "
                   "Upload a resume first.",
        )

    try:
        success = await store.set_cron_enabled(profile_id, enabled)
    except Exception as exc:
        logger.exception("set_cron_enabled failed for profile %s: %s", profile_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {exc}")
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update profile")

    action = "enabled" if enabled else "disabled"
    logger.info("Cron %s for profile %s (%s)", action, profile.name, profile_id)

    return {
        "status": "ok",
        "profile_id": profile_id,
        "profile_name": profile.name,
        "cron_enabled": enabled,
    }


@app.get("/api/profiles/{profile_id}")
async def get_profile_detail(profile_id: str):
    """Return full profile details for the edit form."""
    store = _services.get("store")
    if not store:
        raise HTTPException(status_code=503, detail="Store not available")
    profile = await store.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "id": profile.id,
        "name": profile.name,
        "email": profile.email,
        "current_title": profile.current_title,
        "summary": profile.summary,
        "skills": profile.skills,
        "years_experience": profile.years_experience,
        "desired_titles": profile.desired_titles,
        "preferred_locations": profile.preferred_locations,
        "remote_preference": profile.remote_preference,
        "min_salary": profile.min_salary,
        "industries": profile.industries,
        "has_resume": bool(profile.resume_text),
        "cron_enabled": profile.cron_enabled,
    }


class ProfileSaveRequest(BaseModel):
    id: str | None = None  # None = create new, set = update existing
    name: str
    email: str = ""
    # Resume-extracted fields — not sent from the form; preserved from existing profile
    current_title: str | None = None
    summary: str | None = None
    skills: list[str] | None = None
    years_experience: int | None = None
    # User preference fields — editable in the form
    desired_titles: list[str] = []
    preferred_locations: list[str] = []
    remote_preference: str = "flexible"
    min_salary: int | None = None
    industries: list[str] = []


@app.post("/api/profiles/save")
async def save_profile(req: ProfileSaveRequest):
    """Create or update a user profile from the profile form."""
    from datetime import datetime
    from .models import UserProfile
    from .tools import set_current_profile

    store = _services.get("store")
    if not store:
        raise HTTPException(status_code=503, detail="Store not available")

    # If updating, load existing profile to preserve resume_text / embedding
    existing = None
    if req.id:
        existing = await store.get_profile(req.id)

    profile = UserProfile(
        id=req.id or str(uuid.uuid4()),
        name=req.name,
        email=req.email,
        # Resume-extracted fields: preserve from existing profile
        current_title=existing.current_title if existing else (req.current_title or ""),
        summary=existing.summary if existing else (req.summary or ""),
        skills=existing.skills if existing else (req.skills or []),
        years_experience=existing.years_experience if existing else req.years_experience,
        # User preference fields: always take from form
        desired_titles=req.desired_titles,
        preferred_locations=req.preferred_locations,
        remote_preference=req.remote_preference,
        min_salary=req.min_salary,
        industries=req.industries,
        # Preserve resume content and embedding from existing profile
        resume_text=existing.resume_text if existing else "",
        embedding=existing.embedding if existing else None,
        created_at=existing.created_at if existing else datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    # Generate embedding from profile text if none exists
    if not profile.embedding:
        ranking_svc = _services.get("ranking")
        if ranking_svc:
            try:
                profile = await ranking_svc.embed_user_profile(profile)
            except Exception as exc:
                logger.warning("Could not embed profile: %s", exc)

    # Set as active and persist
    set_current_profile(profile)
    await store.save_profile(profile)

    action = "updated" if existing else "created"
    logger.info("Profile %s: %s (%s)", action, profile.name, profile.current_title)

    return {
        "status": "ok",
        "action": action,
        "id": profile.id,
        "name": profile.name,
        "title": profile.current_title,
        "skills_count": len(profile.skills),
    }


# ---------------------------------------------------------------------------
# Response feedback (thumbs up/down)
# ---------------------------------------------------------------------------

# In-memory feedback store (swap for DB table in production)
_response_feedback: list[ResponseFeedback] = []


class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int
    rating: str  # "up" or "down"
    comment: str | None = None


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Record thumbs up/down feedback on an agent response."""
    try:
        rating = ResponseRating(req.rating)
    except ValueError:
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    fb = ResponseFeedback(
        session_id=req.session_id,
        message_index=req.message_index,
        rating=rating,
        comment=req.comment,
    )
    _response_feedback.append(fb)

    # Also emit an OTel span for the feedback event
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("job_agent.feedback")
        with tracer.start_as_current_span("response_feedback") as span:
            span.set_attribute("feedback.session_id", req.session_id)
            span.set_attribute("feedback.message_index", req.message_index)
            span.set_attribute("feedback.rating", req.rating)
            span.set_attribute("feedback.comment", req.comment or "")
    except Exception:
        pass  # OTel is optional

    logger.info(
        "Feedback recorded: session=%s msg=%d rating=%s",
        req.session_id, req.message_index, req.rating,
    )
    return {"status": "ok", "id": fb.id}


@app.get("/api/feedback")
async def list_feedback(session_id: str | None = None):
    """List recorded response feedback, optionally filtered by session."""
    items = _response_feedback
    if session_id:
        items = [f for f in items if f.session_id == session_id]
    return [
        {
            "id": f.id,
            "session_id": f.session_id,
            "message_index": f.message_index,
            "rating": f.rating.value,
            "comment": f.comment,
            "created_at": f.created_at.isoformat(),
        }
        for f in items
    ]


@app.get("/api/feedback/analytics")
async def feedback_analytics():
    """Aggregate analytics over collected response feedback.

    Returns satisfaction rate, breakdown by rating, per-agent stats,
    low-confidence correlations, and recent negative feedback.
    """
    if not _response_feedback:
        return {
            "total": 0,
            "satisfaction_rate": None,
            "message": "No feedback collected yet.",
        }

    total = len(_response_feedback)
    up_count = sum(1 for f in _response_feedback if f.rating == ResponseRating.UP)
    down_count = total - up_count
    satisfaction_rate = round(up_count / total * 100, 1) if total else 0

    # Per-session satisfaction
    session_ids = {f.session_id for f in _response_feedback}
    sessions_with_negative = sum(
        1
        for sid in session_ids
        if any(
            f.session_id == sid and f.rating == ResponseRating.DOWN
            for f in _response_feedback
        )
    )

    # Recent negative feedback (last 10)
    negatives = [
        {
            "session_id": f.session_id,
            "message_index": f.message_index,
            "comment": f.comment,
            "created_at": f.created_at.isoformat(),
        }
        for f in reversed(_response_feedback)
        if f.rating == ResponseRating.DOWN
    ][:10]

    # Feedback timeline — count per day
    from collections import Counter

    daily = Counter()
    daily_up = Counter()
    daily_down = Counter()
    for f in _response_feedback:
        day = f.created_at.strftime("%Y-%m-%d")
        daily[day] += 1
        if f.rating == ResponseRating.UP:
            daily_up[day] += 1
        else:
            daily_down[day] += 1

    timeline = [
        {
            "date": day,
            "total": daily[day],
            "up": daily_up.get(day, 0),
            "down": daily_down.get(day, 0),
        }
        for day in sorted(daily)
    ]

    return {
        "total": total,
        "thumbs_up": up_count,
        "thumbs_down": down_count,
        "satisfaction_rate": satisfaction_rate,
        "unique_sessions": len(session_ids),
        "sessions_with_negative": sessions_with_negative,
        "recent_negative": negatives,
        "timeline": timeline,
    }


@app.post("/api/chat/reset", response_model=SessionInfo)
async def reset_session(req: ChatRequest):
    """Clear conversation history, traces, and start fresh."""
    session_id = req.session_id
    if session_id and session_id in _sessions:
        del _sessions[session_id]
        _trace_handler.clear_session(session_id)
    new_session = _get_or_create_session(None)
    return SessionInfo(session_id=new_session.id, message_count=0)


# ---------------------------------------------------------------------------
# Cron / automated daily search
# ---------------------------------------------------------------------------

def _verify_cron_key(request) -> None:
    """Validate the X-Cron-Key header against the configured secret."""
    from .config import AppConfig

    config = AppConfig.load()
    expected = config.cron.api_key if config.cron else os.getenv("CRON_API_KEY", "")
    provided = request.headers.get("X-Cron-Key", "")
    if not expected or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Cron-Key")


@app.post("/api/cron/daily-search")
async def cron_daily_search(request: Request, profile_id: str | None = None):
    """Run an automated daily job search for all profiles.

    Secured with an ``X-Cron-Key`` header.  The endpoint:
      1. Loads saved profiles
      2. For each profile, sends a synthetic message through the existing agent
      3. Persists a SearchRun record with results
      4. Returns a summary

    Called by Azure Functions Timer Trigger (or any scheduler).

    Query params:
      profile_id — (optional) run for a single profile only (useful for testing)
    """

    _verify_cron_key(request)

    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    store = _services.get("store")
    if not store:
        raise HTTPException(status_code=503, detail="Store not available")

    from .models import SearchRun, SearchRunStatus
    from .tools import get_current_profile, set_current_profile

    # Load only cron-enabled profiles (with resume)
    if profile_id:
        # Single profile override for testing — still requires cron_enabled or explicit ID
        profile = await store.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")
        if not profile.resume_text:
            raise HTTPException(
                status_code=400,
                detail=f"Profile '{profile.name}' has no resume. Upload one first.",
            )
        profiles = [profile]
    else:
        profiles = await store.list_cron_profiles()
        if not profiles:
            return {"status": "skipped", "reason": "No cron-enabled profiles found"}

    results = []
    for profile in profiles:
        run = SearchRun(profile_id=profile.id, profile_name=profile.name)
        await store.save_search_run(run)

        t0 = time.time()
        try:
            # Activate profile for the agent
            set_current_profile(profile)

            # Build a synthetic prompt that triggers the full search pipeline
            synthetic_msg = (
                "Run a daily automated job search based on my profile preferences. "
                "Use suggest_search_titles to get optimised keywords, then search_jobs "
                "for each keyword with max_results=30. Rank the results and send "
                "notifications for the top matches."
            )

            session = _get_or_create_session(None)
            session.messages.append(
                ChatMessage(role=Role.USER, contents=[TextContent(text=synthetic_msg)])
            )

            response = await _agent.run(session.messages)
            raw_text = response.text or ""

            # Parse METADATA suffix to extract usage info
            import re as _re

            _META_RE = _re.compile(r"\n<!--METADATA:(\{.*?\})-->$", _re.DOTALL)
            match = _META_RE.search(raw_text)
            if match:
                raw_text = raw_text[: match.start()]

            # Gather recently fetched jobs for this profile
            recent_jobs = await store.list_all(limit=30)
            top = [
                {
                    "id": j.id[:8],
                    "title": j.title,
                    "company": j.company,
                    "score": j.score,
                    "url": j.url,
                }
                for j in recent_jobs[:10]
                if j.score is not None
            ]

            # ── Phase 2: Generate application materials for top matches ──
            package_ids: list[str] = []
            prep_candidates = [j for j in recent_jobs[:10]
                               if j.score is not None and j.score > 0.3]
            if prep_candidates:
                top5_ids = [j.id[:8] for j in prep_candidates[:5]]
                prep_msg = (
                    "For each of the following top job matches, prepare application "
                    "materials (cover letter, resume suggestions, and intro email). "
                    "Job IDs: " + ", ".join(top5_ids)
                )
                session.messages.append(
                    ChatMessage(role=Role.USER, contents=[TextContent(text=prep_msg)])
                )
                prep_response = await _agent.run(session.messages)
                logger.info("Cron Phase 2 (app prep) completed for %s", profile.name)

                # Collect packages created in this window
                from datetime import datetime, timedelta, timezone
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
                all_pkgs = await store.list_application_packages(limit=20)
                for pkg in all_pkgs:
                    pkg_ts = pkg.created_at
                    if pkg_ts.tzinfo is None:
                        pkg_ts = pkg_ts.replace(tzinfo=timezone.utc)
                    if pkg_ts >= cutoff:
                        package_ids.append(pkg.id)

            elapsed_ms = int((time.time() - t0) * 1000)
            run.status = SearchRunStatus.COMPLETED
            run.jobs_found = len(recent_jobs)
            run.top_matches = top
            run.duration_ms = elapsed_ms
            # Store generated package IDs in notification_channels field
            if package_ids:
                run.notification_channels = package_ids
            await store.update_search_run(run)

            results.append({
                "profile": profile.name,
                "status": "completed",
                "jobs_found": run.jobs_found,
                "top_matches": len(top),
                "packages_generated": len(package_ids),
                "duration_ms": elapsed_ms,
            })

            # Clean up the throwaway session
            if session.id in _sessions:
                del _sessions[session.id]

        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            run.status = SearchRunStatus.FAILED
            run.error_message = str(exc)[:500]
            run.duration_ms = elapsed_ms
            await store.update_search_run(run)
            logger.exception("Cron search failed for profile %s", profile.name)
            results.append({
                "profile": profile.name,
                "status": "failed",
                "error": str(exc)[:200],
            })

    return {"status": "ok", "profiles_processed": len(results), "results": results}


@app.get("/api/cron/runs")
async def list_cron_runs(limit: int = 20):
    """List recent automated search runs."""
    store = _services.get("store")
    if not store:
        return []

    from .models import SearchRun  # noqa: F811

    runs = await store.list_search_runs(limit=limit)
    return [
        {
            "id": r.id,
            "profile_name": r.profile_name,
            "search_keywords": r.search_keywords,
            "jobs_found": r.jobs_found,
            "top_matches": r.top_matches,
            "status": r.status.value,
            "error_message": r.error_message,
            "duration_ms": r.duration_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the web application with uvicorn."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    # Show agent traces
    logging.getLogger("job_agent.workflows").setLevel(logging.INFO)

    uvicorn.run(
        "job_agent.webapp:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
