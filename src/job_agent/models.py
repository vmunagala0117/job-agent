"""Data models for job listings and related entities."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class JobStatus(Enum):
    """Status of a job listing."""
    NEW = "new"
    REVIEWED = "reviewed"
    APPLIED = "applied"
    REJECTED = "rejected"
    INTERVIEWING = "interviewing"
    ARCHIVED = "archived"


class DatePosted(Enum):
    """Date range filter for job search."""
    YESTERDAY = "yesterday"  # Last 24 hours
    LAST_3_DAYS = "3days"  # Last 3 days
    LAST_WEEK = "week"  # Last 7 days
    LAST_MONTH = "month"  # Last 30 days
    ANY = "any"  # No date filter
    
    @property
    def query_suffix(self) -> str:
        """Return the suffix to append to the search query."""
        mapping = {
            "yesterday": "since yesterday",
            "3days": "in the last 3 days",
            "week": "in the last week",
            "month": "in the last month",
            "any": "",
        }
        return mapping.get(self.value, "")


@dataclass
class Job:
    """Represents a job listing."""
    
    title: str
    company: str
    location: str
    description: str
    
    # Optional fields
    url: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    job_type: Optional[str] = None  # e.g., "full-time", "contract", "remote"
    experience_level: Optional[str] = None  # e.g., "entry", "mid", "senior"
    skills: list[str] = field(default_factory=list)
    
    # Metadata
    id: str = field(default_factory=lambda: str(uuid4()))
    source: str = "unknown"  # e.g., "serpapi", "linkedin", "manual"
    status: JobStatus = JobStatus.NEW
    score: Optional[float] = None  # Ranking score (0-1)
    
    # Embedding for similarity matching
    embedding: Optional[list[float]] = None
    
    # Timestamps
    posted_at: Optional[datetime] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = JobStatus(self.status)


@dataclass
class JobSearchCriteria:
    """Criteria for searching jobs."""
    
    query: str  # Search query (e.g., "python developer")
    location: Optional[str] = None
    remote_only: bool = False
    min_salary: Optional[int] = None
    max_results: int = 20
    date_posted: DatePosted = DatePosted.ANY  # Date filter
    experience_levels: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """User profile with resume and job preferences."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    email: str = ""
    
    # Resume content
    resume_text: str = ""  # Full resume text for embedding
    summary: str = ""  # Brief professional summary
    
    # Skills and experience
    skills: list[str] = field(default_factory=list)
    years_experience: Optional[int] = None
    current_title: str = ""
    desired_titles: list[str] = field(default_factory=list)
    
    # Preferences
    preferred_locations: list[str] = field(default_factory=list)
    remote_preference: str = "flexible"  # "remote", "hybrid", "onsite", "flexible"
    min_salary: Optional[int] = None
    industries: list[str] = field(default_factory=list)
    
    # Embedding (stored for similarity matching)
    embedding: Optional[list[float]] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass  
class RankedJob:
    """A job with ranking score and justification."""
    
    job: Job
    score: float  # 0.0 to 1.0
    justification: str  # LLM-generated explanation
    
    # Score breakdown
    similarity_score: float = 0.0  # Embedding similarity
    skill_match_score: float = 0.0  # Skills overlap
    location_score: float = 0.0  # Location preference match
    salary_score: float = 0.0  # Salary match
    
    def __post_init__(self):
        # Update the job's score field
        self.job.score = self.score


class FeedbackType(Enum):
    """User feedback on job match quality."""
    GOOD_FIT = "good_fit"
    NOT_RELEVANT = "not_relevant"
    TAILOR_RESUME = "tailor_resume"
    DRAFT_COVER_LETTER = "draft_cover_letter"
    ALREADY_APPLIED = "already_applied"
    COMPANY_BLACKLIST = "company_blacklist"


@dataclass
class JobFeedback:
    """User feedback on a job listing."""
    
    job_id: str
    feedback_type: FeedbackType
    notes: Optional[str] = None
    
    # Metadata
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


class ResponseRating(Enum):
    """Thumbs up / down on an agent response."""
    UP = "up"
    DOWN = "down"


@dataclass
class ResponseFeedback:
    """User feedback on an agent response, tied to session + message index."""

    session_id: str
    message_index: int  # 0-based index of the assistant message in the session
    rating: ResponseRating
    comment: Optional[str] = None

    # Metadata
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    TEAMS = "teams"
    SLACK = "slack"
    CONSOLE = "console"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel."""
    
    channel: NotificationChannel
    enabled: bool = True
    
    # Email config
    email_to: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # Teams config
    teams_webhook_url: Optional[str] = None
    
    # Slack config
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None


@dataclass
class ApplicationPackage:
    """Compiled materials for a job application."""
    
    job: Job
    profile: "UserProfile"
    
    # Generated content
    resume_suggestions: list[str] = field(default_factory=list)  # Diff-style suggestions
    cover_letter: str = ""
    intro_email: str = ""
    
    # Recruiter info (if available)
    recruiters: list[dict] = field(default_factory=list)  # [{name, title, linkedin, email}]
    
    # Metadata
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "draft"  # draft, approved, sent
    
    experience_levels: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)
