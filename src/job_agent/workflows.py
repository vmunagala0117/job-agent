"""Multi-agent job search workflow.

Architecture:
  CoordinatorExecutor (Router)
    ├── job_search_agent   (ChatAgent with 12 tools)
    └── application_prep_agent  (ChatAgent with 3 tools)

The Coordinator classifies each user request and delegates to the appropriate
specialist agent. Each specialist has its own tools and instructions.

Built with WorkflowBuilder (single-node executor that internally manages
two sub-agents via programmatic routing).
"""

import json
import logging
import math
import time
from typing import Any, Optional

from agent_framework import (
    ChatMessage,
    Executor,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient
from openai import AsyncAzureOpenAI
from opentelemetry import trace

from .application_prep import ApplicationPrepService, get_application_prep_service
from .models import (
    DatePosted,
    FeedbackType,
    JobFeedback,
    JobSearchCriteria,
    JobStatus,
    UserProfile,
)
from .notifications import NotificationService, get_notification_service
from .providers import JobIngestionProvider, get_provider
from .ranking import RankingService, get_ranking_service
from .resume_parser import ResumeParser
from .store import InMemoryJobStore, JobStore, get_store
from .tools import get_current_profile, set_current_profile

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("job_agent.workflows")


# ---------------------------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------------------------

CLASSIFIER_INSTRUCTIONS = """\
You are a request classifier. Categorize the user's LATEST message into exactly \
one of these two categories. Ignore earlier conversation context — focus only on \
what the user is asking NOW:

JOB_SEARCH — anything related to searching, listing, ranking, filtering jobs, \
setting a profile, uploading a resume, providing a resume file path, \
notifications, feedback, or general greetings / small talk.

APP_PREP — anything related to preparing application materials WHEN a user \
profile already exists: tailored resume suggestions, cover letters, intro emails, \
recruiter search, application packages, analyzing job fit.

IMPORTANT: If the user mentions a resume FILE PATH or asks to upload/parse a \
resume, classify as JOB_SEARCH (that's where the resume parser tool lives).

Reply with ONLY the category name: JOB_SEARCH or APP_PREP. \
Nothing else — no punctuation, no explanation.
"""

JOB_SEARCH_INSTRUCTIONS = """\
You are a job search specialist. You help users find, rank, and manage job \
opportunities.

CRITICAL — ALWAYS CHECK PROFILE FIRST:
- Before asking the user ANY questions, call get_profile() to load their \
  existing profile and preferences.
- The profile contains: desired_titles, preferred_locations, remote_preference, \
  min_salary, industries, skills, current_title, and resume content.
- Use these preferences directly for searches — do NOT re-ask for information \
  that's already in the profile.
- Only ask the user if essential preferences are genuinely missing from the \
  profile (e.g., no desired_titles AND no current_title to infer from).
- If the user says "search for jobs based on my preferences", you have \
  everything you need in the profile — run the searches immediately.

SMART SEARCH STRATEGY:
- When searching based on a user's profile/preferences, call \
  suggest_search_titles() FIRST to get optimized search keywords tailored \
  to their experience level, skills, desired roles, and resume.
- Use the suggested keywords across MULTIPLE search_jobs calls for broader \
  coverage (e.g., one search per keyword category).
- Don't limit yourself to only the user's desired_titles — the suggested \
  keywords include career-growth roles and skill-based variations that the \
  user might not have thought of.
- Combine the keyword suggestions with the user's location and salary \
  preferences from their profile.

SEARCH TIPS:
- For remote jobs: Use remote_only=True, NOT location="Remote"
- For broader results: Try multiple search terms
- For senior roles: Search "director", "VP", "head of", "principal", "staff"
- Many high-paying jobs don't list salary - suggest removing min_salary filter
- If a search returns few results, automatically try broader terms
- Results are auto-sorted by a composite score: 40% recency + 60% profile match
- Default max_results is 30 — use this for broad coverage

DATE FILTERING:
- "yesterday"/"today" = last 24 hours
- "3days" = last 3 days
- "week" = last 7 days
- "month" = last 30 days
- Omit for all jobs regardless of posting date

RESUME HANDLING:
- Use upload_resume when user provides a PDF/DOCX file or base64-encoded resume
- Use set_user_profile when user describes their background in text

WORKFLOW:
1. Call get_profile() first to load existing preferences and resume
2. Call suggest_search_titles() to get smart keyword suggestions based on \
   the full profile (experience, skills, resume, desired roles)
3. Search for jobs using the suggested keywords combined with profile \
   preferences (locations, salary, remote preference)
4. Use rank_saved_jobs to show best matches
5. Help track applications with mark_job_applied / mark_job_rejected
6. Use send_job_notifications to deliver top matches
7. Use provide_feedback to capture user feedback

When presenting jobs, format them clearly with title, company, location, salary, 
and the job posting link (URL). Always include the link so the user can apply.
Be proactive - suggest alternative searches if results are limited.
"""

APP_PREP_INSTRUCTIONS = """\
You are an application preparation specialist. You help users create tailored \
application materials for specific jobs.

IMPORTANT: Always call get_profile() first to load the user's full profile, \
skills, experience, and resume before generating any materials.

CAPABILITIES:
- Analyze how well a user's profile matches a specific job
- Generate targeted resume diff suggestions (not full rewrites)
- Draft concise, compelling cover letters
- Create intro emails for recruiters/hiring managers
- Package all materials for review (prepare_application)

WORKFLOW:
1. Call get_profile() to load the user's profile and resume
2. If the user asks about a specific job, use analyze_job_fit first
3. For complete application packages, use prepare_application
4. To retrieve a previously created package, use get_application_package

Be specific and actionable in your suggestions. Reference actual skills and \
experience from the user's profile.
"""


# ---------------------------------------------------------------------------
# Job Search tools
# ---------------------------------------------------------------------------

class JobSearchTools:
    """Tool implementations for the Job Search Agent."""

    def __init__(
        self,
        store: JobStore,
        provider: JobIngestionProvider,
        ranking_service: RankingService,
        notification_service: NotificationService,
        openai_client: Optional[AsyncAzureOpenAI] = None,
        deployment_name: str = "",
    ):
        self.store = store
        self.provider = provider
        self.ranking_service = ranking_service
        self.notification_service = notification_service
        self._openai_client = openai_client
        self._deployment = deployment_name

    async def search_jobs(
        self,
        query: str,
        location: Optional[str] = None,
        remote_only: bool = False,
        min_salary: Optional[int] = None,
        max_results: int = 30,
        date_posted: Optional[str] = None,
    ) -> str:
        """Search for new job listings matching the criteria.

        Args:
            query: Job search query - use role/title keywords (e.g., "AI engineer",
                   "machine learning", "python developer").
            location: Geographic location (e.g., "San Francisco, CA", "New York").
                      Do NOT use "Remote" here - use remote_only=True instead.
            remote_only: Set to True for remote/work-from-home positions.
            min_salary: Minimum yearly salary requirement (e.g., 150000 for $150K).
                        Many postings don't include salary, so high minimums may filter out jobs.
            max_results: Maximum number of results to return (default 30, max 100).
            date_posted: Filter by posting date. Options: "yesterday", "3days", "week", "month".

        Returns:
            A formatted string with matching job listings.
        """
        # Detect "remote" passed as location
        if location and location.lower() in ("remote", "remote only", "work from home"):
            remote_only = True
            location = "United States"

        # Parse date_posted
        date_filter = DatePosted.ANY
        if date_posted:
            date_mapping = {
                "yesterday": DatePosted.YESTERDAY,
                "today": DatePosted.YESTERDAY,
                "24hours": DatePosted.YESTERDAY,
                "3days": DatePosted.LAST_3_DAYS,
                "3 days": DatePosted.LAST_3_DAYS,
                "week": DatePosted.LAST_WEEK,
                "7days": DatePosted.LAST_WEEK,
                "month": DatePosted.LAST_MONTH,
                "30days": DatePosted.LAST_MONTH,
            }
            date_filter = date_mapping.get(date_posted.lower().replace(" ", ""), DatePosted.ANY)

        criteria = JobSearchCriteria(
            query=query,
            location=location,
            remote_only=remote_only,
            min_salary=min_salary,
            max_results=max_results,
            date_posted=date_filter,
        )

        jobs = await self.provider.fetch_jobs(criteria)
        await self.store.add_many(jobs)

        # Generate and persist embeddings
        if jobs:
            jobs = await self.ranking_service.embed_jobs(jobs)
            job_embeddings = [(j.id, j.embedding) for j in jobs if j.embedding]
            if job_embeddings:
                await self.store.update_job_embeddings(job_embeddings)

        # --- Composite sorting: recency (40%) + profile match (60%) ---
        if jobs:
            jobs = self._sort_jobs_composite(jobs)
            # Persist scores to DB so cron/API can access them
            score_updates = [(j.id, j.score) for j in jobs if j.score is not None]
            if score_updates:
                await self.store.update_job_scores(score_updates)

        if not jobs:
            search_info = f"query='{query}'"
            if location:
                search_info += f", location='{location}'"
            if remote_only:
                search_info += ", remote_only=True"
            if min_salary:
                search_info += f", min_salary=${min_salary:,}"
            if date_filter != DatePosted.ANY:
                search_info += f", date_posted='{date_posted}'"
            return f"No jobs found ({search_info}). Try broader terms, different location, remove salary filter, or expand date range."

        # Format results
        search_desc = query
        if remote_only:
            search_desc += " (remote)"
        if location and not remote_only:
            search_desc += f" in {location}"
        if date_filter != DatePosted.ANY:
            date_labels = {
                "yesterday": "last 24 hours",
                "3days": "last 3 days",
                "week": "last week",
                "month": "last month",
            }
            search_desc += f" [{date_labels.get(date_posted.lower(), date_posted)}]"

        lines = [f"Found {len(jobs)} jobs matching '{search_desc}':"]
        for i, job in enumerate(jobs, 1):
            salary = ""
            if job.salary_min and job.salary_max:
                salary = f" | ${job.salary_min:,}-${job.salary_max:,}"
            score_str = f" | Match: {round(job.score * 100, 1)}%" if job.score else ""
            link = f" | 🔗 {job.url}" if job.url else ""
            lines.append(
                f"{i}. {job.title} at {job.company} ({job.location}){salary}{score_str}{link} [ID: {job.id[:8]}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Composite sorting: recency (40%) + profile match (60%)
    # ------------------------------------------------------------------

    def _sort_jobs_composite(self, jobs: list) -> list:
        """Sort jobs by a composite score combining recency and profile match.

        - Recency (40%): How recently the job was posted. Jobs posted today get
          1.0; jobs 30+ days old get 0.0.  Jobs without a posted_at date get 0.3
          (neutral — don't penalise, don't reward).
        - Profile match (60%): Cosine similarity between the job's embedding and
          the current user profile's embedding.  Jobs without an embedding get
          0.0 and sort to the bottom.

        The composite score is stored on each job's ``score`` field so it's
        visible in formatted output.
        """
        from datetime import datetime, timezone

        from .tools import get_current_profile

        profile = get_current_profile()
        profile_embedding = profile.embedding if profile else None
        now = datetime.utcnow()

        RECENCY_WEIGHT = 0.40
        MATCH_WEIGHT = 0.60

        for job in jobs:
            # --- Recency score ---
            if job.posted_at:
                age_days = max((now - job.posted_at).total_seconds() / 86400, 0)
                # Linear decay over 30 days, clamped to [0, 1]
                recency = max(1.0 - age_days / 30.0, 0.0)
            else:
                recency = 0.3  # neutral

            # --- Profile match score (cosine similarity) ---
            match_score = 0.0
            if profile_embedding and job.embedding:
                match_score = self._cosine_similarity(profile_embedding, job.embedding)

            # --- Composite ---
            job.score = round(RECENCY_WEIGHT * recency + MATCH_WEIGHT * match_score, 4)

        # Sort descending by composite score
        jobs.sort(key=lambda j: j.score or 0.0, reverse=True)
        return jobs

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def list_saved_jobs(self, limit: int = 20) -> str:
        """List all saved job listings.

        Args:
            limit: Maximum number of jobs to return.

        Returns:
            A formatted string with saved job listings.
        """
        jobs = await self.store.list_all(limit=limit)
        if not jobs:
            return "No saved jobs. Use search_jobs to find new opportunities."

        lines = [f"Saved jobs ({len(jobs)}):"]
        for i, job in enumerate(jobs, 1):
            link = f" | 🔗 {job.url}" if job.url else ""
            lines.append(
                f"{i}. [{job.status.value}] {job.title} at {job.company}{link} [ID: {job.id[:8]}]"
            )
        return "\n".join(lines)

    async def get_job_details(self, job_id: str) -> str:
        """Get detailed information about a specific job.

        Args:
            job_id: The unique identifier of the job (can be partial, at least 8 chars).

        Returns:
            Full job details or error message if not found.
        """
        all_jobs = await self.store.list_all(limit=1000)
        job = None
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                job = j
                break

        if not job:
            return f"Job with ID '{job_id}' not found."

        salary = "Not specified"
        if job.salary_min and job.salary_max:
            salary = f"${job.salary_min:,} - ${job.salary_max:,}"

        return f"""Job Details:
- Title: {job.title}
- Company: {job.company}
- Location: {job.location}
- Type: {job.job_type or 'Not specified'}
- Experience: {job.experience_level or 'Not specified'}
- Salary: {salary}
- Skills: {', '.join(job.skills) if job.skills else 'Not specified'}
- Status: {job.status.value}
- URL: {job.url or 'Not available'}

Description:
{job.description}"""

    async def mark_job_applied(self, job_id: str) -> str:
        """Mark a job as applied.

        Args:
            job_id: The unique identifier of the job.

        Returns:
            Status message.
        """
        all_jobs = await self.store.list_all(limit=1000)
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                await self.store.update_status(j.id, JobStatus.APPLIED)
                return f"Marked '{j.title}' at {j.company} as applied."
        return f"Job with ID '{job_id}' not found."

    async def mark_job_rejected(self, job_id: str) -> str:
        """Mark a job as not interested/rejected.

        Args:
            job_id: The unique identifier of the job.

        Returns:
            Status message.
        """
        all_jobs = await self.store.list_all(limit=1000)
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                await self.store.update_status(j.id, JobStatus.REJECTED)
                return f"Marked '{j.title}' at {j.company} as rejected."
        return f"Job with ID '{job_id}' not found."

    async def set_user_profile(
        self,
        name: str,
        resume_text: str,
        skills: list[str],
        current_title: Optional[str] = None,
        desired_titles: Optional[list[str]] = None,
        preferred_locations: Optional[list[str]] = None,
        remote_preference: str = "flexible",
        min_salary: Optional[int] = None,
        years_experience: Optional[int] = None,
    ) -> str:
        """Set or update the user profile with resume and preferences.

        Args:
            name: User's name.
            resume_text: Full resume text or summary of experience.
            skills: List of skills (e.g., ["Python", "AWS", "Machine Learning"]).
            current_title: Current job title.
            desired_titles: List of desired job titles.
            preferred_locations: List of preferred locations.
            remote_preference: One of "remote", "hybrid", "onsite", "flexible".
            min_salary: Minimum acceptable salary.
            years_experience: Years of professional experience.

        Returns:
            Confirmation message.
        """
        profile = UserProfile(
            name=name,
            resume_text=resume_text,
            skills=skills,
            current_title=current_title or "",
            desired_titles=desired_titles or [],
            preferred_locations=preferred_locations or [],
            remote_preference=remote_preference,
            min_salary=min_salary,
            years_experience=years_experience,
        )

        profile = await self.ranking_service.embed_user_profile(profile)
        set_current_profile(profile)
        await self.store.save_profile(profile)

        return f"Profile set for {name} with {len(skills)} skills. Ready to rank jobs."

    async def upload_resume(
        self,
        file_path: Optional[str] = None,
        file_data: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> str:
        """Parse a resume file and create a user profile from it.

        Supports PDF, DOCX, and TXT files. Extracts skills, experience,
        and creates an embedding for job matching.

        Args:
            file_path: Path to the resume file (e.g., "/path/to/resume.pdf").
            file_data: Base64-encoded file content (alternative to file_path).
            file_type: File type when using file_data: "pdf", "docx", or "txt".

        Returns:
            Confirmation with extracted profile details.
        """
        if not file_path and not (file_data and file_type):
            return "Please provide either file_path or (file_data and file_type)."

        try:
            parser = ResumeParser()
            parsed = await parser.parse_and_extract(
                file_path=file_path,
                file_data=file_data,
                file_type=file_type,
                use_llm=False,
            )

            profile = parsed.to_user_profile()
            profile = await self.ranking_service.embed_user_profile(profile)
            set_current_profile(profile)
            await self.store.save_profile(profile)

            skill_list = ", ".join(parsed.skills[:10])
            if len(parsed.skills) > 10:
                skill_list += f" (+{len(parsed.skills) - 10} more)"

            response = "Resume parsed successfully!\n\n"
            if parsed.name:
                response += f"Name: {parsed.name}\n"
            if parsed.email:
                response += f"Email: {parsed.email}\n"
            if parsed.current_title:
                response += f"Title: {parsed.current_title}\n"
            if parsed.years_experience:
                response += f"Experience: {parsed.years_experience} years\n"
            response += f"Skills extracted: {skill_list}\n\n"
            response += "Profile created with embedding. Ready to rank jobs!"
            return response

        except FileNotFoundError:
            return f"File not found: {file_path}"
        except ValueError as e:
            return f"Error parsing resume: {e}"
        except Exception as e:
            return f"Failed to parse resume: {e}"

    async def rank_saved_jobs(self, top_k: int = 10) -> str:
        """Rank all saved jobs against the user profile and return top matches.

        Args:
            top_k: Number of top-ranked jobs to return.

        Returns:
            A formatted string with ranked jobs, scores, and justifications.
        """
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Please provide your resume or skills first using set_user_profile."

        jobs = await self.store.list_all()
        if not jobs:
            return "No saved jobs to rank. Use search_jobs to find jobs first."

        ranked_jobs = await self.ranking_service.rank_jobs(jobs, profile, top_k=top_k)
        if not ranked_jobs:
            return "Could not rank jobs. Please try again."

        # Persist ranking scores to DB so cron / API can access them
        score_updates = [(rj.job.id, rj.score) for rj in ranked_jobs if rj.score is not None]
        if score_updates:
            await self.store.update_job_scores(score_updates)

        lines = [f"Top {len(ranked_jobs)} job matches for {profile.name}:"]
        for i, rj in enumerate(ranked_jobs, 1):
            job = rj.job
            score_pct = round(rj.score * 100, 1)
            lines.append(f"\n{i}. {job.title} at {job.company} - Score: {score_pct}%")
            lines.append(f"   {rj.justification}")
            if job.url:
                lines.append(f"   🔗 {job.url}")
            lines.append(f"   [ID: {job.id[:8]}]")

        return "\n".join(lines)

    async def get_profile(self) -> str:
        """Get the current user profile including resume content.

        Returns:
            Current user profile summary with resume text, or message if not set.
        """
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Use set_user_profile or upload a resume via the 📎 button."

        result = f"""User Profile:
- Name: {profile.name}
- Current Title: {profile.current_title or 'Not specified'}
- Years Experience: {profile.years_experience or 'Not specified'}
- Skills: {', '.join(profile.skills) if profile.skills else 'None'}
- Desired Titles: {', '.join(profile.desired_titles) if profile.desired_titles else 'Any'}
- Preferred Locations: {', '.join(profile.preferred_locations) if profile.preferred_locations else 'Any'}
- Remote Preference: {profile.remote_preference}
- Min Salary: {'$' + f'{profile.min_salary:,}' if profile.min_salary else 'Not specified'}
- Profile Embedding: {'Ready' if profile.embedding else 'Not generated'}"""

        if profile.resume_text:
            result += f"\n\nResume Content:\n{profile.resume_text}"

        return result

    async def send_job_notifications(
        self,
        top_k: int = 10,
        title: str = "New Job Matches",
    ) -> str:
        """Send notifications with the top ranked job matches to configured channels.

        Args:
            top_k: Number of top matches to include (default 10).
            title: Notification title/subject.

        Returns:
            Status message indicating which channels received the notification.
        """
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Upload a resume first."

        jobs = await self.store.list_all()
        if not jobs:
            return "No jobs to notify about. Search for jobs first."

        ranked_jobs = await self.ranking_service.rank_jobs(jobs, profile, top_k=top_k)
        if not ranked_jobs:
            return "No ranked matches found."

        results = await self.notification_service.send_job_matches(
            ranked_jobs=ranked_jobs,
            profile=profile,
            title=title,
        )

        success_channels = [ch for ch, ok in results.items() if ok]
        failed_channels = [ch for ch, ok in results.items() if not ok]

        response = f"Sent {len(ranked_jobs)} job matches via: {', '.join(success_channels) if success_channels else 'none'}"
        if failed_channels:
            response += f"\nFailed channels: {', '.join(failed_channels)}"
        return response

    async def provide_feedback(
        self,
        job_id: str,
        feedback: str,
        notes: Optional[str] = None,
    ) -> str:
        """Provide feedback on a job match to improve future recommendations.

        Args:
            job_id: The job ID (can be partial, at least 8 characters).
            feedback: One of: "good_fit", "not_relevant", "tailor_resume",
                     "draft_cover_letter", "already_applied", "company_blacklist".
            notes: Optional additional notes.

        Returns:
            Confirmation message.
        """
        feedback_map = {
            "good_fit": FeedbackType.GOOD_FIT,
            "not_relevant": FeedbackType.NOT_RELEVANT,
            "tailor_resume": FeedbackType.TAILOR_RESUME,
            "draft_cover_letter": FeedbackType.DRAFT_COVER_LETTER,
            "already_applied": FeedbackType.ALREADY_APPLIED,
            "company_blacklist": FeedbackType.COMPANY_BLACKLIST,
        }

        feedback_type = feedback_map.get(feedback.lower().replace(" ", "_"))
        if not feedback_type:
            return f"Invalid feedback type '{feedback}'. Use: {', '.join(feedback_map.keys())}"

        all_jobs = await self.store.list_all(limit=1000)
        job = None
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                job = j
                break

        if not job:
            return f"Job with ID '{job_id}' not found."

        job_feedback = JobFeedback(
            job_id=job.id,
            feedback_type=feedback_type,
            notes=notes,
        )
        await self.store.save_feedback(job_feedback)

        messages = {
            FeedbackType.TAILOR_RESUME: f"Feedback saved for '{job.title}'. Hand off to application_prep_agent for tailored resume suggestions.",
            FeedbackType.DRAFT_COVER_LETTER: f"Feedback saved for '{job.title}'. Hand off to application_prep_agent for a cover letter.",
            FeedbackType.GOOD_FIT: f"Marked '{job.title}' at {job.company} as a good fit. Consider preparing an application.",
            FeedbackType.NOT_RELEVANT: f"Marked '{job.title}' as not relevant. This will help improve future matches.",
            FeedbackType.COMPANY_BLACKLIST: f"Added {job.company} to your blacklist. Future searches will deprioritize this company.",
        }
        return messages.get(feedback_type, f"Feedback recorded for '{job.title}'.")

    async def suggest_search_titles(self) -> str:
        """Analyze the user's profile and suggest optimized job search title keywords.

        Combines years of experience, technical skills, desired roles, current title,
        and resume content to generate smart search keywords tailored to the user's
        background. Call this before running searches to get the best keywords.

        Returns:
            A categorized list of suggested search title keywords with explanations.
        """
        profile = get_current_profile()
        if not profile:
            return (
                "No user profile set. Please upload a resume or create a profile "
                "first so I can suggest relevant search titles."
            )

        # Build a profile summary for the LLM
        profile_parts = []
        if profile.current_title:
            profile_parts.append(f"Current Title: {profile.current_title}")
        if profile.years_experience is not None:
            profile_parts.append(f"Years of Experience: {profile.years_experience}")
        if profile.skills:
            profile_parts.append(f"Technical Skills: {', '.join(profile.skills)}")
        if profile.desired_titles:
            profile_parts.append(f"Desired Roles: {', '.join(profile.desired_titles)}")
        if profile.industries:
            profile_parts.append(f"Industries: {', '.join(profile.industries)}")
        if profile.preferred_locations:
            profile_parts.append(f"Preferred Locations: {', '.join(profile.preferred_locations)}")
        if profile.remote_preference and profile.remote_preference != "flexible":
            profile_parts.append(f"Remote Preference: {profile.remote_preference}")
        if profile.min_salary:
            profile_parts.append(f"Minimum Salary: ${profile.min_salary:,}")

        # Include resume excerpt for deeper context (truncate to avoid token bloat)
        resume_excerpt = ""
        if profile.resume_text:
            resume_excerpt = profile.resume_text[:3000]
            profile_parts.append(f"\nResume Content (excerpt):\n{resume_excerpt}")

        profile_summary = "\n".join(profile_parts)

        prompt = f"""\
Analyze this job seeker's profile and suggest optimized search title keywords \
they should use when searching for jobs. Consider ALL of the following:

1. **Experience level** — Map years of experience to seniority prefixes \
   (e.g., 0-2yr→Junior/Associate, 3-5yr→Mid-level, 5-8yr→Senior, \
   8-12yr→Staff/Lead, 12+yr→Principal/Director/VP)
2. **Technical skills** — Generate role titles tied to their strongest \
   technologies (e.g., Python + ML → "Machine Learning Engineer")
3. **Desired roles** — Include these directly, plus common variations and \
   synonyms used in job postings
4. **Resume content** — Infer specializations from project descriptions, \
   past titles, and accomplishments
5. **Career progression** — Suggest both lateral moves and logical next-step \
   roles

PROFILE:
{profile_summary}

Return the suggestions in this exact format:

BASED ON DESIRED ROLES:
- [keyword 1]
- [keyword 2]
...

BASED ON EXPERIENCE LEVEL ({profile.years_experience or 'unknown'} years):
- [keyword 1]
- [keyword 2]
...

BASED ON TECHNICAL SKILLS:
- [keyword 1]
- [keyword 2]
...

INFERRED FROM RESUME:
- [keyword 1]
- [keyword 2]
...

CAREER GROWTH OPPORTUNITIES:
- [keyword 1]
- [keyword 2]
...

For each category, suggest 3-6 specific job title keywords optimized for \
job search engines (concise, like actual job posting titles). After the \
lists, add a brief RECOMMENDED SEARCH STRATEGY section with 2-3 tips \
specific to this person's profile."""

        # Use the direct OpenAI client if available, otherwise return a
        # rule-based fallback
        if self._openai_client and self._deployment:
            try:
                resp = await self._openai_client.chat.completions.create(
                    model=self._deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a career advisor and job search strategist. "
                                "Generate specific, actionable job title keywords "
                                "optimized for job search engines."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=1000,
                    temperature=0.7,
                )
                return resp.choices[0].message.content or "No suggestions generated."
            except Exception as e:
                logger.warning("LLM suggest_search_titles failed: %s", e)
                # Fall through to rule-based fallback

        # Rule-based fallback when no OpenAI client is available
        return self._suggest_titles_fallback(profile)

    @staticmethod
    def _suggest_titles_fallback(profile: UserProfile) -> str:
        """Generate rule-based search title suggestions without an LLM."""
        lines = ["Suggested search title keywords based on your profile:\n"]

        # Desired roles (direct)
        if profile.desired_titles:
            lines.append("BASED ON DESIRED ROLES:")
            for t in profile.desired_titles:
                lines.append(f"- {t}")
            lines.append("")

        # Experience-level titles
        yoe = profile.years_experience or 0
        if yoe <= 2:
            prefixes = ["Junior", "Associate", "Entry-Level"]
        elif yoe <= 5:
            prefixes = ["Mid-Level", ""]
        elif yoe <= 8:
            prefixes = ["Senior"]
        elif yoe <= 12:
            prefixes = ["Staff", "Lead", "Senior"]
        else:
            prefixes = ["Principal", "Staff", "Director", "Head of", "VP"]

        base_title = profile.current_title or (
            profile.desired_titles[0] if profile.desired_titles else "Engineer"
        )
        lines.append(f"BASED ON EXPERIENCE LEVEL ({yoe} years):")
        seen = set()
        for p in prefixes:
            suggestion = f"{p} {base_title}".strip()
            if suggestion not in seen:
                lines.append(f"- {suggestion}")
                seen.add(suggestion)
        lines.append("")

        # Skill-based titles
        skill_title_map = {
            "python": ["Python Developer", "Python Engineer"],
            "javascript": ["JavaScript Developer", "Frontend Engineer"],
            "typescript": ["TypeScript Developer", "Frontend Engineer"],
            "react": ["React Developer", "Frontend Engineer"],
            "java": ["Java Developer", "Java Engineer"],
            "c#": [".NET Developer", "C# Engineer"],
            ".net": [".NET Developer", ".NET Engineer"],
            "aws": ["Cloud Engineer", "AWS Engineer"],
            "azure": ["Cloud Engineer", "Azure Engineer"],
            "kubernetes": ["DevOps Engineer", "Platform Engineer"],
            "docker": ["DevOps Engineer", "Platform Engineer"],
            "machine learning": ["ML Engineer", "Machine Learning Engineer"],
            "data science": ["Data Scientist", "Data Analyst"],
            "sql": ["Data Engineer", "Database Developer"],
            "ai": ["AI Engineer", "ML Engineer"],
        }

        skill_suggestions = set()
        for skill in profile.skills:
            for key, titles in skill_title_map.items():
                if key in skill.lower():
                    skill_suggestions.update(titles)

        if skill_suggestions:
            lines.append("BASED ON TECHNICAL SKILLS:")
            for s in sorted(skill_suggestions)[:8]:
                lines.append(f"- {s}")
            lines.append("")

        lines.append(
            "TIP: Use these keywords with search_jobs for more targeted results. "
            "Try multiple searches with different keywords for broader coverage."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Application Prep tools
# ---------------------------------------------------------------------------

class AppPrepTools:
    """Tool implementations for the Application Prep Agent."""

    def __init__(
        self,
        store: JobStore,
        application_prep_service: ApplicationPrepService,
    ):
        self.store = store
        self.app_prep = application_prep_service

    async def prepare_application(self, job_id: str) -> str:
        """Generate a tailored application package for a specific job.

        Creates resume diff suggestions, cover letter, and intro email.

        Args:
            job_id: The job ID (can be partial, at least 8 characters).

        Returns:
            Formatted application package for review.
        """
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Upload a resume first via the job search agent."

        all_jobs = await self.store.list_all(limit=1000)
        job = None
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                job = j
                break

        if not job:
            return f"Job with ID '{job_id}' not found."

        try:
            package = await self.app_prep.prepare_application(job, profile)
            await self.store.save_application_package(package)
            summary = await self.app_prep.format_package_summary(package)
            return f"Application package created (ID: {package.id[:8]}):\n\n{summary}"
        except Exception as e:
            logger.error(f"Failed to prepare application: {e}")
            return f"Error preparing application: {e}"

    async def get_application_package(self, package_id: str) -> str:
        """Retrieve a previously generated application package.

        Args:
            package_id: The package ID (can be partial).

        Returns:
            Formatted application package or error message.
        """
        packages = await self.store.list_application_packages(limit=100)
        package = None
        for p in packages:
            if p.id.startswith(package_id) or package_id in p.id:
                package = p
                break

        if not package:
            return f"Application package '{package_id}' not found. Use prepare_application to create one."

        return await self.app_prep.format_package_summary(package)

    async def analyze_job_fit(self, job_id: str) -> str:
        """Analyze how well the user's profile matches a specific job.

        Provides a detailed fit analysis including matching skills, gaps,
        and overall compatibility assessment.

        Args:
            job_id: The job ID (can be partial, at least 8 characters).

        Returns:
            Detailed fit analysis.
        """
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Upload a resume first via the job search agent."

        all_jobs = await self.store.list_all(limit=1000)
        job = None
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                job = j
                break

        if not job:
            return f"Job with ID '{job_id}' not found."

        # Compute skill overlap
        profile_skills = {s.lower() for s in profile.skills}
        job_skills = {s.lower() for s in (job.skills or [])}
        matching = profile_skills & job_skills
        missing = job_skills - profile_skills
        extra = profile_skills - job_skills

        lines = [
            f"Job Fit Analysis: {job.title} at {job.company}",
            "=" * 50,
            "",
            f"Matching Skills ({len(matching)}):",
        ]
        if matching:
            lines.append("  " + ", ".join(sorted(matching)))
        else:
            lines.append("  None directly matching (check description for related skills)")

        lines.append(f"\nMissing Skills ({len(missing)}):")
        if missing:
            lines.append("  " + ", ".join(sorted(missing)))
        else:
            lines.append("  None - you have all required skills!")

        lines.append(f"\nAdditional Skills You Bring ({len(extra)}):")
        if extra:
            lines.append("  " + ", ".join(sorted(list(extra)[:15])))

        # Location match
        loc_match = "Unknown"
        if profile.preferred_locations:
            if any(loc.lower() in job.location.lower() for loc in profile.preferred_locations):
                loc_match = "Yes - matches your preferred location"
            elif "remote" in job.location.lower():
                loc_match = "Remote position available"
            else:
                loc_match = f"No - job is in {job.location}"
        lines.append(f"\nLocation Match: {loc_match}")

        # Salary match
        if job.salary_min and profile.min_salary:
            if job.salary_min >= profile.min_salary:
                lines.append(f"Salary Match: Yes - ${job.salary_min:,}+ meets your minimum of ${profile.min_salary:,}")
            else:
                lines.append(f"Salary Match: Below target - ${job.salary_min:,} vs your minimum ${profile.min_salary:,}")
        else:
            lines.append("Salary Match: Salary not disclosed")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Coordinator executor (multi-agent router)
# ---------------------------------------------------------------------------

class CoordinatorExecutor(Executor):
    """Routes user requests to specialist sub-agents.

    Architecture:
        classifier  →  intent
          ├── JOB_SEARCH   →  job_search_agent   (12 tools)
          └── APP_PREP     →  application_prep_agent (3 tools)

    The classifier is a lightweight LLM call that outputs "JOB_SEARCH" or
    "APP_PREP".  The specialist agent then runs with the full conversation
    history and yields the response.
    """

    def __init__(
        self,
        client: AzureOpenAIChatClient,
        store: JobStore,
        provider: JobIngestionProvider,
        ranking_service: RankingService,
        notification_service: NotificationService,
        application_prep_service: Optional[ApplicationPrepService] = None,
        openai_client: Optional[AsyncAzureOpenAI] = None,
        deployment_name: str = "",
        id: str = "coordinator",
    ):
        self.store = store
        self._openai_client = openai_client  # Direct OpenAI client for logprobs
        self._deployment = deployment_name

        # --- Tool objects ---
        job_tools = JobSearchTools(
            store, provider, ranking_service, notification_service,
            openai_client=openai_client, deployment_name=deployment_name,
        )
        app_tools = AppPrepTools(store, application_prep_service) if application_prep_service else None

        # --- Classifier (no tools, cheap routing call) ---
        self.classifier = client.create_agent(
            name="coordinator",
            instructions=CLASSIFIER_INSTRUCTIONS,
        )

        # --- Job Search specialist ---
        self.job_search_agent = client.create_agent(
            name="job_search_agent",
            instructions=JOB_SEARCH_INSTRUCTIONS,
            tools=[
                job_tools.search_jobs,
                job_tools.list_saved_jobs,
                job_tools.get_job_details,
                job_tools.mark_job_applied,
                job_tools.mark_job_rejected,
                job_tools.set_user_profile,
                job_tools.upload_resume,
                job_tools.rank_saved_jobs,
                job_tools.get_profile,
                job_tools.send_job_notifications,
                job_tools.provide_feedback,
                job_tools.suggest_search_titles,
            ],
        )

        # --- Application Prep specialist ---
        app_prep_tools_list = []
        if app_tools:
            app_prep_tools_list = [
                app_tools.prepare_application,
                app_tools.get_application_package,
                app_tools.analyze_job_fit,
            ]

        self.app_prep_agent = client.create_agent(
            name="application_prep_agent",
            instructions=APP_PREP_INSTRUCTIONS,
            tools=app_prep_tools_list,
        )

        super().__init__(id=id)
        self._initialized = False

    async def _ensure_profile(self) -> None:
        """Load the default user profile from the store on first request."""
        if self._initialized:
            return
        self._initialized = True
        try:
            profile = await self.store.get_default_profile()
            if profile:
                set_current_profile(profile)
                logger.info("Loaded profile for %s from database", profile.name)
        except Exception as exc:
            logger.warning("Could not load default profile: %s", exc)

    @handler
    async def handle(self, messages: list[ChatMessage], ctx: WorkflowContext[Any, Any]) -> None:
        await self._ensure_profile()

        request_start = time.time()
        usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Log the user's message for trace context
        user_msg = ""
        if messages:
            last = messages[-1]
            if last.contents:
                for c in last.contents:
                    if hasattr(c, "text") and c.text:
                        user_msg = c.text[:120]
                        break
        logger.info("[TRACE] User message: %s", user_msg)

        # 1. Classify intent — use direct OpenAI call for logprobs if available
        logger.info("[TRACE] Classifying user intent...")
        confidence = None
        intent = "JOB_SEARCH"  # default

        if self._openai_client and self._deployment:
            try:
                # Build condensed messages for classifier
                classify_messages = [
                    {"role": "system", "content": CLASSIFIER_INSTRUCTIONS},
                ]
                for m in messages[-3:]:  # last 3 for context
                    role = "user" if m.role == Role.USER else "assistant"
                    text = ""
                    if m.contents:
                        for c in m.contents:
                            if hasattr(c, "text") and c.text:
                                text = c.text
                                break
                    if text:
                        classify_messages.append({"role": role, "content": text})

                with _tracer.start_as_current_span("classifier") as span:
                    resp = await self._openai_client.chat.completions.create(
                        model=self._deployment,
                        messages=classify_messages,
                        logprobs=True,
                        top_logprobs=3,
                        max_completion_tokens=150,
                    )
                    choice = resp.choices[0]
                    intent = (choice.message.content or "JOB_SEARCH").strip().upper()

                    # Extract confidence from logprobs
                    if choice.logprobs and choice.logprobs.content:
                        first_token = choice.logprobs.content[0]
                        confidence = round(math.exp(first_token.logprob) * 100, 1)
                        # Log alternatives
                        alternatives = []
                        for alt in first_token.top_logprobs:
                            alt_prob = round(math.exp(alt.logprob) * 100, 1)
                            alternatives.append(f"{alt.token}={alt_prob}%")
                        span.set_attribute("classifier.alternatives", ", ".join(alternatives))

                    # Track classifier token usage
                    if resp.usage:
                        usage_totals["input_tokens"] += resp.usage.prompt_tokens
                        usage_totals["output_tokens"] += resp.usage.completion_tokens
                        usage_totals["total_tokens"] += resp.usage.total_tokens

                    span.set_attribute("classifier.intent", intent)
                    span.set_attribute("classifier.confidence_pct", confidence or 0)
                    span.set_attribute("classifier.user_message", user_msg[:200])

            except Exception as exc:
                logger.warning("[TRACE] Logprobs classifier failed, falling back: %s", exc)
                classification = await self.classifier.run(messages)
                intent = (classification.text or "").strip().upper()
        else:
            classification = await self.classifier.run(messages)
            intent = (classification.text or "").strip().upper()

        conf_str = f" (confidence: {confidence}%)" if confidence is not None else ""
        logger.info("[TRACE] Classifier → %s%s", intent, conf_str)

        # 2. Route to the appropriate specialist
        with _tracer.start_as_current_span("specialist_agent") as agent_span:
            if "APP_PREP" in intent:
                agent_name = "application_prep_agent"
                logger.info("[TRACE] Routing to Application Prep Agent (3 tools)")
                response = await self.app_prep_agent.run(messages)
            else:
                agent_name = "job_search_agent"
                logger.info("[TRACE] Routing to Job Search Agent (12 tools)")
                response = await self.job_search_agent.run(messages)

            agent_span.set_attribute("agent.name", agent_name)
            agent_span.set_attribute("agent.intent", intent)

        # Track specialist agent token usage
        if hasattr(response, "usage_details") and response.usage_details:
            ud = response.usage_details
            in_tok = getattr(ud, "input_token_count", 0) or 0
            out_tok = getattr(ud, "output_token_count", 0) or 0
            tot_tok = getattr(ud, "total_token_count", 0) or 0
            usage_totals["input_tokens"] += in_tok
            usage_totals["output_tokens"] += out_tok
            usage_totals["total_tokens"] += tot_tok

        # 3. Audit log — structured OTel spans for each tool call
        #    Agent Framework uses FunctionCallContent (attrs: name, call_id, arguments)
        #    and FunctionResultContent (attrs: call_id, result).
        tool_calls = []
        for message in response.messages:
            if not message.contents:
                continue
            for c in message.contents:
                # FunctionCallContent: type="function_call", attrs: name, call_id, arguments
                if getattr(c, "type", None) == "function_call":
                    tool_name = c.name
                    args_str = ""
                    if c.arguments:
                        args_str = str(c.arguments)[:500]

                    # Find the matching tool result (FunctionResultContent has .call_id and .result)
                    tool_result = ""
                    if c.call_id:
                        for rm in response.messages:
                            if rm.contents:
                                for rc in rm.contents:
                                    if getattr(rc, "call_id", None) == c.call_id and hasattr(rc, "result"):
                                        tool_result = str(rc.result or "")[:500]
                                        break

                    # Emit structured audit span
                    with _tracer.start_as_current_span("tool_call") as tool_span:
                        tool_span.set_attribute("tool.name", tool_name)
                        tool_span.set_attribute("tool.arguments", args_str)
                        tool_span.set_attribute("tool.result_preview", tool_result[:300])
                        tool_span.set_attribute("tool.agent", agent_name)
                        tool_span.set_attribute("audit.type", "tool_invocation")

                    tool_calls.append(tool_name)
                    logger.info("[TRACE] 🔧 Tool call: %s(%s)", tool_name, args_str[:200])

                elif getattr(c, "type", None) == "function_result":
                    result_preview = str(c.result or "")[:150]
                    logger.info("[TRACE] ← Tool result: %s", result_preview)

        # Summarize tool flow
        if tool_calls:
            logger.info("[TRACE] Tool chain: %s", " → ".join(tool_calls))
        else:
            logger.info("[TRACE] %s responded without tool calls", agent_name)

        # Token usage summary
        elapsed_ms = round((time.time() - request_start) * 1000)
        logger.info(
            "[TRACE] Tokens: %d in + %d out = %d total | %dms",
            usage_totals["input_tokens"],
            usage_totals["output_tokens"],
            usage_totals["total_tokens"],
            elapsed_ms,
        )

        logger.info("[TRACE] Response length: %d chars", len(response.text or ""))

        # 4. Yield final text + metadata for the HTTP response body
        #    Pack usage and confidence into a JSON metadata suffix that
        #    the webapp will parse and strip before rendering.
        metadata = {
            "usage": usage_totals,
            "elapsed_ms": elapsed_ms,
            "classifier_confidence": confidence,
            "agent": agent_name,
            "tool_count": len(tool_calls),
        }
        response_text = response.text or ""
        # Append metadata as a parseable suffix
        output = f"{response_text}\n<!--METADATA:{json.dumps(metadata)}-->"
        await ctx.yield_output(output)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

async def _init_services(use_database: bool = True):
    """Initialize shared services (store, provider, ranking, notifications, app prep)."""
    from .config import AppConfig

    config = AppConfig.load()

    # Store
    if use_database and config.database and config.database.is_configured:
        store = await get_store(config.database)
    else:
        store = InMemoryJobStore()

    provider = get_provider()
    ranking_service = get_ranking_service()
    notification_service = get_notification_service()
    application_prep_service = get_application_prep_service()

    return store, provider, ranking_service, notification_service, application_prep_service


def build_agent(
    client: AzureOpenAIChatClient,
    store: Optional[JobStore] = None,
    provider: Optional[JobIngestionProvider] = None,
    ranking_service: Optional[RankingService] = None,
    notification_service: Optional[NotificationService] = None,
    application_prep_service: Optional[ApplicationPrepService] = None,
    openai_client: Optional[AsyncAzureOpenAI] = None,
    deployment_name: str = "",
):
    """Build the multi-agent job workflow (sync, for testing).

    Args:
        client: Azure OpenAI client.
        store: Job store (defaults to in-memory).
        provider: Job search provider.
        ranking_service: Ranking / embedding service.
        notification_service: Notification delivery service.
        application_prep_service: Application material generator.
        openai_client: Direct AsyncAzureOpenAI client for logprobs-based classification.
        deployment_name: Azure OpenAI deployment name for the direct client.
    """
    if store is None:
        store = InMemoryJobStore()
    if provider is None:
        provider = get_provider()
    if ranking_service is None:
        ranking_service = get_ranking_service()
    if notification_service is None:
        notification_service = get_notification_service()
    if application_prep_service is None:
        application_prep_service = get_application_prep_service()

    workflow = (
        WorkflowBuilder()
        .register_executor(
            lambda: CoordinatorExecutor(
                client, store, provider, ranking_service,
                notification_service, application_prep_service,
                openai_client=openai_client,
                deployment_name=deployment_name,
            ),
            name="coordinator",
        )
        .set_start_executor("coordinator")
        .build()
    )
    return workflow.as_agent()


async def create_agent(
    client: AzureOpenAIChatClient,
    use_database: bool = True,
) -> tuple:
    """Create the multi-agent job workflow with async initialization.

    Architecture:
        CoordinatorExecutor
          ├── classifier            (lightweight intent routing with logprobs)
          ├── job_search_agent      (11 tools — search, rank, profile, etc.)
          └── application_prep_agent (3 tools — resume, cover letter, email)

    Args:
        client: Azure OpenAI chat client.
        use_database: If True, uses PostgreSQL when configured; otherwise in-memory.

    Returns:
        Tuple of (agent, store, ranking_service) so callers can reuse services.
    """
    store, provider, ranking_service, notification_service, app_prep_service = (
        await _init_services(use_database)
    )

    # Build a direct AsyncAzureOpenAI client for logprobs-based classification
    import os
    openai_client = None
    deployment_name = ""
    try:
        from .config import AppConfig
        config = AppConfig.load()
        oc = config.azure_openai
        deployment_name = oc.deployment_name
        if oc.api_key:
            openai_client = AsyncAzureOpenAI(
                azure_endpoint=oc.endpoint,
                api_key=oc.api_key,
                api_version=oc.api_version,
            )
        else:
            from azure.identity.aio import DefaultAzureCredential as AsyncCredential
            openai_client = AsyncAzureOpenAI(
                azure_endpoint=oc.endpoint,
                azure_ad_token_provider=AsyncCredential(),
                api_version=oc.api_version,
            )
        logger.info("Direct OpenAI client created for logprobs classification")
    except Exception as exc:
        logger.warning("Could not create direct OpenAI client (logprobs disabled): %s", exc)

    agent = build_agent(
        client, store, provider, ranking_service,
        notification_service, app_prep_service,
        openai_client=openai_client,
        deployment_name=deployment_name,
    )
    return agent, store, ranking_service
