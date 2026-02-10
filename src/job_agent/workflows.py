import logging
from typing import Any, Optional
from uuid import uuid4

from agent_framework import (
    AgentRunResponseUpdate,
    AgentRunUpdateEvent,
    ChatMessage,
    Executor,
    Role,
    TextContent,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.azure import AzureOpenAIChatClient

from .application_prep import ApplicationPrepService, get_application_prep_service
from .models import FeedbackType, JobFeedback, JobSearchCriteria, UserProfile
from .notifications import NotificationService, get_notification_service
from .providers import JobIngestionProvider, MockJobProvider, get_provider
from .ranking import RankingService, get_ranking_service
from .resume_parser import ResumeParser, ParsedResume
from .store import InMemoryJobStore, JobStore, get_store
from .tools import get_current_profile, set_current_profile

logger = logging.getLogger(__name__)


class CoordinatorExecutor(Executor):
    """Single-node coordinator that delegates reasoning to a chat agent."""

    def __init__(
        self,
        client: AzureOpenAIChatClient,
        store: JobStore,
        provider: JobIngestionProvider,
        ranking_service: Optional[RankingService] = None,
        notification_service: Optional[NotificationService] = None,
        application_prep_service: Optional[ApplicationPrepService] = None,
        id: str = "coordinator",
    ):
        self.store = store
        self.provider = provider
        self.ranking_service = ranking_service or get_ranking_service()
        self.notification_service = notification_service or get_notification_service()
        self.application_prep_service = application_prep_service or get_application_prep_service()
        
        # Create agent with tools for job operations
        self.agent = client.create_agent(
            name="JobCoordinator",
            instructions="""You are a job search assistant that helps users find and manage job opportunities.

You have access to tools for searching jobs, managing saved jobs, setting user profiles, ranking jobs,
sending notifications, and preparing job applications.

SEARCH TIPS:
- For remote jobs: Use remote_only=True, NOT location="Remote"
- For broader results: Try multiple search terms (e.g., "AI engineer", "machine learning lead", "ML director")
- For senior roles: Search for "director", "VP", "head of", "principal", "staff" combined with the domain
- Many high-paying jobs don't list salary publicly - suggest removing min_salary filter if few results
- If a specific search returns few results, automatically try broader terms

DATE FILTERING:
- Use date_posted parameter to filter by posting recency
- "yesterday" or "today" = last 24 hours
- "3days" = last 3 days  
- "week" = last 7 days
- "month" = last 30 days
- Omit for all jobs regardless of posting date

RESUME HANDLING:
- Use upload_resume when user provides a PDF/DOCX file or base64-encoded resume
- Use set_user_profile when user describes their background in text
- The resume parser extracts skills, experience, and creates an embedding automatically

WORKFLOW:
1. When user describes their background, use set_user_profile to save it
2. Search for jobs matching their interests
3. Use rank_saved_jobs to show best matches based on their profile
4. Help track applications with mark_job_applied/mark_job_rejected
5. Use send_job_notifications to deliver top matches via email/Teams/Slack
6. Use provide_feedback to capture user feedback on job matches
7. Use prepare_application to generate tailored resume suggestions, cover letter, and intro email

APPLICATION PREP:
- When user wants to apply for a specific job, use prepare_application
- This generates resume diff suggestions (not full rewrite), cover letter, and intro email
- User can review and approve the package before applying

When presenting jobs, format them clearly with title, company, location, and salary if available.
Be proactive - suggest alternative searches if results are limited.""",
            tools=[
                self.search_jobs,
                self.list_saved_jobs,
                self.get_job_details,
                self.mark_job_applied,
                self.mark_job_rejected,
                self.set_user_profile,
                self.upload_resume,
                self.rank_saved_jobs,
                self.get_profile,
                self.send_job_notifications,
                self.provide_feedback,
                self.prepare_application,
                self.get_application_package,
            ],
        )
        super().__init__(id=id)
    
    async def initialize(self):
        """Load default profile from store if available."""
        profile = await self.store.get_default_profile()
        if profile:
            set_current_profile(profile)
            logger.info(f"Loaded profile for {profile.name} from database")
    
    async def search_jobs(
        self,
        query: str,
        location: Optional[str] = None,
        remote_only: bool = False,
        min_salary: Optional[int] = None,
        max_results: int = 10,
        date_posted: Optional[str] = None,
    ) -> str:
        """
        Search for new job listings matching the criteria.
        
        Args:
            query: Job search query - use role/title keywords (e.g., "AI engineer", "machine learning", 
                   "python developer"). For broader results, try variations like "ML engineer" or "data scientist".
            location: Geographic location (e.g., "San Francisco, CA", "New York", "United States"). 
                      Do NOT use "Remote" here - use remote_only=True instead.
            remote_only: Set to True for remote/work-from-home positions. This filters for remote jobs.
            min_salary: Minimum yearly salary requirement (e.g., 150000 for $150K). Note: Many job postings 
                        don't include salary, so high minimums may filter out valid jobs.
            max_results: Maximum number of results to return (default 10, max 100)
            date_posted: Filter by posting date. Options: "yesterday" (last 24 hours), "3days" (last 3 days), 
                         "week" (last 7 days), "month" (last 30 days), or None for any time.
            
        Returns:
            A formatted string with matching job listings
            
        Tips:
            - For executive/senior roles, try broader terms: "AI", "machine learning", "director engineering"
            - For remote jobs, set remote_only=True and optionally location="United States"
            - If few results, try variations of the job title or remove salary filter
            - Use date_posted="yesterday" for jobs posted in the last 24 hours
        """
        from .models import DatePosted
        
        # Detect if user passed "remote" as location and convert to remote_only
        if location and location.lower() in ("remote", "remote only", "work from home"):
            remote_only = True
            location = "United States"  # Default to US for remote searches
        
        # Parse date_posted string to DatePosted enum
        date_filter = DatePosted.ANY
        if date_posted:
            date_mapping = {
                "yesterday": DatePosted.YESTERDAY,
                "today": DatePosted.YESTERDAY,  # Alias
                "24hours": DatePosted.YESTERDAY,  # Alias
                "3days": DatePosted.LAST_3_DAYS,
                "3 days": DatePosted.LAST_3_DAYS,  # Alias
                "week": DatePosted.LAST_WEEK,
                "7days": DatePosted.LAST_WEEK,  # Alias
                "month": DatePosted.LAST_MONTH,
                "30days": DatePosted.LAST_MONTH,  # Alias
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
        
        # Generate embeddings for the jobs and persist them
        if jobs:
            jobs = await self.ranking_service.embed_jobs(jobs)
            # Update embeddings in the store
            job_embeddings = [(j.id, j.embedding) for j in jobs if j.embedding]
            if job_embeddings:
                await self.store.update_job_embeddings(job_embeddings)
        
        if not jobs:
            # Provide helpful context about what was searched
            search_info = f"query='{query}'"
            if location:
                search_info += f", location='{location}'"
            if remote_only:
                search_info += ", remote_only=True"
            if min_salary:
                search_info += f", min_salary=${min_salary:,}"
            if date_filter != DatePosted.ANY:
                search_info += f", date_posted='{date_posted}'"
            
            suggestion = "Try broader search terms, different location, remove the salary filter, or expand the date range."
            return f"No jobs found ({search_info}). {suggestion}"
        
        # Build results with helpful summary
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
            lines.append(
                f"{i}. {job.title} at {job.company} ({job.location}){salary} [ID: {job.id[:8]}]"
            )
        return "\n".join(lines)
    
    async def list_saved_jobs(self, limit: int = 20) -> str:
        """
        List all saved job listings.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            A formatted string with saved job listings
        """
        jobs = await self.store.list_all(limit=limit)
        
        if not jobs:
            return "No saved jobs. Use search_jobs to find new opportunities."
        
        lines = [f"Saved jobs ({len(jobs)}):"]
        for i, job in enumerate(jobs, 1):
            lines.append(
                f"{i}. [{job.status.value}] {job.title} at {job.company} [ID: {job.id[:8]}]"
            )
        return "\n".join(lines)
    
    async def get_job_details(self, job_id: str) -> str:
        """
        Get detailed information about a specific job.
        
        Args:
            job_id: The unique identifier of the job (can be partial, at least 8 chars)
            
        Returns:
            Full job details or error message if not found
        """
        # Support partial ID matching
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
        
        return f"""
Job Details:
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
{job.description}
""".strip()
    
    async def mark_job_applied(self, job_id: str) -> str:
        """
        Mark a job as applied.
        
        Args:
            job_id: The unique identifier of the job
            
        Returns:
            Status message
        """
        from .models import JobStatus
        
        all_jobs = await self.store.list_all(limit=1000)
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                await self.store.update_status(j.id, JobStatus.APPLIED)
                return f"Marked '{j.title}' at {j.company} as applied."
        
        return f"Job with ID '{job_id}' not found."
    
    async def mark_job_rejected(self, job_id: str) -> str:
        """
        Mark a job as not interested/rejected.
        
        Args:
            job_id: The unique identifier of the job
            
        Returns:
            Status message
        """
        from .models import JobStatus
        
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
        """
        Set or update the user profile with resume and preferences.
        
        Args:
            name: User's name
            resume_text: Full resume text or summary of experience
            skills: List of skills (e.g., ["Python", "AWS", "Machine Learning"])
            current_title: Current job title
            desired_titles: List of desired job titles
            preferred_locations: List of preferred locations
            remote_preference: One of "remote", "hybrid", "onsite", "flexible"
            min_salary: Minimum acceptable salary
            years_experience: Years of professional experience
            
        Returns:
            Confirmation message
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
        
        # Generate embedding for the profile
        profile = await self.ranking_service.embed_user_profile(profile)
        set_current_profile(profile)
        
        # Persist to database
        await self.store.save_profile(profile)
        
        return f"Profile set for {name} with {len(skills)} skills. Ready to rank jobs."

    async def upload_resume(
        self,
        file_path: Optional[str] = None,
        file_data: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> str:
        """
        Parse a resume file and create a user profile from it.
        
        Supports PDF, DOCX, and TXT files. Extracts skills, experience, 
        and creates an embedding for job matching.
        
        Args:
            file_path: Path to the resume file (e.g., "/path/to/resume.pdf")
            file_data: Base64-encoded file content (alternative to file_path)
            file_type: File type when using file_data: "pdf", "docx", or "txt"
            
        Returns:
            Confirmation with extracted profile details
            
        Example:
            upload_resume(file_path="C:/Users/john/resume.pdf")
            upload_resume(file_data="<base64>", file_type="pdf")
        """
        if not file_path and not (file_data and file_type):
            return "Please provide either file_path or (file_data and file_type)."
        
        try:
            # Create parser (without LLM for now - uses regex extraction)
            parser = ResumeParser()
            
            # Parse and extract
            parsed = await parser.parse_and_extract(
                file_path=file_path,
                file_data=file_data,
                file_type=file_type,
                use_llm=False,  # Use regex extraction for speed
            )
            
            # Convert to UserProfile
            profile = parsed.to_user_profile()
            
            # Generate embedding
            profile = await self.ranking_service.embed_user_profile(profile)
            set_current_profile(profile)
            
            # Persist to database
            await self.store.save_profile(profile)
            
            # Build response
            skill_list = ", ".join(parsed.skills[:10])
            if len(parsed.skills) > 10:
                skill_list += f" (+{len(parsed.skills) - 10} more)"
            
            response = f"Resume parsed successfully!\n\n"
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
        """
        Rank all saved jobs against the user profile and return top matches.
        
        Args:
            top_k: Number of top-ranked jobs to return
            
        Returns:
            A formatted string with ranked jobs, scores, and justifications
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
        
        lines = [f"Top {len(ranked_jobs)} job matches for {profile.name}:"]
        for i, rj in enumerate(ranked_jobs, 1):
            job = rj.job
            score_pct = round(rj.score * 100, 1)
            lines.append(
                f"\n{i}. {job.title} at {job.company} - Score: {score_pct}%"
            )
            lines.append(f"   {rj.justification}")
            lines.append(f"   [ID: {job.id[:8]}]")
        
        return "\n".join(lines)

    async def get_profile(self) -> str:
        """
        Get the current user profile.
        
        Returns:
            Current user profile summary or message if not set
        """
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Use set_user_profile to add your resume and preferences."
        
        return f"""
User Profile:
- Name: {profile.name}
- Current Title: {profile.current_title or 'Not specified'}
- Years Experience: {profile.years_experience or 'Not specified'}
- Skills: {', '.join(profile.skills) if profile.skills else 'None'}
- Desired Titles: {', '.join(profile.desired_titles) if profile.desired_titles else 'Any'}
- Preferred Locations: {', '.join(profile.preferred_locations) if profile.preferred_locations else 'Any'}
- Remote Preference: {profile.remote_preference}
- Min Salary: {'$' + f'{profile.min_salary:,}' if profile.min_salary else 'Not specified'}
- Profile Embedding: {'Ready' if profile.embedding else 'Not generated'}
""".strip()

    async def send_job_notifications(
        self,
        top_k: int = 10,
        title: str = "New Job Matches",
    ) -> str:
        """
        Send notifications with the top ranked job matches to configured channels.
        
        Sends via email, Teams, Slack, or console depending on configuration.
        Includes job details, match scores, and quick action buttons.
        
        Args:
            top_k: Number of top matches to include (default 10)
            title: Notification title/subject
            
        Returns:
            Status message indicating which channels received the notification
        """
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Upload a resume first."
        
        # Get ranked jobs
        jobs = await self.store.list_all()
        if not jobs:
            return "No jobs to notify about. Search for jobs first."
        
        ranked_jobs = await self.ranking_service.rank_jobs(jobs, profile, top_k=top_k)
        if not ranked_jobs:
            return "No ranked matches found."
        
        # Send notifications
        results = await self.notification_service.send_job_matches(
            ranked_jobs=ranked_jobs,
            profile=profile,
            title=title,
        )
        
        # Format results
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
        """
        Provide feedback on a job match to improve future recommendations.
        
        Args:
            job_id: The job ID (can be partial, at least 8 characters)
            feedback: One of: "good_fit", "not_relevant", "tailor_resume", 
                     "draft_cover_letter", "already_applied", "company_blacklist"
            notes: Optional additional notes about why
            
        Returns:
            Confirmation message
            
        Example:
            provide_feedback(job_id="abc123", feedback="good_fit", notes="Great match!")
            provide_feedback(job_id="def456", feedback="not_relevant", notes="Wrong industry")
        """
        # Map feedback string to enum
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
        
        # Find the job
        all_jobs = await self.store.list_all(limit=1000)
        job = None
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                job = j
                break
        
        if not job:
            return f"Job with ID '{job_id}' not found."
        
        # Save feedback
        job_feedback = JobFeedback(
            job_id=job.id,
            feedback_type=feedback_type,
            notes=notes,
        )
        await self.store.save_feedback(job_feedback)
        
        # Take action based on feedback
        if feedback_type == FeedbackType.TAILOR_RESUME:
            return f"Feedback saved for '{job.title}'. Use prepare_application to get tailored resume suggestions."
        elif feedback_type == FeedbackType.DRAFT_COVER_LETTER:
            return f"Feedback saved for '{job.title}'. Use prepare_application to generate a cover letter."
        elif feedback_type == FeedbackType.GOOD_FIT:
            return f"Marked '{job.title}' at {job.company} as a good fit. Consider using prepare_application to start your application."
        elif feedback_type == FeedbackType.NOT_RELEVANT:
            return f"Marked '{job.title}' as not relevant. This feedback will help improve future matches."
        elif feedback_type == FeedbackType.COMPANY_BLACKLIST:
            return f"Added {job.company} to your blacklist. Future searches will deprioritize this company."
        else:
            return f"Feedback recorded for '{job.title}'."

    async def prepare_application(
        self,
        job_id: str,
    ) -> str:
        """
        Generate a tailored application package for a specific job.
        
        Creates:
        - Resume diff suggestions (targeted improvements, not full rewrite)
        - Concise cover letter draft
        - Intro email for recruiters
        - Recruiter list (if API configured)
        
        Args:
            job_id: The job ID (can be partial, at least 8 characters)
            
        Returns:
            Formatted application package for review
        """
        if not self.application_prep_service:
            return "Application prep service not configured. Check Azure OpenAI settings."
        
        profile = get_current_profile()
        if not profile:
            return "No user profile set. Upload a resume first."
        
        # Find the job
        all_jobs = await self.store.list_all(limit=1000)
        job = None
        for j in all_jobs:
            if j.id.startswith(job_id) or job_id in j.id:
                job = j
                break
        
        if not job:
            return f"Job with ID '{job_id}' not found."
        
        # Generate application package
        try:
            package = await self.application_prep_service.prepare_application(job, profile)
            
            # Save the package
            await self.store.save_application_package(package)
            
            # Format and return
            summary = await self.application_prep_service.format_package_summary(package)
            return f"Application package created (ID: {package.id[:8]}):\n\n{summary}"
            
        except Exception as e:
            logger.error(f"Failed to prepare application: {e}")
            return f"Error preparing application: {e}"

    async def get_application_package(
        self,
        package_id: str,
    ) -> str:
        """
        Retrieve a previously generated application package.
        
        Args:
            package_id: The package ID (can be partial)
            
        Returns:
            Formatted application package or error message
        """
        if not self.application_prep_service:
            return "Application prep service not configured."
        
        # Find the package
        packages = await self.store.list_application_packages(limit=100)
        package = None
        for p in packages:
            if p.id.startswith(package_id) or package_id in p.id:
                package = p
                break
        
        if not package:
            return f"Application package '{package_id}' not found. Use prepare_application to create one."
        
        return await self.application_prep_service.format_package_summary(package)

    @handler
    async def handle(self, messages: list[ChatMessage], ctx: WorkflowContext[Any, Any]) -> None:
        # Initialize on first request (load profile from database)
        if not getattr(self, '_initialized', False):
            await self.initialize()
            self._initialized = True
        
        # Run the agent and forward assistant messages as streaming updates
        response = await self.agent.run(messages)
        for message in response.messages:
            if message.role == Role.ASSISTANT and message.contents:
                await ctx.add_event(
                    AgentRunUpdateEvent(
                        self.id,
                        data=AgentRunResponseUpdate(
                            contents=[TextContent(text=message.contents[-1].text)],
                            role=Role.ASSISTANT,
                            response_id=str(uuid4()),
                        ),
                    )
                )
        # Yield a simple string output for the HTTP response body
        await ctx.yield_output(response.text)


def build_agent(
    client: AzureOpenAIChatClient,
    store: Optional[JobStore] = None,
    provider: Optional[JobIngestionProvider] = None,
    ranking_service: Optional[RankingService] = None,
    notification_service: Optional[NotificationService] = None,
    application_prep_service: Optional[ApplicationPrepService] = None,
):
    """Build the job agent workflow.
    
    Args:
        client: Azure OpenAI client for the chat model
        store: Job storage backend (defaults to in-memory)
        provider: Job ingestion provider (defaults to mock provider)
        ranking_service: Ranking service for job matching (defaults to mock embeddings)
        notification_service: Service for sending notifications
        application_prep_service: Service for preparing job applications
    """
    if store is None:
        store = InMemoryJobStore()  # Use create_agent() for async PostgreSQL initialization
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
                notification_service, application_prep_service
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
) -> Any:
    """Create the job agent workflow with async initialization.
    
    This is the preferred way to create the agent when using PostgreSQL,
    as it properly initializes the async database connection pool.
    
    Args:
        client: Azure OpenAI client for the chat model
        use_database: If True, uses PostgreSQL when configured; otherwise in-memory
    
    Returns:
        The configured agent workflow
    """
    from .config import AppConfig
    
    config = AppConfig.load()
    
    # Get store (async - will use PostgreSQL if configured)
    if use_database and config.database and config.database.is_configured:
        store = await get_store(config.database)
    else:
        store = InMemoryJobStore()
    
    # Get provider (sync)
    provider = get_provider()
    
    # Get ranking service (sync)
    ranking_service = get_ranking_service()
    
    return build_agent(client, store, provider, ranking_service)
