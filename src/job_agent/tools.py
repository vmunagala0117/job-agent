"""Tools that the coordinator agent can use to search and manage jobs."""

from typing import Optional

from .models import Job, JobSearchCriteria, RankedJob, UserProfile
from .providers import JobIngestionProvider
from .ranking import RankingService
from .store import JobStore


# Global user profile storage (in production, use persistent storage)
_current_profile: Optional[UserProfile] = None


def get_current_profile() -> Optional[UserProfile]:
    """Get the current user profile."""
    return _current_profile


def set_current_profile(profile: UserProfile) -> None:
    """Set the current user profile."""
    global _current_profile
    _current_profile = profile


class JobTools:
    """Collection of job-related tools for the agent."""
    
    def __init__(self, store: JobStore, provider: JobIngestionProvider):
        self.store = store
        self.provider = provider
    
    async def search_jobs(
        self,
        query: str,
        location: Optional[str] = None,
        remote_only: bool = False,
        min_salary: Optional[int] = None,
        max_results: int = 10,
    ) -> list[dict]:
        """
        Search for new job listings matching the criteria.
        
        Args:
            query: Job search query (e.g., "python developer", "data scientist")
            location: Optional location filter (e.g., "San Francisco", "Remote")
            remote_only: If True, only include remote positions
            min_salary: Minimum salary requirement
            max_results: Maximum number of results to return
            
        Returns:
            List of matching job listings with title, company, location, salary info
        """
        criteria = JobSearchCriteria(
            query=query,
            location=location,
            remote_only=remote_only,
            min_salary=min_salary,
            max_results=max_results,
        )
        
        # Fetch from provider and store
        jobs = await self.provider.fetch_jobs(criteria)
        await self.store.add_many(jobs)
        
        # Return simplified job info for the agent
        return [self._job_to_summary(job) for job in jobs]
    
    async def list_saved_jobs(self, limit: int = 20) -> list[dict]:
        """
        List all saved job listings.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of saved job summaries
        """
        jobs = await self.store.list_all(limit=limit)
        return [self._job_to_summary(job) for job in jobs]
    
    async def get_job_details(self, job_id: str) -> Optional[dict]:
        """
        Get detailed information about a specific job.
        
        Args:
            job_id: The unique identifier of the job
            
        Returns:
            Full job details or None if not found
        """
        job = await self.store.get(job_id)
        if job:
            return self._job_to_detail(job)
        return None
    
    async def mark_job_applied(self, job_id: str) -> str:
        """
        Mark a job as applied.
        
        Args:
            job_id: The unique identifier of the job
            
        Returns:
            Status message
        """
        from .models import JobStatus
        job = await self.store.update_status(job_id, JobStatus.APPLIED)
        if job:
            return f"Marked '{job.title}' at {job.company} as applied."
        return f"Job with ID {job_id} not found."
    
    async def mark_job_rejected(self, job_id: str) -> str:
        """
        Mark a job as not interested/rejected.
        
        Args:
            job_id: The unique identifier of the job
            
        Returns:
            Status message
        """
        from .models import JobStatus
        job = await self.store.update_status(job_id, JobStatus.REJECTED)
        if job:
            return f"Marked '{job.title}' at {job.company} as rejected."
        return f"Job with ID {job_id} not found."
    
    def _job_to_summary(self, job: Job) -> dict:
        """Convert job to a summary dict for agent responses."""
        salary_str = ""
        if job.salary_min and job.salary_max:
            salary_str = f"${job.salary_min:,} - ${job.salary_max:,}"
        elif job.salary_min:
            salary_str = f"${job.salary_min:,}+"
        
        return {
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "salary": salary_str,
            "job_type": job.job_type or "Not specified",
            "status": job.status.value,
        }
    
    def _job_to_detail(self, job: Job) -> dict:
        """Convert job to full detail dict."""
        summary = self._job_to_summary(job)
        summary.update({
            "description": job.description,
            "url": job.url,
            "experience_level": job.experience_level,
            "skills": job.skills,
            "source": job.source,
            "posted_at": job.posted_at.isoformat() if job.posted_at else None,
        })
        return summary


class RankingTools:
    """Collection of ranking and profile-related tools for the agent."""
    
    def __init__(self, store: JobStore, ranking_service: RankingService):
        self.store = store
        self.ranking_service = ranking_service
    
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
        
        return f"Profile set for {name} with {len(skills)} skills. Ready to rank jobs."
    
    async def rank_saved_jobs(
        self,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Rank all saved jobs against the user profile and return top matches.
        
        Args:
            top_k: Number of top-ranked jobs to return
            
        Returns:
            List of ranked jobs with scores and justifications
        """
        profile = get_current_profile()
        if not profile:
            return [{"error": "No user profile set. Use set_user_profile first."}]
        
        jobs = await self.store.list_all()
        if not jobs:
            return [{"error": "No saved jobs. Use search_jobs to find jobs first."}]
        
        ranked_jobs = await self.ranking_service.rank_jobs(jobs, profile, top_k=top_k)
        return [self._ranked_job_to_dict(rj) for rj in ranked_jobs]
    
    async def get_profile(self) -> dict:
        """
        Get the current user profile.
        
        Returns:
            Current user profile or error message
        """
        profile = get_current_profile()
        if not profile:
            return {"error": "No user profile set. Use set_user_profile first."}
        
        return {
            "name": profile.name,
            "current_title": profile.current_title,
            "skills": profile.skills,
            "years_experience": profile.years_experience,
            "desired_titles": profile.desired_titles,
            "preferred_locations": profile.preferred_locations,
            "remote_preference": profile.remote_preference,
            "min_salary": profile.min_salary,
            "has_embedding": profile.embedding is not None,
        }
    
    def _ranked_job_to_dict(self, ranked_job: RankedJob) -> dict:
        """Convert ranked job to dict for agent responses."""
        job = ranked_job.job
        salary_str = ""
        if job.salary_min and job.salary_max:
            salary_str = f"${job.salary_min:,} - ${job.salary_max:,}"
        elif job.salary_min:
            salary_str = f"${job.salary_min:,}+"
        
        return {
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "salary": salary_str,
            "score": round(ranked_job.score * 100, 1),
            "justification": ranked_job.justification,
            "score_breakdown": {
                "similarity": round(ranked_job.similarity_score * 100, 1),
                "skills": round(ranked_job.skill_match_score * 100, 1),
                "location": round(ranked_job.location_score * 100, 1),
                "salary": round(ranked_job.salary_score * 100, 1),
            },
        }
