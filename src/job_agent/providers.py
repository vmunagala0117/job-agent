"""Job ingestion providers for fetching job listings from external sources."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import logging
import random
from typing import Optional

import aiohttp

from .models import Job, JobSearchCriteria

logger = logging.getLogger(__name__)


class JobIngestionProvider(ABC):
    """Abstract base class for job ingestion providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        ...
    
    @abstractmethod
    async def fetch_jobs(self, criteria: JobSearchCriteria) -> list[Job]:
        """Fetch jobs matching the given criteria from the external source."""
        ...


class MockJobProvider(JobIngestionProvider):
    """Mock provider for development and testing. Generates sample job listings."""
    
    @property
    def name(self) -> str:
        return "mock"
    
    async def fetch_jobs(self, criteria: JobSearchCriteria) -> list[Job]:
        """Generate mock job listings based on the search criteria."""
        
        # Sample data for generating realistic-looking jobs
        companies = [
            "TechCorp", "DataDriven Inc", "CloudFirst", "AI Solutions",
            "DevOps Masters", "StartupXYZ", "Enterprise Systems", "Digital Dynamics"
        ]
        
        locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
            "Remote", "Boston, MA", "Denver, CO", "Chicago, IL"
        ]
        
        job_types = ["full-time", "contract", "full-time remote", "hybrid"]
        experience_levels = ["entry", "mid", "senior", "lead"]
        
        skills_pool = [
            "Python", "JavaScript", "TypeScript", "React", "Node.js",
            "AWS", "Azure", "Docker", "Kubernetes", "SQL", "PostgreSQL",
            "Machine Learning", "Data Analysis", "REST APIs", "GraphQL"
        ]
        
        jobs = []
        num_jobs = min(criteria.max_results, random.randint(5, 15))
        
        for i in range(num_jobs):
            # Generate a job title based on the query
            base_title = criteria.query.title() if criteria.query else "Software Engineer"
            seniority = random.choice(["", "Senior ", "Lead ", "Staff "])
            title = f"{seniority}{base_title}"
            
            # Pick random attributes
            company = random.choice(companies)
            location = criteria.location if criteria.location else random.choice(locations)
            job_type = random.choice(job_types)
            exp_level = random.choice(experience_levels)
            
            # Generate salary range
            base_salary = random.randint(80, 180) * 1000
            salary_min = base_salary
            salary_max = base_salary + random.randint(20, 50) * 1000
            
            # Pick random skills
            num_skills = random.randint(3, 7)
            skills = random.sample(skills_pool, num_skills)
            
            # Generate a description
            description = (
                f"We are looking for a {title} to join our team at {company}. "
                f"This is a {job_type} position based in {location}. "
                f"Required skills: {', '.join(skills[:3])}. "
                f"Nice to have: {', '.join(skills[3:])}. "
                f"Competitive salary and benefits package."
            )
            
            # Random posted date within the last 30 days
            days_ago = random.randint(0, 30)
            posted_at = datetime.utcnow() - timedelta(days=days_ago)
            
            job = Job(
                title=title,
                company=company,
                location=location,
                description=description,
                url=f"https://example.com/jobs/{company.lower().replace(' ', '-')}-{i}",
                salary_min=salary_min,
                salary_max=salary_max,
                job_type=job_type,
                experience_level=exp_level,
                skills=skills,
                source=self.name,
                posted_at=posted_at,
            )
            jobs.append(job)
        
        return jobs


class SerpAPIJobProvider(JobIngestionProvider):
    """Provider that fetches jobs from SerpAPI Google Jobs.
    
    Uses the Google Jobs API which aggregates listings from multiple sources
    including LinkedIn, Indeed, Glassdoor, and company career pages.
    
    API docs: https://serpapi.com/google-jobs-api
    """
    
    GOOGLE_JOBS_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: str):
        self._api_key = api_key
    
    @property
    def name(self) -> str:
        return "serpapi_google_jobs"
    
    async def fetch_jobs(self, criteria: JobSearchCriteria) -> list[Job]:
        """Fetch jobs from Google Jobs via SerpAPI."""
        from .models import DatePosted
        
        # Build the search query
        query = criteria.query
        
        # Handle remote preference - add "remote" to query text (this works better than chips)
        is_remote_search = criteria.remote_only or (
            criteria.location and criteria.location.lower() in ("remote", "remote only", "work from home")
        )
        
        if is_remote_search:
            # Add "remote" to query text - this mimics what users type on Google
            if "remote" not in query.lower():
                query = f"{query} remote"
        
        # Add date filter as query text (more reliable than chips)
        date_suffix = criteria.date_posted.query_suffix if criteria.date_posted else ""
        if date_suffix:
            query = f"{query} {date_suffix}"
        
        # Clean location - don't include "remote" as a location
        clean_location = None
        if criteria.location:
            loc_lower = criteria.location.lower()
            if loc_lower not in ("remote", "remote only", "work from home"):
                clean_location = criteria.location
        
        params = {
            "engine": "google_jobs",
            "q": query,
            "api_key": self._api_key,
        }
        
        # Add location parameter if specified (but not "Remote")
        if clean_location:
            params["location"] = clean_location
        elif is_remote_search:
            # For remote searches, use United States as base location
            params["location"] = "United States"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.GOOGLE_JOBS_URL, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"SerpAPI request failed: {response.status} - {error_text}")
                        print(f"SerpAPI request failed: {response.status} - {error_text}")
                        return []
                    
                    data = await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"SerpAPI connection error: {e}")
            return []
        
        # Parse the response
        jobs_data = data.get("jobs_results", [])
        
        if not jobs_data:
            logger.info(f"No jobs found for query: {criteria.query}")
            return []
        
        jobs = []
        for job_data in jobs_data[:criteria.max_results * 2]:  # Fetch extra in case we filter
            job = self._parse_job(job_data)
            if job:
                # Apply client-side salary filter if specified
                if criteria.min_salary:
                    # Only filter if job has salary data
                    if job.salary_max is not None and job.salary_max < criteria.min_salary:
                        continue  # Skip jobs below salary requirement
                    # Jobs without salary data are included (don't filter them out)
                
                jobs.append(job)
                if len(jobs) >= criteria.max_results:
                    break
        
        logger.info(f"Fetched {len(jobs)} jobs from SerpAPI for: {criteria.query}")
        return jobs
    
    def _parse_job(self, data: dict) -> Optional[Job]:
        """Parse a single job from SerpAPI response."""
        try:
            # Extract basic info
            title = data.get("title", "Unknown Title")
            company = data.get("company_name", "Unknown Company")
            location = data.get("location", "Unknown Location")
            description = data.get("description", "")
            
            # Get the job URL (apply link or share link)
            url = None
            apply_options = data.get("apply_options", [])
            if apply_options:
                url = apply_options[0].get("link")
            if not url:
                url = data.get("share_link") or data.get("job_id")
            
            # Parse salary if available
            salary_min, salary_max = self._parse_salary(data)
            
            # Detect job type from extensions
            job_type = None
            extensions = data.get("detected_extensions", {})
            if extensions.get("work_from_home"):
                job_type = "remote"
            schedule_type = extensions.get("schedule_type")
            if schedule_type:
                job_type = schedule_type.lower()
            
            # Extract posted date
            posted_at = self._parse_posted_date(data.get("detected_extensions", {}))
            
            # Extract highlights as skills (if available)
            skills = []
            highlights = data.get("job_highlights", [])
            for highlight in highlights:
                if highlight.get("title") == "Qualifications":
                    # Extract skill-like items from qualifications
                    items = highlight.get("items", [])
                    for item in items[:5]:  # Limit to first 5
                        # Simple extraction - take first few words
                        words = item.split()[:4]
                        if words:
                            skills.append(" ".join(words))
            
            return Job(
                title=title,
                company=company,
                location=location,
                description=description,
                url=url,
                salary_min=salary_min,
                salary_max=salary_max,
                job_type=job_type,
                skills=skills,
                source=self.name,
                posted_at=posted_at,
            )
        except Exception as e:
            logger.warning(f"Failed to parse job: {e}")
            return None
    
    def _parse_salary(self, data: dict) -> tuple[Optional[int], Optional[int]]:
        """Extract salary range from job data."""
        extensions = data.get("detected_extensions", {})
        salary_str = extensions.get("salary")
        
        if not salary_str:
            return None, None
        
        # Try to parse salary like "$120K - $180K" or "$150,000 - $200,000"
        import re
        
        # Find all dollar amounts
        amounts = re.findall(r'\$?([\d,]+)(?:K|k)?', salary_str)
        
        if len(amounts) >= 2:
            try:
                min_sal = int(amounts[0].replace(",", ""))
                max_sal = int(amounts[1].replace(",", ""))
                
                # If values are in thousands (K format)
                if "K" in salary_str or "k" in salary_str:
                    min_sal *= 1000
                    max_sal *= 1000
                elif min_sal < 1000:  # Likely hourly, skip
                    return None, None
                
                return min_sal, max_sal
            except ValueError:
                pass
        elif len(amounts) == 1:
            try:
                sal = int(amounts[0].replace(",", ""))
                if "K" in salary_str or "k" in salary_str:
                    sal *= 1000
                if sal >= 20000:  # Reasonable annual salary
                    return sal, sal
            except ValueError:
                pass
        
        return None, None
    
    def _parse_posted_date(self, extensions: dict) -> Optional[datetime]:
        """Parse the posted date from extensions."""
        posted_at = extensions.get("posted_at")
        
        if not posted_at:
            return None
        
        # Parse relative dates like "2 days ago", "1 week ago", "Just posted"
        posted_at_lower = posted_at.lower()
        now = datetime.utcnow()
        
        if "just" in posted_at_lower or "today" in posted_at_lower:
            return now
        
        import re
        match = re.search(r'(\d+)\s*(hour|day|week|month)', posted_at_lower)
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            
            if unit == "hour":
                return now - timedelta(hours=num)
            elif unit == "day":
                return now - timedelta(days=num)
            elif unit == "week":
                return now - timedelta(weeks=num)
            elif unit == "month":
                return now - timedelta(days=num * 30)
        
        return None


def get_job_provider(api_key: Optional[str] = None) -> JobIngestionProvider:
    """Factory function to get the appropriate job provider.
    
    Returns SerpAPIJobProvider if API key is provided, otherwise MockJobProvider.
    """
    if api_key:
        logger.info("Using SerpAPI Google Jobs provider")
        return SerpAPIJobProvider(api_key)
    else:
        logger.info("No SerpAPI key configured, using mock provider")
        return MockJobProvider()


def get_provider() -> JobIngestionProvider:
    """Get the job provider using configuration from environment.
    
    Uses SerpAPI if SERPAPI_API_KEY is configured, otherwise returns MockJobProvider.
    """
    from .config import SerpAPIConfig
    
    config = SerpAPIConfig.from_env()
    return get_job_provider(config.api_key)
