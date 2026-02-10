"""Application preparation service for generating tailored application materials."""

import logging
from dataclasses import dataclass
from typing import Optional

from openai import AsyncAzureOpenAI

from .config import AzureOpenAIConfig
from .models import ApplicationPackage, Job, UserProfile

logger = logging.getLogger(__name__)


@dataclass
class ApplicationPrepConfig:
    """Configuration for the application prep service."""
    
    # LLM settings
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Recruiter search (Proxycurl or similar)
    proxycurl_api_key: Optional[str] = None
    enable_recruiter_search: bool = False


class ApplicationPrepService:
    """Service for generating tailored job application materials.
    
    Generates:
    - Resume diff suggestions (not full rewrite)
    - Concise cover letter draft
    - Recruiter list (if API configured)
    - Intro email template
    """
    
    def __init__(
        self,
        openai_config: AzureOpenAIConfig,
        prep_config: Optional[ApplicationPrepConfig] = None,
    ):
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        
        if openai_config.api_key:
            self.client = AsyncAzureOpenAI(
                api_key=openai_config.api_key,
                api_version=openai_config.api_version,
                azure_endpoint=openai_config.endpoint,
            )
        else:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=openai_config.api_version,
                azure_endpoint=openai_config.endpoint,
            )
        
        self.model = openai_config.deployment_name
        self.prep_config = prep_config or ApplicationPrepConfig()
    
    async def prepare_application(
        self,
        job: Job,
        profile: UserProfile,
    ) -> ApplicationPackage:
        """Generate a complete application package for a job.
        
        Args:
            job: The job to apply for
            profile: The user's profile with resume
            
        Returns:
            ApplicationPackage with all generated materials
        """
        logger.info(f"Preparing application for {job.title} at {job.company}")
        
        # Generate all materials
        resume_suggestions = await self.generate_resume_suggestions(job, profile)
        cover_letter = await self.generate_cover_letter(job, profile)
        intro_email = await self.generate_intro_email(job, profile)
        
        # Search for recruiters if enabled
        recruiters = []
        if self.prep_config.enable_recruiter_search:
            recruiters = await self.find_recruiters(job.company)
        
        return ApplicationPackage(
            job=job,
            profile=profile,
            resume_suggestions=resume_suggestions,
            cover_letter=cover_letter,
            intro_email=intro_email,
            recruiters=recruiters,
            status="draft",
        )
    
    async def generate_resume_suggestions(
        self,
        job: Job,
        profile: UserProfile,
    ) -> list[str]:
        """Generate specific resume diff suggestions (not a full rewrite).
        
        Returns a list of specific changes to make to the resume.
        """
        prompt = f"""Analyze this job posting and resume, then provide SPECIFIC suggestions for tailoring the resume.

JOB POSTING:
Title: {job.title}
Company: {job.company}
Description:
{job.description[:2000]}

Required Skills: {', '.join(job.skills) if job.skills else 'Not specified'}

CURRENT RESUME:
{profile.resume_text[:3000]}

Skills: {', '.join(profile.skills)}
Current Title: {profile.current_title}
Years Experience: {profile.years_experience or 'Not specified'}

Provide 5-7 SPECIFIC, ACTIONABLE suggestions in diff format. Each suggestion should:
1. Reference a specific section or bullet point in the resume
2. Explain what to change and why
3. Be directly relevant to this job posting

Format each suggestion as:
[SECTION] Specific change to make
- Why: Brief justification tied to job requirements

Do NOT suggest a complete rewrite. Focus on targeted improvements."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert resume coach who provides specific, actionable feedback."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.prep_config.temperature,
                max_completion_tokens=self.prep_config.max_tokens,
            )
            
            suggestions_text = response.choices[0].message.content
            # Parse into individual suggestions
            suggestions = [s.strip() for s in suggestions_text.split("\n\n") if s.strip()]
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate resume suggestions: {e}")
            return [f"Error generating suggestions: {e}"]
    
    async def generate_cover_letter(
        self,
        job: Job,
        profile: UserProfile,
    ) -> str:
        """Generate a concise, tailored cover letter draft."""
        prompt = f"""Write a CONCISE cover letter for this job application.

JOB:
Title: {job.title}
Company: {job.company}
Description (key points):
{job.description[:1500]}

CANDIDATE:
Name: {profile.name}
Current Title: {profile.current_title}
Key Skills: {', '.join(profile.skills[:10])}
Years Experience: {profile.years_experience or 'Not specified'}
Summary: {profile.summary or profile.resume_text[:500]}

Requirements:
1. Keep it under 250 words
2. Open with a hook that shows genuine interest in {job.company}
3. Highlight 2-3 specific qualifications that match the job
4. Include a concrete achievement with metrics if possible
5. Close with a clear call to action
6. Be professional but personable - avoid generic phrases

Write the cover letter now:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional writer who creates compelling, concise cover letters."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.prep_config.temperature,
                max_completion_tokens=800,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate cover letter: {e}")
            return f"Error generating cover letter: {e}"
    
    async def generate_intro_email(
        self,
        job: Job,
        profile: UserProfile,
        recruiter_name: Optional[str] = None,
    ) -> str:
        """Generate an intro email to send to a recruiter or hiring manager."""
        recipient = recruiter_name or "Hiring Manager"
        
        prompt = f"""Write a brief, professional email to a recruiter about this job opportunity.

JOB:
Title: {job.title}
Company: {job.company}

CANDIDATE:
Name: {profile.name}
Current Title: {profile.current_title}
Key Skills: {', '.join(profile.skills[:5])}

Requirements:
1. Subject line that stands out (include job title)
2. Keep the email under 100 words
3. Get straight to the point
4. Mention 1-2 relevant qualifications
5. Request a conversation (not just "let me know")
6. Professional but warm tone

To: {recipient}

Write the email now (include subject line):"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a networking expert who writes effective outreach emails."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.prep_config.temperature,
                max_completion_tokens=400,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate intro email: {e}")
            return f"Error generating intro email: {e}"
    
    async def find_recruiters(
        self,
        company: str,
        limit: int = 5,
    ) -> list[dict]:
        """Find recruiters at a company using Proxycurl or similar API.
        
        Returns list of dicts with: name, title, linkedin_url, email (if available)
        """
        if not self.prep_config.proxycurl_api_key:
            logger.info("Recruiter search disabled - no API key configured")
            return []
        
        try:
            import httpx
            
            # Proxycurl Company Search API (simplified example)
            # In production, you'd use the full Proxycurl API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://nubela.co/proxycurl/api/linkedin/company/employees/",
                    params={
                        "url": f"https://www.linkedin.com/company/{company.lower().replace(' ', '-')}",
                        "role_search": "recruiter|talent|hiring",
                        "page_size": str(limit),
                    },
                    headers={
                        "Authorization": f"Bearer {self.prep_config.proxycurl_api_key}",
                    },
                    timeout=30.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return [
                        {
                            "name": emp.get("full_name", "Unknown"),
                            "title": emp.get("title", "Recruiter"),
                            "linkedin_url": emp.get("profile_url", ""),
                            "email": emp.get("email"),  # May require additional lookup
                        }
                        for emp in data.get("employees", [])
                    ]
                else:
                    logger.warning(f"Proxycurl API returned {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to search for recruiters: {e}")
            return []
    
    async def format_package_summary(self, package: ApplicationPackage) -> str:
        """Format an application package as a readable summary."""
        lines = [
            f"📋 APPLICATION PACKAGE",
            f"{'=' * 50}",
            f"Job: {package.job.title} at {package.job.company}",
            f"Status: {package.status}",
            f"Created: {package.created_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "📝 RESUME SUGGESTIONS",
            "-" * 30,
        ]
        
        for i, suggestion in enumerate(package.resume_suggestions, 1):
            lines.append(f"{i}. {suggestion[:200]}...")
        
        lines.extend([
            "",
            "✉️ COVER LETTER DRAFT",
            "-" * 30,
            package.cover_letter[:500] + ("..." if len(package.cover_letter) > 500 else ""),
            "",
            "📧 INTRO EMAIL",
            "-" * 30,
            package.intro_email,
        ])
        
        if package.recruiters:
            lines.extend([
                "",
                "👥 RECRUITERS FOUND",
                "-" * 30,
            ])
            for r in package.recruiters:
                lines.append(f"• {r['name']} - {r['title']}")
                if r.get('linkedin_url'):
                    lines.append(f"  LinkedIn: {r['linkedin_url']}")
        
        lines.extend([
            "",
            "=" * 50,
            "Use 'approve_application' to finalize, or 'edit_package' to modify.",
        ])
        
        return "\n".join(lines)


def get_application_prep_service(
    openai_config: Optional[AzureOpenAIConfig] = None,
    prep_config: Optional[ApplicationPrepConfig] = None,
) -> Optional[ApplicationPrepService]:
    """Factory function to get application prep service."""
    import os
    from .config import AppConfig
    
    if openai_config is None:
        config = AppConfig.load()
        openai_config = config.azure_openai
    
    if not openai_config or not openai_config.endpoint:
        logger.warning("Azure OpenAI not configured - application prep disabled")
        return None
    
    # Check for Proxycurl API key
    if prep_config is None:
        proxycurl_key = os.getenv("PROXYCURL_API_KEY")
        prep_config = ApplicationPrepConfig(
            proxycurl_api_key=proxycurl_key,
            enable_recruiter_search=bool(proxycurl_key),
        )
    
    return ApplicationPrepService(openai_config, prep_config)
