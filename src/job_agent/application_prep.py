"""Application preparation service for generating tailored application materials."""

import logging
from dataclasses import dataclass
from typing import Optional

from openai import AsyncAzureOpenAI
from opentelemetry import trace

from .config import AzureOpenAIConfig
from .models import ApplicationPackage, Job, UserProfile

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("job_agent.application_prep")


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
        with _tracer.start_as_current_span("app_prep.prepare_application") as span:
            span.set_attribute("job.title", job.title)
            span.set_attribute("job.company", job.company)
            logger.info("[APP_PREP] Preparing application for %s at %s", job.title, job.company)
        
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
        with _tracer.start_as_current_span("app_prep.generate_resume_suggestions") as span:
            span.set_attribute("job.title", job.title)
            logger.info("[APP_PREP] Generating resume suggestions for %s", job.title)
            # Use resume parsing helpers to feed explicit, labeled bullets into the prompt
            from .resume_parser import create_parser

        parser = create_parser()
        # Limit raw resume context to a reasonable size
        raw_resume = (profile.resume_text or "")[:8000]

        # Extract experience entries and normalized skills for clear references
        try:
            entries = parser.extract_experience_entries(raw_resume)
            normalized_skills = parser.normalize_skill_terms(profile.skills or [])
        except Exception:
            logger.warning("[APP_PREP] Resume structured extraction failed, using raw text")
            entries = []
            normalized_skills = profile.skills or []

        # Build a compact, labeled resume-bullets section so the LLM can reference exact bullets
        resume_bullets = []
        for e in entries[:4]:
            label = (e.get("company") or e.get("title") or "Company").strip()
            for idx, b in enumerate(e.get("bullets", [])[:3], start=1):
                resume_bullets.append(f"Work Experience: {label} — bullet {idx}: {b}")

        resume_bullets_text = "\n".join(resume_bullets) or (raw_resume[:1200] + ("..." if len(raw_resume) > 1200 else ""))
        skills_text = ", ".join(normalized_skills[:40])

        prompt = f"""You are an expert Executive Recruiter and Resume Optimizer.

Analyze the Job Description and the candidate's resume. Provide 5-7 targeted, high-impact resume edits that increase ATS match and recruiter relevance.

JOB POSTING:
Title: {job.title}
Company: {job.company}
Description:
{job.description[:2000]}

Key skills (from JD): {', '.join(job.skills) if job.skills else 'Not specified'}

RESUME BULLETS (labeled excerpts from the candidate's resume):
{resume_bullets_text}

Profile skills (normalized): {skills_text}
Current Title: {profile.current_title}
Years Experience: {profile.years_experience or 'Not specified'}

Requirements:
- Provide 5-7 SPECIFIC suggestions.
- Do NOT provide a full rewrite — only precise edits.

Output FORMAT (strictly follow):
For each suggestion include these fields separated by a blank line:
[SECTION] (e.g., Professional Summary, Work Experience: Company X, KEY SKILLS)
Original: <the existing resume text or bullet you are modifying — quote briefly>
Suggested Change: <exact replacement text or short rephrase — ATS-friendly and uses JD terminology>
- Why: <1-2 lines tying the change to the JD, action verbs, and a measurable result if possible>

Also at the end, output a short "Gap Analysis" listing any MUST-HAVE JD requirements missing from the resume (one-line bullets).

Keep tone professional and use the JD's exact terminology where appropriate."""

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
            # Return raw text blocks (each block follows the specified format)
            suggestions = [s.strip() for s in suggestions_text.split("\n\n") if s.strip()]
            logger.info("[APP_PREP] Generated %d resume suggestions", len(suggestions))
            return suggestions
            
        except Exception as e:
            logger.error("[APP_PREP] Failed to generate resume suggestions: %s", e)
            span.record_exception(e)
            return [f"Error generating suggestions: {e}"]
    
    async def generate_cover_letter(
        self,
        job: Job,
        profile: UserProfile,
    ) -> str:
        """Generate a concise, tailored cover letter draft."""
        with _tracer.start_as_current_span("app_prep.generate_cover_letter") as span:
            span.set_attribute("job.title", job.title)
            logger.info("[APP_PREP] Generating cover letter for %s at %s", job.title, job.company)
            prompt = f"""Act as a high-end Career Strategist and Persuasive Copywriter.

    Write a concise, human-friendly cover letter that creates a clear "Value Bridge" between this Job Description and the candidate's experience.

    JOB:
    Title: {job.title}
    Company: {job.company}
    Key points from description:
    {job.description[:1500]}

    CANDIDATE:
    Name: {profile.name}
    Current Title: {profile.current_title}
    Top Skills: {', '.join(profile.skills[:10])}
    Years Experience: {profile.years_experience or 'Not specified'}
    Summary: {profile.summary or profile.resume_text[:2000]}

    Requirements (structure):
    1) Hook: Start with a specific, non-generic reason you're drawn to {job.company} (no "I am writing to apply").
    2) Requirement vs. Reality: For 2-3 key JD requirements, state "You’re looking for [Skill A] to solve [Problem B]. In my role at [Company], I used [Skill A] together with [unique approach] to achieve [specific result with metrics]."
    3) Plus-One Value: Name 1-2 capabilities from the resume that aren't in the JD and explain how they'd help the team.
    4) Tone: Professional, warm, conversational; avoid corporate buzzwords.
    5) Closing: Confident, low-pressure call to action.

    Keep under 250 words. Use concrete metrics where available. Produce the cover letter now."""

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
                
                logger.info("[APP_PREP] Cover letter generated")
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error("[APP_PREP] Failed to generate cover letter: %s", e)
                span.record_exception(e)
                return f"Error generating cover letter: {e}"
    
    async def generate_intro_email(
        self,
        job: Job,
        profile: UserProfile,
        recruiter_name: Optional[str] = None,
    ) -> str:
        """Generate an intro email to send to a recruiter or hiring manager."""
        with _tracer.start_as_current_span("app_prep.generate_intro_email") as span:
            span.set_attribute("job.title", job.title)
            logger.info("[APP_PREP] Generating intro email for %s", job.title)
            recipient = recruiter_name or "Hiring Manager"

        prompt = f"""You are a Networking Expert and Relationship Builder.

    Produce two outputs: (A) a concise outreach email (subject + body) suitable for email, and (B) a brief LinkedIn message (1-2 short paragraphs) to a recruiter about this specific job. Both should be low-pressure, show homework, and include one "extra" value.

    JOB:
    Title: {job.title}
    Company: {job.company}

    CANDIDATE:
    Name: {profile.name}
    Current Title: {profile.current_title}
    Top Skills: {', '.join(profile.skills[:5])}

    Constraints:
    - Email: subject line that includes the job title; body under 100 words; ask for a conversation.
    - LinkedIn message: ~50-80 words, mention a specific reason you're interested and one extra benefit you bring beyond the JD.

    To: {recipient}

    Write outputs labeled clearly as "EMAIL SUBJ:" and "EMAIL BODY:" then "LINKEDIN:" so they can be separated programmatically."""

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
            
            logger.info("[APP_PREP] Intro email generated")
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error("[APP_PREP] Failed to generate intro email: %s", e)
            span.record_exception(e)
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
            f"Job: {package.job.title} at {package.job.company}"
            + (f" | 🔗 {package.job.url}" if package.job.url else ""),
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
