"""Quick test script for notifications and application prep features."""
import asyncio
import os
from datetime import datetime
from uuid import uuid4

# Load env
from dotenv import load_dotenv
load_dotenv()

from src.job_agent.config import AzureOpenAIConfig
from src.job_agent.models import (
    Job, JobStatus, UserProfile, RankedJob,
    NotificationChannel, NotificationConfig,
    FeedbackType, JobFeedback,
)
from src.job_agent.store import InMemoryJobStore
from src.job_agent.notifications import NotificationService
from src.job_agent.application_prep import ApplicationPrepService, ApplicationPrepConfig


async def main():
    print("=" * 60)
    print("Testing Notifications & Application Prep")
    print("=" * 60)
    
    # Create sample data
    profile = UserProfile(
        id=str(uuid4()),
        name="John Developer",
        email="john@example.com",
        skills=["Python", "FastAPI", "PostgreSQL", "Azure"],
        years_experience=5,
        preferred_locations=["Seattle, WA"],
        desired_titles=["Backend Engineer", "Software Engineer"],
        resume_text="Experienced Python developer with 5 years building scalable APIs...",
    )
    
    job = Job(
        id=str(uuid4()),
        title="Senior Python Developer",
        company="TechCorp",
        location="Seattle, WA",
        description="Looking for an experienced Python developer to build APIs with FastAPI...",
        url="https://example.com/jobs/123",
        source="test",
        posted_at=datetime.now(),
        salary_min=150000,
        salary_max=180000,
        status=JobStatus.NEW,
    )
    
    ranked_job = RankedJob(
        job=job,
        score=0.92,
        justification="Excellent fit! Strong skill alignment with Python and FastAPI.",
        similarity_score=0.95,
        skill_match_score=0.90,
        location_score=1.0,
        salary_score=0.85,
    )
    
    store = InMemoryJobStore()
    
    # Test 1: Notifications
    print("\n1. Testing Notification Service...")
    config = NotificationConfig(
        channel=NotificationChannel.CONSOLE,
    )
    notification_service = NotificationService(configs=[config])
    
    result = await notification_service.send_job_matches(
        ranked_jobs=[ranked_job],
        profile=profile,
    )
    print(f"   Notification sent: {result}")
    
    # Test 2: Feedback
    print("\n2. Testing Feedback Storage...")
    feedback = JobFeedback(
        job_id=job.id,
        feedback_type=FeedbackType.GOOD_FIT,
        notes="Looks like a great opportunity!",
    )
    await store.save_feedback(feedback)
    
    feedback_list = await store.get_feedback(job.id)
    print(f"   Feedback stored: {feedback_list[0].feedback_type.value if feedback_list else 'None'}")
    
    # Test 3: Application Prep
    print("\n3. Testing Application Prep Service...")
    try:
        openai_config = AzureOpenAIConfig.from_env()
        prep_config = ApplicationPrepConfig(
            max_tokens=800,
            temperature=0.7,
        )
        prep_service = ApplicationPrepService(openai_config, prep_config)
        
        package = await prep_service.prepare_application(job, profile)
        
        print(f"   Package ID: {package.id}")
        print(f"   Status: {package.status}")
        print(f"\n   Resume Suggestions:\n{package.resume_suggestions[:300]}...")
        print(f"\n   Cover Letter:\n{package.cover_letter[:300]}...")
        print(f"\n   Intro Email:\n{package.intro_email[:300]}...")
        
        # Store the package
        await store.save_application_package(package)
        retrieved_pkg = await store.get_application_package(package.id)
        print(f"\n   Package stored and retrieved: {retrieved_pkg.id if retrieved_pkg else 'None'}")
        
    except Exception as e:
        print(f"   Error in application prep: {e}")
    
    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
