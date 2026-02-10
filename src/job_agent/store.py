"""Job storage abstraction and implementations."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .models import ApplicationPackage, FeedbackType, Job, JobFeedback, JobSearchCriteria, JobStatus, UserProfile

if TYPE_CHECKING:
    from .config import DatabaseConfig

logger = logging.getLogger(__name__)


class JobStore(ABC):
    """Abstract base class for job storage."""
    
    @abstractmethod
    async def add(self, job: Job) -> Job:
        """Add a job to the store. Returns the job with its ID."""
        ...
    
    @abstractmethod
    async def add_many(self, jobs: list[Job]) -> list[Job]:
        """Add multiple jobs to the store. Returns jobs with their IDs."""
        ...
    
    @abstractmethod
    async def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID. Returns None if not found."""
        ...
    
    @abstractmethod
    async def search(self, criteria: JobSearchCriteria) -> list[Job]:
        """Search jobs matching the given criteria."""
        ...
    
    @abstractmethod
    async def update_status(self, job_id: str, status: JobStatus) -> Optional[Job]:
        """Update a job's status. Returns the updated job or None if not found."""
        ...
    
    @abstractmethod
    async def list_all(self, limit: int = 100) -> list[Job]:
        """List all jobs, up to the given limit."""
        ...
    
    @abstractmethod
    async def delete(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if deleted, False if not found."""
        ...
    
    # Profile methods (optional - default to no-op for stores that don't support profiles)
    async def save_profile(self, profile: UserProfile) -> UserProfile:
        """Save a user profile. Returns the saved profile."""
        return profile
    
    async def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID. Returns None if not found."""
        return None
    
    async def get_default_profile(self) -> Optional[UserProfile]:
        """Get the default/most recent user profile."""
        return None
    
    async def update_job_embeddings(self, job_embeddings: list[tuple[str, list[float]]]) -> int:
        """Update embeddings for multiple jobs. Returns count of updated jobs."""
        return 0
    
    async def get_jobs_without_embeddings(self, limit: int = 100) -> list[Job]:
        """Get jobs that don't have embeddings yet."""
        return []
    
    # Feedback methods
    async def save_feedback(self, feedback: JobFeedback) -> JobFeedback:
        """Save user feedback on a job. Returns the saved feedback."""
        return feedback
    
    async def get_feedback(self, job_id: str) -> list[JobFeedback]:
        """Get all feedback for a job."""
        return []
    
    async def list_feedback(self, feedback_type: Optional[FeedbackType] = None, limit: int = 100) -> list[JobFeedback]:
        """List feedback, optionally filtered by type."""
        return []
    
    # Application package methods
    async def save_application_package(self, package: ApplicationPackage) -> ApplicationPackage:
        """Save an application package. Returns the saved package."""
        return package
    
    async def get_application_package(self, package_id: str) -> Optional[ApplicationPackage]:
        """Get an application package by ID."""
        return None
    
    async def list_application_packages(self, status: Optional[str] = None, limit: int = 100) -> list[ApplicationPackage]:
        """List application packages, optionally filtered by status."""
        return []


class InMemoryJobStore(JobStore):
    """Simple in-memory job store for development and testing."""
    
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._profiles: dict[str, UserProfile] = {}
        self._feedback: dict[str, list[JobFeedback]] = {}  # job_id -> feedback list
        self._packages: dict[str, ApplicationPackage] = {}
        self._default_profile_id: Optional[str] = None
    
    async def add(self, job: Job) -> Job:
        self._jobs[job.id] = job
        return job
    
    async def add_many(self, jobs: list[Job]) -> list[Job]:
        for job in jobs:
            self._jobs[job.id] = job
        return jobs
    
    async def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)
    
    async def search(self, criteria: JobSearchCriteria) -> list[Job]:
        """Basic search implementation with simple text matching."""
        results = []
        query_lower = criteria.query.lower()
        
        for job in self._jobs.values():
            # Text match on title, company, description
            text_match = (
                query_lower in job.title.lower() or
                query_lower in job.company.lower() or
                query_lower in job.description.lower()
            )
            
            if not text_match:
                continue
            
            # Location filter
            if criteria.location and criteria.location.lower() not in job.location.lower():
                continue
            
            # Remote filter
            if criteria.remote_only and job.job_type and "remote" not in job.job_type.lower():
                continue
            
            # Salary filter
            if criteria.min_salary and job.salary_max and job.salary_max < criteria.min_salary:
                continue
            
            results.append(job)
            
            if len(results) >= criteria.max_results:
                break
        
        return results
    
    async def update_status(self, job_id: str, status: JobStatus) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if job:
            job.status = status
        return job
    
    async def list_all(self, limit: int = 100) -> list[Job]:
        return list(self._jobs.values())[:limit]
    
    async def delete(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False
    
    def clear(self):
        """Clear all jobs (useful for testing)."""
        self._jobs.clear()
    
    async def save_profile(self, profile: UserProfile) -> UserProfile:
        """Save a user profile."""
        self._profiles[profile.id] = profile
        self._default_profile_id = profile.id
        return profile
    
    async def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID."""
        return self._profiles.get(profile_id)
    
    async def get_default_profile(self) -> Optional[UserProfile]:
        """Get the default/most recent user profile."""
        if self._default_profile_id:
            return self._profiles.get(self._default_profile_id)
        return None
    
    async def update_job_embeddings(self, job_embeddings: list[tuple[str, list[float]]]) -> int:
        """Update embeddings for multiple jobs."""
        count = 0
        for job_id, embedding in job_embeddings:
            if job_id in self._jobs:
                self._jobs[job_id].embedding = embedding
                count += 1
        return count
    
    async def get_jobs_without_embeddings(self, limit: int = 100) -> list[Job]:
        """Get jobs that don't have embeddings yet."""
        return [job for job in list(self._jobs.values())[:limit] if job.embedding is None]
    
    async def save_feedback(self, feedback: JobFeedback) -> JobFeedback:
        """Save user feedback on a job."""
        if feedback.job_id not in self._feedback:
            self._feedback[feedback.job_id] = []
        self._feedback[feedback.job_id].append(feedback)
        return feedback
    
    async def get_feedback(self, job_id: str) -> list[JobFeedback]:
        """Get all feedback for a job."""
        return self._feedback.get(job_id, [])
    
    async def list_feedback(self, feedback_type: Optional[FeedbackType] = None, limit: int = 100) -> list[JobFeedback]:
        """List feedback, optionally filtered by type."""
        all_feedback = []
        for fb_list in self._feedback.values():
            all_feedback.extend(fb_list)
        if feedback_type:
            all_feedback = [fb for fb in all_feedback if fb.feedback_type == feedback_type]
        return sorted(all_feedback, key=lambda x: x.created_at, reverse=True)[:limit]
    
    async def save_application_package(self, package: ApplicationPackage) -> ApplicationPackage:
        """Save an application package."""
        self._packages[package.id] = package
        return package
    
    async def get_application_package(self, package_id: str) -> Optional[ApplicationPackage]:
        """Get an application package by ID."""
        return self._packages.get(package_id)
    
    async def list_application_packages(self, status: Optional[str] = None, limit: int = 100) -> list[ApplicationPackage]:
        """List application packages, optionally filtered by status."""
        packages = list(self._packages.values())
        if status:
            packages = [p for p in packages if p.status == status]
        return sorted(packages, key=lambda x: x.created_at, reverse=True)[:limit]


class PostgresJobStore(JobStore):
    """PostgreSQL-based job store with pgvector support for embeddings."""
    
    # SQL for creating tables
    INIT_SQL = """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Jobs table
    CREATE TABLE IF NOT EXISTS jobs (
        id VARCHAR(36) PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        company VARCHAR(500) NOT NULL,
        location VARCHAR(500) NOT NULL,
        description TEXT NOT NULL,
        url TEXT,
        salary_min INTEGER,
        salary_max INTEGER,
        job_type VARCHAR(50),
        experience_level VARCHAR(50),
        skills JSONB DEFAULT '[]',
        source VARCHAR(100) DEFAULT 'unknown',
        status VARCHAR(50) DEFAULT 'new',
        score FLOAT,
        posted_at TIMESTAMP,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        embedding vector(1536),  -- For text-embedding-3-small
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- User profiles table
    CREATE TABLE IF NOT EXISTS user_profiles (
        id VARCHAR(36) PRIMARY KEY,
        name VARCHAR(255),
        email VARCHAR(255),
        resume_text TEXT,
        summary TEXT,
        skills JSONB DEFAULT '[]',
        years_experience INTEGER,
        current_title VARCHAR(255),
        desired_titles JSONB DEFAULT '[]',
        preferred_locations JSONB DEFAULT '[]',
        remote_preference VARCHAR(50) DEFAULT 'flexible',
        min_salary INTEGER,
        industries JSONB DEFAULT '[]',
        embedding vector(1536),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
    CREATE INDEX IF NOT EXISTS idx_jobs_fetched_at ON jobs(fetched_at DESC);
    CREATE INDEX IF NOT EXISTS idx_jobs_salary ON jobs(salary_min, salary_max);
    
    -- Full-text search index
    CREATE INDEX IF NOT EXISTS idx_jobs_search ON jobs 
        USING GIN (to_tsvector('english', title || ' ' || company || ' ' || description));
    
    -- Vector similarity index (IVFFlat for faster approximate search)
    CREATE INDEX IF NOT EXISTS idx_jobs_embedding ON jobs 
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """
    
    def __init__(self, pool):
        """Initialize with an asyncpg connection pool."""
        self._pool = pool
    
    @classmethod
    async def create(cls, config: "DatabaseConfig") -> "PostgresJobStore":
        """Create a new PostgresJobStore with connection pool."""
        import asyncpg
        import ssl
        from pgvector.asyncpg import register_vector
        
        # Configure SSL for Azure PostgreSQL
        ssl_context = None
        if config.ssl_mode == "require":
            ssl_context = ssl.create_default_context()
            # Azure PostgreSQL uses a trusted CA, so we can verify
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        pool = await asyncpg.create_pool(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
            min_size=config.min_pool_size,
            max_size=config.max_pool_size,
            ssl=ssl_context,  # SSL for Azure PostgreSQL
            init=register_vector,  # Register pgvector type
        )
        
        store = cls(pool)
        await store._init_schema()
        return store
    
    async def _init_schema(self):
        """Initialize database schema."""
        async with self._pool.acquire() as conn:
            await conn.execute(self.INIT_SQL)
        logger.info("Database schema initialized")
    
    async def close(self):
        """Close the connection pool."""
        await self._pool.close()
    
    async def add(self, job: Job) -> Job:
        """Add a job to the store."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs (
                    id, title, company, location, description, url,
                    salary_min, salary_max, job_type, experience_level,
                    skills, source, status, score, posted_at, fetched_at, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    company = EXCLUDED.company,
                    location = EXCLUDED.location,
                    description = EXCLUDED.description,
                    url = EXCLUDED.url,
                    salary_min = EXCLUDED.salary_min,
                    salary_max = EXCLUDED.salary_max,
                    job_type = EXCLUDED.job_type,
                    experience_level = EXCLUDED.experience_level,
                    skills = EXCLUDED.skills,
                    source = EXCLUDED.source,
                    status = EXCLUDED.status,
                    score = EXCLUDED.score,
                    posted_at = EXCLUDED.posted_at,
                    fetched_at = EXCLUDED.fetched_at,
                    embedding = COALESCE(EXCLUDED.embedding, jobs.embedding),
                    updated_at = CURRENT_TIMESTAMP
                """,
                job.id, job.title, job.company, job.location, job.description,
                job.url, job.salary_min, job.salary_max, job.job_type,
                job.experience_level, json.dumps(job.skills), job.source,
                job.status.value, job.score, job.posted_at, job.fetched_at, job.embedding
            )
        return job
    
    async def add_many(self, jobs: list[Job]) -> list[Job]:
        """Add multiple jobs efficiently."""
        if not jobs:
            return jobs
        
        async with self._pool.acquire() as conn:
            # Use executemany for batch insert
            await conn.executemany(
                """
                INSERT INTO jobs (
                    id, title, company, location, description, url,
                    salary_min, salary_max, job_type, experience_level,
                    skills, source, status, score, posted_at, fetched_at, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    embedding = COALESCE(EXCLUDED.embedding, jobs.embedding),
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        job.id, job.title, job.company, job.location, job.description,
                        job.url, job.salary_min, job.salary_max, job.job_type,
                        job.experience_level, json.dumps(job.skills), job.source,
                        job.status.value, job.score, job.posted_at, job.fetched_at, job.embedding
                    )
                    for job in jobs
                ]
            )
        return jobs
    
    async def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM jobs WHERE id = $1", job_id
            )
        if row:
            return self._row_to_job(row)
        return None
    
    async def search(self, criteria: JobSearchCriteria) -> list[Job]:
        """Search jobs with full-text search and filters."""
        conditions = []
        params = []
        param_idx = 1
        
        # Full-text search on title, company, description
        if criteria.query:
            # Convert query to tsquery format
            query_terms = " & ".join(criteria.query.split())
            conditions.append(
                f"to_tsvector('english', title || ' ' || company || ' ' || description) "
                f"@@ to_tsquery('english', ${param_idx})"
            )
            params.append(query_terms)
            param_idx += 1
        
        # Location filter (case-insensitive partial match)
        if criteria.location:
            conditions.append(f"LOWER(location) LIKE ${param_idx}")
            params.append(f"%{criteria.location.lower()}%")
            param_idx += 1
        
        # Remote filter
        if criteria.remote_only:
            conditions.append(
                f"(LOWER(job_type) LIKE '%remote%' OR LOWER(location) LIKE '%remote%')"
            )
        
        # Salary filter
        if criteria.min_salary:
            conditions.append(
                f"(salary_max IS NULL OR salary_max >= ${param_idx})"
            )
            params.append(criteria.min_salary)
            param_idx += 1
        
        # Build query
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        query = f"""
            SELECT * FROM jobs
            WHERE {where_clause}
            ORDER BY fetched_at DESC
            LIMIT ${param_idx}
        """
        params.append(criteria.max_results)
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        return [self._row_to_job(row) for row in rows]
    
    async def search_by_embedding(
        self, 
        embedding: list[float], 
        limit: int = 20,
        min_similarity: float = 0.5
    ) -> list[tuple[Job, float]]:
        """Search jobs by embedding similarity using pgvector."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, 1 - (embedding <=> $1::vector) as similarity
                FROM jobs
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding, min_similarity, limit
            )
        
        return [(self._row_to_job(row), row['similarity']) for row in rows]
    
    async def update_embedding(self, job_id: str, embedding: list[float]) -> bool:
        """Update a job's embedding vector."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE jobs 
                SET embedding = $1::vector, updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
                """,
                embedding, job_id
            )
        return result == "UPDATE 1"
    
    async def update_status(self, job_id: str, status: JobStatus) -> Optional[Job]:
        """Update a job's status."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE jobs 
                SET status = $1, updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
                RETURNING *
                """,
                status.value, job_id
            )
        if row:
            return self._row_to_job(row)
        return None
    
    async def list_all(self, limit: int = 100) -> list[Job]:
        """List all jobs, newest first."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM jobs ORDER BY fetched_at DESC LIMIT $1",
                limit
            )
        return [self._row_to_job(row) for row in rows]
    
    async def delete(self, job_id: str) -> bool:
        """Delete a job by ID."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM jobs WHERE id = $1", job_id
            )
        return result == "DELETE 1"
    
    async def get_jobs_without_embeddings(self, limit: int = 100) -> list[Job]:
        """Get jobs that don't have embeddings yet (for batch processing)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM jobs 
                WHERE embedding IS NULL
                ORDER BY fetched_at DESC
                LIMIT $1
                """,
                limit
            )
        return [self._row_to_job(row) for row in rows]
    
    # User Profile methods
    async def save_profile(self, profile: UserProfile) -> UserProfile:
        """Save or update a user profile."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_profiles (
                    id, name, email, resume_text, summary, skills,
                    years_experience, current_title, desired_titles,
                    preferred_locations, remote_preference, min_salary,
                    industries, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    resume_text = EXCLUDED.resume_text,
                    summary = EXCLUDED.summary,
                    skills = EXCLUDED.skills,
                    years_experience = EXCLUDED.years_experience,
                    current_title = EXCLUDED.current_title,
                    desired_titles = EXCLUDED.desired_titles,
                    preferred_locations = EXCLUDED.preferred_locations,
                    remote_preference = EXCLUDED.remote_preference,
                    min_salary = EXCLUDED.min_salary,
                    industries = EXCLUDED.industries,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                """,
                profile.id, profile.name, profile.email, profile.resume_text,
                profile.summary, json.dumps(profile.skills), profile.years_experience,
                profile.current_title, json.dumps(profile.desired_titles),
                json.dumps(profile.preferred_locations), profile.remote_preference,
                profile.min_salary, json.dumps(profile.industries), profile.embedding
            )
        return profile
    
    async def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE id = $1", profile_id
            )
        if row:
            return self._row_to_profile(row)
        return None
    
    async def get_default_profile(self) -> Optional[UserProfile]:
        """Get the most recently updated user profile as default."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles ORDER BY updated_at DESC LIMIT 1"
            )
        if row:
            return self._row_to_profile(row)
        return None
    
    def _row_to_job(self, row) -> Job:
        """Convert a database row to a Job object."""
        skills = row['skills'] if isinstance(row['skills'], list) else json.loads(row['skills'] or '[]')
        return Job(
            id=row['id'],
            title=row['title'],
            company=row['company'],
            location=row['location'],
            description=row['description'],
            url=row['url'],
            salary_min=row['salary_min'],
            salary_max=row['salary_max'],
            job_type=row['job_type'],
            experience_level=row['experience_level'],
            skills=skills,
            source=row['source'],
            status=JobStatus(row['status']),
            score=row['score'],
            embedding=list(row['embedding']) if row['embedding'] is not None else None,
            posted_at=row['posted_at'],
            fetched_at=row['fetched_at'],
        )
    
    def _row_to_profile(self, row) -> UserProfile:
        """Convert a database row to a UserProfile object."""
        def parse_json_list(val):
            if isinstance(val, list):
                return val
            return json.loads(val or '[]')
        
        return UserProfile(
            id=row['id'],
            name=row['name'] or "",
            email=row['email'] or "",
            resume_text=row['resume_text'] or "",
            summary=row['summary'] or "",
            skills=parse_json_list(row['skills']),
            years_experience=row['years_experience'],
            current_title=row['current_title'] or "",
            desired_titles=parse_json_list(row['desired_titles']),
            preferred_locations=parse_json_list(row['preferred_locations']),
            remote_preference=row['remote_preference'] or "flexible",
            min_salary=row['min_salary'],
            industries=parse_json_list(row['industries']),
            embedding=list(row['embedding']) if row['embedding'] is not None else None,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )
    
    async def update_job_embeddings(self, job_embeddings: list[tuple[str, list[float]]]) -> int:
        """Update embeddings for multiple jobs."""
        if not job_embeddings:
            return 0
        
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                UPDATE jobs SET embedding = $2, updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
                """,
                job_embeddings
            )
        return len(job_embeddings)
    
    async def get_jobs_without_embeddings(self, limit: int = 100) -> list[Job]:
        """Get jobs that don't have embeddings yet."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM jobs WHERE embedding IS NULL LIMIT $1",
                limit
            )
        return [self._row_to_job(row) for row in rows]


async def get_store(config: Optional["DatabaseConfig"] = None) -> JobStore:
    """Factory function to get the appropriate job store.
    
    If config is provided and valid, returns PostgresJobStore.
    Otherwise, returns InMemoryJobStore.
    """
    if config and config.is_configured:
        try:
            store = await PostgresJobStore.create(config)
            logger.info(f"Connected to PostgreSQL at {config.host}:{config.port}/{config.database}")
            return store
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL: {e}. Falling back to in-memory store.")
    
    logger.info("Using in-memory job store")
    return InMemoryJobStore()
