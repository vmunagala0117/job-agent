"""Ranking service for matching jobs to user profiles using embeddings."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from openai import AsyncAzureOpenAI

from .config import AzureOpenAIConfig
from .models import Job, RankedJob, UserProfile


class EmbeddingService(ABC):
    """Abstract base class for embedding generation."""
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass


class AzureOpenAIEmbeddingService(EmbeddingService):
    """Azure OpenAI embedding service."""
    
    def __init__(self, config: AzureOpenAIConfig):
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        
        if config.api_key:
            self.client = AsyncAzureOpenAI(
                api_key=config.api_key,
                api_version=config.api_version,
                azure_endpoint=config.endpoint,
            )
        else:
            # Use managed identity
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=config.api_version,
                azure_endpoint=config.endpoint,
            )
        self.model = config.embedding_model
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        # Azure OpenAI supports batch embedding
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [d.embedding for d in response.data]


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing."""
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
    
    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic mock embedding based on text hash."""
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        # Generate pseudo-random but deterministic values
        values = []
        for i in range(0, min(len(h), self.dimensions * 2), 2):
            val = int(h[i:i+2], 16) / 255.0 - 0.5
            values.append(val)
        # Pad if needed
        while len(values) < self.dimensions:
            values.append(0.0)
        return values[:self.dimensions]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


@dataclass
class RankingWeights:
    """Weights for different ranking factors."""
    
    similarity: float = 0.5  # Embedding similarity weight
    skills: float = 0.25  # Skill match weight
    location: float = 0.15  # Location preference weight
    salary: float = 0.10  # Salary match weight


class RankingService:
    """Service for ranking jobs against user profiles."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        weights: Optional[RankingWeights] = None,
    ):
        self.embedding_service = embedding_service
        self.weights = weights or RankingWeights()
    
    async def embed_user_profile(self, profile: UserProfile) -> UserProfile:
        """Generate and store embedding for user profile."""
        # Combine resume and summary for embedding
        text = f"{profile.summary}\n\n{profile.resume_text}"
        if profile.skills:
            text += f"\n\nSkills: {', '.join(profile.skills)}"
        if profile.desired_titles:
            text += f"\n\nDesired roles: {', '.join(profile.desired_titles)}"
        
        profile.embedding = await self.embedding_service.embed(text)
        return profile
    
    async def embed_jobs(self, jobs: list[Job]) -> list[Job]:
        """Generate embeddings for jobs that don't have them.
        
        Returns the jobs with embeddings populated.
        """
        if not jobs:
            return jobs
        
        # Find jobs that need embeddings
        jobs_needing_embeddings = [j for j in jobs if j.embedding is None]
        if not jobs_needing_embeddings:
            return jobs
        
        # Generate embeddings for jobs without them
        job_texts = [self._job_to_text(job) for job in jobs_needing_embeddings]
        embeddings = await self.embedding_service.embed_batch(job_texts)
        
        # Assign embeddings to jobs
        for job, embedding in zip(jobs_needing_embeddings, embeddings):
            job.embedding = embedding
        
        return jobs
    
    async def rank_jobs(
        self,
        jobs: list[Job],
        profile: UserProfile,
        top_k: Optional[int] = None,
    ) -> list[RankedJob]:
        """Rank jobs against user profile and return sorted results."""
        if not jobs:
            return []
        
        # Ensure profile has embedding
        if profile.embedding is None:
            profile = await self.embed_user_profile(profile)
        
        # Generate embeddings only for jobs that don't have them
        jobs_needing_embeddings = [(i, j) for i, j in enumerate(jobs) if j.embedding is None]
        if jobs_needing_embeddings:
            indices, jobs_to_embed = zip(*jobs_needing_embeddings)
            job_texts = [self._job_to_text(job) for job in jobs_to_embed]
            new_embeddings = await self.embedding_service.embed_batch(job_texts)
            for idx, embedding in zip(indices, new_embeddings):
                jobs[idx].embedding = embedding
        
        # Score each job using its embedding
        ranked_jobs = []
        for job in jobs:
            ranked_job = self._score_job(job, job.embedding, profile)
            ranked_jobs.append(ranked_job)
        
        # Sort by score descending
        ranked_jobs.sort(key=lambda rj: rj.score, reverse=True)
        
        # Return top K if specified
        if top_k is not None:
            ranked_jobs = ranked_jobs[:top_k]
        
        return ranked_jobs
    
    def _job_to_text(self, job: Job) -> str:
        """Convert job to text for embedding."""
        parts = [job.title, job.company]
        if job.description:
            parts.append(job.description)
        if job.skills:
            parts.append(f"Skills: {', '.join(job.skills)}")
        if job.location:
            parts.append(f"Location: {job.location}")
        return "\n".join(parts)
    
    def _score_job(
        self,
        job: Job,
        job_embedding: list[float],
        profile: UserProfile,
    ) -> RankedJob:
        """Calculate composite score for a job."""
        # Similarity score (embedding-based)
        similarity_score = 0.0
        if profile.embedding:
            similarity_score = (cosine_similarity(profile.embedding, job_embedding) + 1) / 2
        
        # Skills match score
        skill_match_score = self._calculate_skill_match(job, profile)
        
        # Location score
        location_score = self._calculate_location_match(job, profile)
        
        # Salary score
        salary_score = self._calculate_salary_match(job, profile)
        
        # Weighted composite score
        composite_score = (
            self.weights.similarity * similarity_score +
            self.weights.skills * skill_match_score +
            self.weights.location * location_score +
            self.weights.salary * salary_score
        )
        
        # Generate justification
        justification = self._generate_justification(
            job, profile, similarity_score, skill_match_score, location_score, salary_score
        )
        
        return RankedJob(
            job=job,
            score=composite_score,
            justification=justification,
            similarity_score=similarity_score,
            skill_match_score=skill_match_score,
            location_score=location_score,
            salary_score=salary_score,
        )
    
    def _calculate_skill_match(self, job: Job, profile: UserProfile) -> float:
        """Calculate skill overlap score."""
        if not job.skills or not profile.skills:
            return 0.5  # Neutral if no data
        
        job_skills = {s.lower() for s in job.skills}
        profile_skills = {s.lower() for s in profile.skills}
        
        if not job_skills:
            return 0.5
        
        matches = job_skills & profile_skills
        return len(matches) / len(job_skills)
    
    def _calculate_location_match(self, job: Job, profile: UserProfile) -> float:
        """Calculate location preference match."""
        if not job.location:
            return 0.5  # Neutral if no data
        
        job_loc = job.location.lower()
        
        # Check for remote
        if "remote" in job_loc:
            if profile.remote_preference in ("remote", "flexible"):
                return 1.0
            elif profile.remote_preference == "hybrid":
                return 0.7
            else:
                return 0.3
        
        # Check preferred locations
        if profile.preferred_locations:
            for pref_loc in profile.preferred_locations:
                if pref_loc.lower() in job_loc or job_loc in pref_loc.lower():
                    return 1.0
            return 0.3  # Not in preferred locations
        
        return 0.5  # Neutral if no preference
    
    def _calculate_salary_match(self, job: Job, profile: UserProfile) -> float:
        """Calculate salary match score."""
        if not profile.min_salary:
            return 0.5  # Neutral if no preference
        
        if job.salary_max is None and job.salary_min is None:
            return 0.5  # Neutral if no job salary data
        
        # Use max salary if available, else min
        job_salary = job.salary_max or job.salary_min or 0
        
        if job_salary >= profile.min_salary:
            return 1.0
        
        # Partial credit for close matches
        ratio = job_salary / profile.min_salary
        return max(0.0, ratio)
    
    def _generate_justification(
        self,
        job: Job,
        profile: UserProfile,
        similarity: float,
        skills: float,
        location: float,
        salary: float,
    ) -> str:
        """Generate human-readable justification for ranking."""
        reasons = []
        
        if similarity >= 0.7:
            reasons.append("Strong match with your experience and background")
        elif similarity >= 0.5:
            reasons.append("Good alignment with your profile")
        
        if skills >= 0.8:
            matching_skills = set(s.lower() for s in job.skills or []) & set(s.lower() for s in profile.skills)
            if matching_skills:
                reasons.append(f"Matches your skills: {', '.join(list(matching_skills)[:3])}")
        elif skills >= 0.5:
            reasons.append("Some skill overlap with your profile")
        
        if location >= 0.9:
            if "remote" in (job.location or "").lower():
                reasons.append("Remote position matches your preference")
            else:
                reasons.append("Location matches your preference")
        
        if salary >= 0.9 and (job.salary_min or job.salary_max):
            reasons.append("Salary meets your requirements")
        
        if not reasons:
            reasons.append("May be worth exploring based on overall fit")
        
        return "; ".join(reasons) + "."


def get_embedding_service(config: Optional[AzureOpenAIConfig] = None) -> EmbeddingService:
    """Factory function to get appropriate embedding service."""
    if config and config.is_configured:
        return AzureOpenAIEmbeddingService(config)
    return MockEmbeddingService()


def get_ranking_service(
    config: Optional[AzureOpenAIConfig] = None,
    weights: Optional[RankingWeights] = None,
) -> RankingService:
    """Factory function to create ranking service."""
    embedding_service = get_embedding_service(config)
    return RankingService(embedding_service, weights)
