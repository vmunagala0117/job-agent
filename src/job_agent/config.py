from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import os


@dataclass
class AzureOpenAIConfig:
    endpoint: str
    deployment_name: str
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    embedding_model: str = "text-embedding-3-small"

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        # override=True ensures deployed env vars take precedence
        load_dotenv(override=True)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip() or None
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()
        embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        
        if not endpoint or not deployment:
            pairs = [
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_DEPLOYMENT_NAME", deployment),
            ]
            missing = [name for name, value in pairs if not value]
            raise ValueError(f"Missing required Azure OpenAI config: {', '.join(missing)}")
        return cls(
            endpoint=endpoint,
            deployment_name=deployment,
            api_key=api_key,
            api_version=api_version,
            embedding_model=embedding_model,
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        return bool(self.endpoint and self.deployment_name)


@dataclass
class SerpAPIConfig:
    """Configuration for SerpAPI job search."""
    api_key: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "SerpAPIConfig":
        load_dotenv(override=True)
        api_key = os.getenv("SERPAPI_API_KEY", "").strip() or None
        return cls(api_key=api_key)
    
    @property
    def is_configured(self) -> bool:
        """Check if SerpAPI is properly configured."""
        return bool(self.api_key)


@dataclass
class DatabaseConfig:
    """Configuration for PostgreSQL database."""
    host: str = "localhost"
    port: int = 5432
    database: str = "job_agent"
    user: str = "postgres"
    password: Optional[str] = None
    ssl_mode: str = "prefer"
    min_pool_size: int = 2
    max_pool_size: int = 10
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        load_dotenv(override=True)
        
        # Support both DATABASE_URL and individual components
        database_url = os.getenv("DATABASE_URL", "").strip()
        if database_url:
            return cls.from_url(database_url)
        
        return cls(
            host=os.getenv("DB_HOST", "localhost").strip(),
            port=int(os.getenv("DB_PORT", "5432").strip()),
            database=os.getenv("DB_NAME", "job_agent").strip(),
            user=os.getenv("DB_USER", "postgres").strip(),
            password=os.getenv("DB_PASSWORD", "").strip() or None,
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer").strip(),
            min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", "2").strip()),
            max_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "10").strip()),
        )
    
    @classmethod
    def from_url(cls, url: str) -> "DatabaseConfig":
        """Parse a DATABASE_URL into config components."""
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") or "job_agent",
            user=parsed.username or "postgres",
            password=parsed.password,
            ssl_mode="require" if "sslmode=require" in url else "prefer",
        )
    
    @property
    def dsn(self) -> str:
        """Return connection string for asyncpg."""
        password_part = f":{self.password}" if self.password else ""
        return f"postgresql://{self.user}{password_part}@{self.host}:{self.port}/{self.database}"
    
    @property
    def is_configured(self) -> bool:
        """Check if database is properly configured."""
        return bool(self.host and self.database and self.user)


@dataclass
class CronConfig:
    """Configuration for automated daily job search."""
    enabled: bool = False
    schedule: str = "0 6 * * *"  # cron expression: default 6 AM daily
    api_key: str = ""  # Shared secret for X-Cron-Key header
    app_url: str = "http://localhost:8080"  # Base URL of the running webapp

    @classmethod
    def from_env(cls) -> "CronConfig":
        load_dotenv(override=True)
        return cls(
            enabled=os.getenv("CRON_ENABLED", "false").strip().lower() in ("true", "1", "yes"),
            schedule=os.getenv("CRON_SCHEDULE", "0 6 * * *").strip(),
            api_key=os.getenv("CRON_API_KEY", "").strip(),
            app_url=os.getenv("CRON_APP_URL", "http://localhost:8080").strip(),
        )

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.api_key)


@dataclass
class AppConfig:
    azure_openai: AzureOpenAIConfig
    serpapi: SerpAPIConfig
    database: Optional[DatabaseConfig] = None
    cron: Optional[CronConfig] = None

    @classmethod
    def load(cls) -> "AppConfig":
        db_config = DatabaseConfig.from_env()
        cron_config = CronConfig.from_env()
        return cls(
            azure_openai=AzureOpenAIConfig.from_env(),
            serpapi=SerpAPIConfig.from_env(),
            database=db_config if db_config.is_configured else None,
            cron=cron_config if cron_config.is_configured else None,
        )
