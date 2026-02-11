#!/usr/bin/env python
"""Initialize the PostgreSQL database for job-agent.

This script creates the database schema including tables and indexes.
Run this once before starting the application with PostgreSQL storage.

Usage:
    python scripts/init_db.py

Environment variables:
    DATABASE_URL - Full connection string (e.g., postgresql://user:pass@host:5432/db)
    
    Or individual components:
    DB_HOST - Database host (default: localhost)
    DB_PORT - Database port (default: 5432)
    DB_NAME - Database name (default: job_agent)
    DB_USER - Database user (default: postgres)
    DB_PASSWORD - Database password
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from job_agent.config import DatabaseConfig


async def init_database():
    """Initialize the database schema."""
    import asyncpg
    import ssl as ssl_module
    
    config = DatabaseConfig.from_env()
    
    if not config.is_configured:
        print("Error: Database not configured. Set DATABASE_URL or DB_* environment variables.")
        sys.exit(1)
    
    print(f"Connecting to PostgreSQL at {config.host}:{config.port}...")
    
    # Configure SSL for Azure PostgreSQL
    ssl_context = None
    if config.ssl_mode == "require":
        ssl_context = ssl_module.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl_module.CERT_REQUIRED
        print("Using SSL connection (required for Azure PostgreSQL)")
    
    try:
        # First, connect to default 'postgres' database to create our database if needed
        sys_conn = await asyncpg.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database="postgres",
            ssl=ssl_context,
        )
        
        # Check if database exists
        exists = await sys_conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            config.database
        )
        
        if not exists:
            print(f"Creating database '{config.database}'...")
            await sys_conn.execute(f'CREATE DATABASE "{config.database}"')
            print(f"Database '{config.database}' created.")
        else:
            print(f"Database '{config.database}' already exists.")
        
        await sys_conn.close()
        
        # Now connect to our database and create schema
        conn = await asyncpg.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database=config.database,
            ssl=ssl_context,
        )
        
        print("Creating pgvector extension...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        print("Creating tables...")
        await conn.execute("""
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
                embedding vector(1536),
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
            
            -- Feedback table for user feedback on job matches
            CREATE TABLE IF NOT EXISTS feedback (
                id VARCHAR(36) PRIMARY KEY,
                job_id VARCHAR(36) REFERENCES jobs(id) ON DELETE CASCADE,
                feedback_type VARCHAR(50) NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Application packages table for prepared application materials
            CREATE TABLE IF NOT EXISTS application_packages (
                id VARCHAR(36) PRIMARY KEY,
                job_id VARCHAR(36) REFERENCES jobs(id) ON DELETE CASCADE,
                profile_id VARCHAR(36) REFERENCES user_profiles(id) ON DELETE CASCADE,
                resume_suggestions JSONB,
                cover_letter TEXT,
                intro_email TEXT,
                recruiters JSONB DEFAULT '[]',
                status VARCHAR(50) DEFAULT 'draft',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        print("Creating indexes...")
        await conn.execute("""
            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
            CREATE INDEX IF NOT EXISTS idx_jobs_fetched_at ON jobs(fetched_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jobs_salary ON jobs(salary_min, salary_max);
            
            -- Full-text search index
            CREATE INDEX IF NOT EXISTS idx_jobs_search ON jobs 
                USING GIN (to_tsvector('english', title || ' ' || company || ' ' || description));
            
            -- Feedback indexes
            CREATE INDEX IF NOT EXISTS idx_feedback_job_id ON feedback(job_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC);
            
            -- Application packages indexes
            CREATE INDEX IF NOT EXISTS idx_packages_job_id ON application_packages(job_id);
            CREATE INDEX IF NOT EXISTS idx_packages_profile_id ON application_packages(profile_id);
            CREATE INDEX IF NOT EXISTS idx_packages_status ON application_packages(status);
            CREATE INDEX IF NOT EXISTS idx_packages_created_at ON application_packages(created_at DESC);
        """)
        
        # Create vector index only if there are jobs with embeddings
        job_count = await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE embedding IS NOT NULL")
        if job_count >= 100:
            print("Creating vector similarity index (IVFFlat)...")
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_embedding ON jobs 
                    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """)
        else:
            print(f"Skipping vector index (need 100+ jobs with embeddings, have {job_count})")
        
        await conn.close()
        
        print("\n✓ Database initialization complete!")
        print(f"  Host: {config.host}:{config.port}")
        print(f"  Database: {config.database}")
        print(f"  User: {config.user}")
        
    except asyncpg.InvalidCatalogNameError:
        print(f"Error: Database '{config.database}' does not exist and could not be created.")
        print("Please create it manually: CREATE DATABASE job_agent;")
        sys.exit(1)
    except asyncpg.InvalidPasswordError:
        print("Error: Invalid password for database user.")
        sys.exit(1)
    except asyncpg.CannotConnectNowError as e:
        print(f"Error: Cannot connect to database: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Job Agent Database Initialization")
    print("=" * 40)
    asyncio.run(init_database())
