# Job Agent (Azure Agent Framework)

A production-intent job search assistant using Microsoft Agent Framework. It exposes an HTTP-hosted agent that can search for jobs (via SerpAPI/Google Jobs), rank them against your resume using embeddings, and help manage your application workflow.

## Features

- **Job Search**: Search for jobs using SerpAPI's Google Jobs API (aggregates LinkedIn, Indeed, Glassdoor, etc.)
- **Resume Parsing**: Upload PDF/DOCX resumes to auto-extract skills, experience, and create embeddings
- **Resume-Based Ranking**: Get jobs ranked by fit using Azure OpenAI embeddings and skill matching
- **Job Management**: Track job status (new, applied, rejected, interviewing)
- **Persistent Storage**: PostgreSQL with pgvector for job and profile persistence
- **Conversational Interface**: Natural language interaction with the agent

## Architecture

```
src/job_agent/
├── config.py        # Configuration (Azure OpenAI, SerpAPI, PostgreSQL)
├── clients.py       # Azure OpenAI client factory
├── models.py        # Data models (Job, UserProfile, RankedJob)
├── store.py         # Job storage (InMemoryJobStore, PostgresJobStore)
├── providers.py     # Job ingestion (SerpAPI, Mock)
├── ranking.py       # Embedding-based job ranking service
├── resume_parser.py # PDF/DOCX resume parsing and skill extraction
├── tools.py         # Agent tools (search, rank, manage)
├── workflows.py     # Agent workflow and coordinator
└── server.py        # HTTP server entry point

scripts/
└── init_db.py     # Database initialization script
```

## Prerequisites
- Python 3.10+ (tested with 3.14)
- Azure subscription with Azure OpenAI deployment (chat model + embeddings model)
- SerpAPI account (optional, for real job data)
- Auth: `az login` for `DefaultAzureCredential` or configure a service principal

## Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Copy `.env.example` to `.env` and configure:
   ```env
   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
   AZURE_OPENAI_DEPLOYMENT_NAME=<chat-deployment>
   AZURE_OPENAI_API_KEY=<optional-if-using-managed-identity>
   SERPAPI_API_KEY=<your-serpapi-key>
   
   # PostgreSQL (optional - defaults to in-memory storage)
   DATABASE_URL=postgresql://user:password@localhost:5432/job_agent
   # Or individual components:
   # DB_HOST=localhost
   # DB_PORT=5432
   # DB_NAME=job_agent
   # DB_USER=postgres
   # DB_PASSWORD=your_password
   ```

4. (Optional) Initialize PostgreSQL database:
   ```bash
   python scripts/init_db.py
   ```

5. Run the server:
   ```bash
   python -m job_agent.server --server
   ```

## Testing

### Quick Test with CLI

Run the interactive test CLI (no server required):

```bash
python test_cli.py
```

This starts an interactive session where you can:
1. Search for jobs (uses mock data if SerpAPI not configured)
2. Set your profile with skills and preferences
3. Rank jobs against your profile
4. Manage job status

### Example Test Session

```
You: Search for Python developer jobs in Seattle
Agent: Found 3 jobs matching 'Python developer':
1. Senior Python Developer at TechCorp (Seattle, WA) | $140,000-$180,000 [ID: abc12345]
2. Python Backend Engineer at StartupXYZ (Remote) | $120,000-$160,000 [ID: def67890]
...

You: Set my profile: My name is John, I have 5 years of experience with Python, AWS, and Kubernetes. I prefer remote work and minimum salary of $150,000.
Agent: Profile set for John with 3 skills. Ready to rank jobs.

You: Rank my saved jobs
Agent: Top 3 job matches for John:
1. Python Backend Engineer at StartupXYZ - Score: 85.2%
   Matches your skills: python, aws; Remote position matches your preference.
   [ID: def67890]
...
```

### Testing Without Azure OpenAI

If you don't have Azure OpenAI access yet, you can still test the data models and providers:

```python
# test_offline.py
import asyncio
from job_agent.providers import MockJobProvider
from job_agent.models import JobSearchCriteria
from job_agent.ranking import MockEmbeddingService, RankingService

async def test():
    provider = MockJobProvider()
    jobs = await provider.fetch_jobs(JobSearchCriteria(query="python"))
    print(f"Found {len(jobs)} mock jobs")
    for job in jobs:
        print(f"  - {job.title} at {job.company}")

asyncio.run(test())
```

## Available Agent Commands

The agent understands natural language. Example interactions:

### Job Search
- "Search for Python developer jobs in Seattle"
- "Find remote machine learning engineer positions"
- "Look for data scientist jobs with salary above $150,000"

### Profile Setup
- "Set my profile with skills Python, AWS, Kubernetes"
- "I prefer remote work and minimum salary of $180,000"

### Job Ranking
- "Rank my saved jobs"
- "Show me the best job matches"

### Job Management
- "List my saved jobs"
- "Show details for job abc123"
- "Mark job abc123 as applied"

## Configuration Options

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Chat model deployment name (e.g., gpt-4o) | Yes |
| `AZURE_OPENAI_API_KEY` | API key (leave empty for managed identity) | No |
| `AZURE_OPENAI_API_VERSION` | API version (default: 2024-02-15-preview) | No |
| `AZURE_OPENAI_EMBEDDING_MODEL` | Embedding model name (default: text-embedding-3-small) | No |
| `SERPAPI_API_KEY` | SerpAPI key for Google Jobs | No* |
| `DATABASE_URL` | PostgreSQL connection URL | No** |
| `DB_HOST` | PostgreSQL host (default: localhost) | No** |
| `DB_PORT` | PostgreSQL port (default: 5432) | No** |
| `DB_NAME` | Database name (default: job_agent) | No** |
| `DB_USER` | Database user (default: postgres) | No** |
| `DB_PASSWORD` | Database password | No** |

*Without SerpAPI, the agent uses mock job data for testing.

**Without PostgreSQL, the agent uses in-memory storage (data lost on restart).

## PostgreSQL Storage

For persistent storage with vector similarity search, configure PostgreSQL with pgvector:

### Quick Setup (Docker)

```bash
# Start PostgreSQL with pgvector
docker run -d --name job-agent-db \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -e POSTGRES_DB=job_agent \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Set environment variable
export DATABASE_URL=postgresql://postgres:mysecretpassword@localhost:5432/job_agent

# Initialize schema
python scripts/init_db.py
```

### Azure Database for PostgreSQL

For production, use Azure Database for PostgreSQL Flexible Server:

1. Create a PostgreSQL Flexible Server with pgvector extension enabled
2. Set the `DATABASE_URL` environment variable
3. Run `python scripts/init_db.py` to create tables

### Features

- **Full-text search**: Fast search across job titles, companies, and descriptions
- **Vector similarity**: Find jobs semantically similar to your profile using pgvector
- **Persistent profiles**: User profiles and preferences stored across sessions
- **Batch embedding updates**: Efficiently update job embeddings in batches

## How Ranking Works

1. **Profile Embedding**: Your resume and skills are embedded using Azure OpenAI
2. **Job Embedding**: Each job's title, description, and skills are embedded
3. **Similarity Scoring**: Cosine similarity between profile and job embeddings
4. **Heuristic Boosts**: Additional scores for skill overlap (25%), location (15%), salary (10%)
5. **Composite Score**: Weighted combination of all factors (0-100%)

## Next Steps

- [x] Persistent storage (PostgreSQL + pgvector)
- [x] Resume parsing (PDF/DOCX with skill extraction)
- [ ] Application-prep agent (resume tailoring, cover letters)
- [ ] Notification delivery (email/Slack/Teams)
- [ ] Scheduled job ingestion
- [ ] Evaluation and tracing
