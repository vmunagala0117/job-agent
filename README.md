# Job Agent (Azure Agent Framework)

Initial scaffold for a production-intent job search assistant using Microsoft Agent Framework. It exposes a minimal HTTP-hosted agent (via azure-ai-agentserver) that can later orchestrate job ingestion, ranking, and application-prep flows.

## Prereqs
- Python 3.10+
- Azure subscription with Azure OpenAI deployment (chat model)
- Auth: `az login` for `DefaultAzureCredential` or configure a service principal

## Setup
1) Create/activate a virtual environment
2) Install deps: `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and fill values:
   - `AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com`
   - `AZURE_OPENAI_DEPLOYMENT_NAME=<chat-deployment>`
4) Run the server: `python -m job_agent.server --server`

## What’s here
- Minimal workflow with a coordinator agent that handles inbound chat and yields a response
- HTTP server wrapper using `azure.ai.agentserver.agentframework`
- Configuration and client factory for Azure OpenAI

## Next steps
- Add ingestion providers (SerpAPI/Proxycurl) and a job store
- Implement ranking and application-prep executors and wire them into the workflow
- Add notification delivery (email/Slack/Teams) and feedback loop
- Expand evaluation, tracing, and deployment tasks once core logic solidifies
