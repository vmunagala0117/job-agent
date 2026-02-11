#!/usr/bin/env python
"""Interactive CLI for testing the job agent locally."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Enable agent tracing — shows classifier decisions, routing, and tool calls
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)
# Show our workflow traces but keep framework/OpenAI noise quiet
logging.getLogger("job_agent.workflows").setLevel(logging.INFO)

# --- OpenTelemetry ---
# Agent Framework has built-in OTel support.  Set env vars to control:
#   ENABLE_INSTRUMENTATION=true        → emit spans for invoke_agent, chat, execute_tool
#   ENABLE_SENSITIVE_DATA=true          → include prompts/responses in span attributes
#   OTEL_EXPORTER_OTLP_ENDPOINT=...    → send to Aspire Dashboard / Jaeger / etc.
#   ENABLE_CONSOLE_EXPORTERS=true       → print spans to console (noisy but useful)
from dotenv import load_dotenv
load_dotenv()

from agent_framework.observability import configure_otel_providers
configure_otel_providers()   # reads OTEL_* and ENABLE_* from env/.env

from job_agent.clients import build_azure_openai_client
from job_agent.config import AppConfig
from job_agent.workflows import create_agent


async def main():
    print("=" * 60)
    print("Job Agent - Interactive Test CLI")
    print("=" * 60)
    
    # Load config
    try:
        config = AppConfig.load()
        print(f"✓ Azure OpenAI endpoint: {config.azure_openai.endpoint}")
        print(f"✓ Deployment: {config.azure_openai.deployment_name}")
        
        if config.serpapi.is_configured:
            print("✓ SerpAPI: Configured (real job data)")
        else:
            print("⚠ SerpAPI: Not configured (using mock data)")
        
        if config.database and config.database.is_configured:
            print(f"✓ PostgreSQL: {config.database.host}:{config.database.port}/{config.database.database}")
        else:
            print("⚠ PostgreSQL: Not configured (using in-memory storage)")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        print("\nMake sure you have a .env file with:")
        print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com")
        print("  AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o")
        return
    
    # Build agent using the async factory (handles DB init)
    print("\nInitializing multi-agent workflow...")
    client = build_azure_openai_client(config.azure_openai)
    use_database = bool(config.database and config.database.is_configured)
    agent, _store, _ranking = await create_agent(client, use_database=use_database)
    
    print("✓ Agent ready! (Coordinator → JobSearch + AppPrep)\n")
    print("Commands:")
    print("  - Type your message and press Enter")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'help' for example commands")
    print("-" * 60)
    
    from agent_framework import ChatMessage, Role, TextContent
    
    conversation: list[ChatMessage] = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        
        if user_input.lower() == "help":
            print("\nExample commands:")
            print("  - Search for Python developer jobs in Seattle")
            print("  - Find remote machine learning engineer positions")
            print("  - List my saved jobs")
            print("  - Set my profile: My name is John, I have 5 years experience")
            print("    with Python, AWS, and Docker. I prefer remote work.")
            print("  - Rank my saved jobs")
            print("  - Show details for job <id>")
            print("  - Mark job <id> as applied")
            continue
        
        # Add user message to conversation
        conversation.append(
            ChatMessage(
                role=Role.USER,
                contents=[TextContent(text=user_input)],
            )
        )
        
        # Get response from agent
        try:
            response = await agent.run(conversation)
            assistant_text = response.text or "(No response)"
            
            # Add assistant response to conversation for context
            conversation.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[TextContent(text=assistant_text)],
                )
            )
            
            print(f"\nAgent: {assistant_text}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            # Remove the failed user message
            conversation.pop()


if __name__ == "__main__":
    asyncio.run(main())
