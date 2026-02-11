#!/usr/bin/env python
"""Quick test: verify OTel traces are emitted by Agent Framework.

Run with:
    ENABLE_INSTRUMENTATION=true ENABLE_CONSOLE_EXPORTERS=true python scripts/test_otel.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Force observability ON for this test
os.environ["ENABLE_INSTRUMENTATION"] = "true"
os.environ["ENABLE_CONSOLE_EXPORTERS"] = "true"
os.environ["ENABLE_SENSITIVE_DATA"] = "true"

from dotenv import load_dotenv
load_dotenv()

from agent_framework.observability import configure_otel_providers
configure_otel_providers()

from agent_framework import ChatMessage, Role, TextContent
from job_agent.clients import build_azure_openai_client
from job_agent.config import AppConfig
from job_agent.workflows import create_agent


async def main():
    print("=" * 60)
    print("OpenTelemetry Trace Test")
    print("=" * 60)

    config = AppConfig.load()
    client = build_azure_openai_client(config.azure_openai)
    agent, _store, _ranking = await create_agent(client, use_database=False)

    messages = [
        ChatMessage(role=Role.USER, contents=[TextContent(text="Hello, what can you do?")])
    ]

    print("\nSending message... (trace spans will appear below)\n")
    response = await agent.run(messages)
    print(f"\n{'=' * 60}")
    print(f"Response ({len(response.text)} chars): {response.text[:200]}...")
    print("=" * 60)
    print("\nLook for spans above named:")
    print("  - workflow.run")
    print("  - executor.process")
    print("  - invoke_agent <name>")
    print("  - chat <model>")
    print("  - execute_tool <function>")


if __name__ == "__main__":
    asyncio.run(main())
