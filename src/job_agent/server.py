from __future__ import annotations

import asyncio

from azure.ai.agentserver.agentframework import from_agent_framework

from .clients import build_azure_openai_client
from .config import AppConfig
from .workflows import create_agent


async def _create_agent():
    """Async helper to create the agent with database connection."""
    config = AppConfig.load()
    client = build_azure_openai_client(config.azure_openai)
    return await create_agent(client, use_database=True)


def main() -> None:
    # Create agent with async database initialization
    agent = asyncio.run(_create_agent())
    from_agent_framework(agent).run()


if __name__ == "__main__":
    main()
