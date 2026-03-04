from __future__ import annotations

import asyncio
import logging

from azure.ai.agentserver.agentframework import from_agent_framework

from .clients import build_azure_openai_client
from .config import AppConfig
from .workflows import create_agent

logger = logging.getLogger(__name__)


async def _create_agent():
    """Async helper to create the agent with database connection."""
    config = AppConfig.load()
    client = build_azure_openai_client(config.azure_openai)
    agent, _store, _ranking = await create_agent(client, use_database=True)
    logger.info("Agent created successfully")
    return agent


def main() -> None:
    logger.info("Starting Job Agent server")
    try:
        # Create agent with async database initialization
        agent = asyncio.run(_create_agent())
        from_agent_framework(agent).run()
    except Exception:
        logger.exception("Server startup failed")
        raise


if __name__ == "__main__":
    main()
