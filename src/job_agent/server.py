from __future__ import annotations

from azure.ai.agentserver.agentframework import from_agent_framework

from .clients import build_azure_openai_client
from .config import AppConfig
from .workflows import build_agent


def main() -> None:
    config = AppConfig.load()
    client = build_azure_openai_client(config.azure_openai)
    agent = build_agent(client)
    from_agent_framework(agent).run()


if __name__ == "__main__":
    main()
