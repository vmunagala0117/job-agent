from __future__ import annotations

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential

from .config import AzureOpenAIConfig


def build_azure_openai_client(config: AzureOpenAIConfig) -> AzureOpenAIChatClient:
    # DefaultAzureCredential uses az login / env vars; suitable for local and hosted use
    credential = DefaultAzureCredential()
    return AzureOpenAIChatClient(
        endpoint=config.endpoint,
        deployment_name=config.deployment_name,
        credential=credential,
    )
