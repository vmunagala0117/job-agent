from __future__ import annotations

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from .config import AzureOpenAIConfig


def build_azure_openai_client(config: AzureOpenAIConfig) -> AzureOpenAIChatClient:
    """Build Azure OpenAI client with API key or managed identity."""
    if config.api_key:
        # Use API key authentication
        return AzureOpenAIChatClient(
            endpoint=config.endpoint,
            deployment_name=config.deployment_name,
            credential=AzureKeyCredential(config.api_key),
        )
    else:
        # Use DefaultAzureCredential (az login / managed identity)
        credential = DefaultAzureCredential()
        return AzureOpenAIChatClient(
            endpoint=config.endpoint,
            deployment_name=config.deployment_name,
            credential=credential,
        )
