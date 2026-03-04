from __future__ import annotations

import logging

from agent_framework.azure import AzureOpenAIChatClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from .config import AzureOpenAIConfig

logger = logging.getLogger(__name__)


def build_azure_openai_client(config: AzureOpenAIConfig) -> AzureOpenAIChatClient:
    """Build Azure OpenAI client with API key or managed identity."""
    if config.api_key:
        logger.info("Building Azure OpenAI client with API key (endpoint=%s)", config.endpoint)
        # Use API key authentication
        return AzureOpenAIChatClient(
            endpoint=config.endpoint,
            deployment_name=config.deployment_name,
            credential=AzureKeyCredential(config.api_key),
        )
    else:
        logger.info("Building Azure OpenAI client with managed identity (endpoint=%s)", config.endpoint)
        # Use DefaultAzureCredential (az login / managed identity)
        credential = DefaultAzureCredential()
        return AzureOpenAIChatClient(
            endpoint=config.endpoint,
            deployment_name=config.deployment_name,
            credential=credential,
        )
