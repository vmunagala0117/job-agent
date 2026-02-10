from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv
import os


@dataclass
class AzureOpenAIConfig:
    endpoint: str
    deployment_name: str

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        # override=True ensures deployed env vars take precedence
        load_dotenv(override=True)
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
        if not endpoint or not deployment:
            pairs = [
                ("AZURE_OPENAI_ENDPOINT", endpoint),
                ("AZURE_OPENAI_DEPLOYMENT_NAME", deployment),
            ]
            missing = [name for name, value in pairs if not value]
            raise ValueError(f"Missing required Azure OpenAI config: {', '.join(missing)}")
        return cls(endpoint=endpoint, deployment_name=deployment)


@dataclass
class AppConfig:
    azure_openai: AzureOpenAIConfig

    @classmethod
    def load(cls) -> "AppConfig":
        return cls(azure_openai=AzureOpenAIConfig.from_env())
