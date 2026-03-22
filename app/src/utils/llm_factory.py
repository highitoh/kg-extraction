"""Factory class for creating chat model instances with consistent configuration."""

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI

DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite-preview"
_SUPPORTED_LLM_PROVIDERS = {"gemini_api", "gcp_vertexai", "openai_api", "azure_openai"}
_GEMINI_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


class LLMFactory:
    """Factory class for creating LLM instances with standardized configuration."""

    @staticmethod
    def _get_optional_int_env(env_name: str) -> Optional[int]:
        value = os.getenv(env_name)
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_provider(provider: Optional[str]) -> Optional[str]:
        if provider is None:
            return None

        if provider in _SUPPORTED_LLM_PROVIDERS:
            return provider

        supported_values = ", ".join(sorted(_SUPPORTED_LLM_PROVIDERS))
        raise ValueError(
            f"Unsupported LLM provider {provider!r}. Supported values: {supported_values}"
        )

    @staticmethod
    def create_llm(
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.0,
        reasoning_effort: str = "low",
        timeout: float = 90.0,
    ) -> BaseChatModel:
        """
        Create a chat model instance for the provider specified by LLM_PROVIDER.


        Args:
            model: The model or deployment name to use (default: "gemini-3.1-flash-lite-preview")
            temperature: The temperature setting for the model (default: 0.0)
            reasoning_effort: The reasoning effort level - "low", "minimal", etc. (default: "low")
            timeout: The client-side request timeout in seconds (default: 90)

        Returns:
            A configured chat model instance

        Environment Variables:
            Common provider selection:
                - LLM_PROVIDER: Required provider selector
                  Supported values: gemini_api, gcp_vertexai, openai_api, azure_openai

            For Gemini Developer API:
                - GEMINI_API_KEY: Gemini API key
                - LLM_SEED: Optional seed

            For Gemini via Vertex AI:
                - GOOGLE_CLOUD_PROJECT: Google Cloud project ID
                - GOOGLE_CLOUD_LOCATION: Vertex AI location (optional, defaults to "global")
                - GOOGLE_APPLICATION_CREDENTIALS: Optional path to ADC/service account JSON

            For standard OpenAI:
                - OPENAI_API_KEY: Your OpenAI API key

            For Azure OpenAI:
                - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint (triggers Azure mode)
                - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
                - AZURE_OPENAI_DEPLOYMENT: Deployment name (optional, defaults to model parameter)
                - AZURE_OPENAI_API_VERSION: API version (optional, defaults to "2024-10-21")
            Other behavior:
                - timeout: client-side request timeout in seconds (default: 60)
                - max_retries: SDK-level retries are disabled (0); all retries handled by call_with_retry
        """
        resolved_model = model or DEFAULT_LLM_MODEL
        provider = LLMFactory._normalize_provider(os.getenv("LLM_PROVIDER"))
        if provider is None:
            supported_values = ", ".join(sorted(_SUPPORTED_LLM_PROVIDERS))
            raise ValueError(
                f"LLM_PROVIDER must be set. Supported values: {supported_values}"
            )

        try:
            if provider in {"gemini_api", "gcp_vertexai"}:
                from langchain_google_genai import ChatGoogleGenerativeAI

                gemini_kwargs = {
                    "model": resolved_model,
                    "temperature": temperature,
                    "timeout": timeout,
                    "max_retries": 0,  # Disable SDK retries; handled by call_with_retry
                }

                if provider == "gcp_vertexai":
                    project = os.getenv("GOOGLE_CLOUD_PROJECT")
                    if not project:
                        raise ValueError("GOOGLE_CLOUD_PROJECT must be set when using Vertex AI")

                    gemini_kwargs["vertexai"] = True
                    gemini_kwargs["project"] = project
                    gemini_kwargs["location"] = os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
                else:
                    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                    if api_key:
                        gemini_kwargs["api_key"] = api_key

                seed = LLMFactory._get_optional_int_env("LLM_SEED")
                if seed is not None:
                    gemini_kwargs["seed"] = seed

                if reasoning_effort in _GEMINI_THINKING_LEVELS:
                    gemini_kwargs["thinking_level"] = reasoning_effort

                return ChatGoogleGenerativeAI(**gemini_kwargs)

            if provider == "azure_openai":
                return AzureChatOpenAI(
                    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", resolved_model),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                    max_retries=0,  # Disable SDK retries; handled by call_with_retry
                )

            if provider == "openai_api":
                return ChatOpenAI(
                    model=resolved_model,
                    temperature=temperature,
                    reasoning={"effort": reasoning_effort},
                    output_version="responses/v1",
                    timeout=timeout,
                    max_retries=0,  # Disable SDK retries; handled by call_with_retry
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize LLM provider {provider!r}: {exc}"
            ) from exc

        raise ValueError(f"Unsupported LLM provider: {provider}")
