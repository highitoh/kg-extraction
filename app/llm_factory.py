"""Factory class for creating ChatOpenAI instances with consistent configuration."""

import os
from typing import Union

from langchain_openai import ChatOpenAI, AzureChatOpenAI


class LLMFactory:
    """Factory class for creating LLM instances with standardized configuration."""

    @staticmethod
    def create_llm(
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        reasoning_effort: str = "low",
    ) -> Union[ChatOpenAI, AzureChatOpenAI]:
        """
        Create a ChatOpenAI or AzureChatOpenAI instance based on environment variables.

        Automatically detects whether to use Azure OpenAI or standard OpenAI based on
        the presence of AZURE_OPENAI_ENDPOINT environment variable.

        Args:
            model: The model name or deployment name to use (default: "gpt-5-mini")
            temperature: The temperature setting for the model (default: 0.0)
            reasoning_effort: The reasoning effort level - "low", "minimal", etc. (default: "low")

        Returns:
            A configured ChatOpenAI or AzureChatOpenAI instance

        Environment Variables:
            For standard OpenAI:
                - OPENAI_API_KEY: Your OpenAI API key

            For Azure OpenAI:
                - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint (triggers Azure mode)
                - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
                - AZURE_OPENAI_DEPLOYMENT: Deployment name (optional, defaults to model parameter)
                - AZURE_OPENAI_API_VERSION: API version (optional, defaults to "2024-10-21")
        """
        # Check if Azure OpenAI configuration exists
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if azure_endpoint:
            # Use Azure OpenAI Service
            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", model),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        else:
            # Use standard OpenAI API
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                reasoning={"effort": reasoning_effort},
                output_version="responses/v1",
            )
