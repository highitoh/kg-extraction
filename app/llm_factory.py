"""Factory class for creating ChatOpenAI instances with consistent configuration."""

from langchain_openai import ChatOpenAI


class LLMFactory:
    """Factory class for creating LLM instances with standardized configuration."""

    @staticmethod
    def create_llm(
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        reasoning_effort: str = "low",
    ) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance with standardized configuration.

        Args:
            model: The model name to use (default: "gpt-5-mini")
            temperature: The temperature setting for the model (default: 0.0)
            reasoning_effort: The reasoning effort level - "low", "minimal", etc. (default: "low")

        Returns:
            A configured ChatOpenAI instance
        """
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            reasoning={"effort": reasoning_effort},
            output_version="responses/v1",
        )
