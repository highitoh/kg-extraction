"""Utility functions for prompt template handling."""


def fill_prompt_template(template: str, **kwargs) -> str:
    """
    Replace <<PLACEHOLDER>> markers with provided values.

    Args:
        template: The prompt template string containing <<PLACEHOLDER>> markers
        **kwargs: Keyword arguments where keys are lowercase placeholder names
                 and values are the strings to substitute

    Returns:
        The template with all placeholders replaced

    Example:
        >>> template = "Hello <<NAME>>, you are <<AGE>> years old."
        >>> fill_prompt_template(template, name="Alice", age=30)
        'Hello Alice, you are 30 years old.'
    """
    result = template
    for key, value in kwargs.items():
        placeholder = f"<<{key.upper()}>>"
        result = result.replace(placeholder, str(value))
    return result
