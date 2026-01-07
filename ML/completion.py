import json
import os
import re
from typing import Optional
from loguru import logger
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"


class MistralAPIError(Exception):
    """Custom exception for Mistral API errors."""
    pass


def _get_mistral_api_key() -> str:
    """Get Mistral API key from environment variable."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY environment variable is not set")
        raise MistralAPIError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Set it to your Mistral API key."
        )
    logger.debug("Mistral API key retrieved successfully")
    return api_key


class CandidateProfile(BaseModel):
    """Structured representation of extracted resume information."""

    model_config = ConfigDict(populate_by_name=True)

    education: list[dict] | str = Field(..., alias="Education",
                                        description="Summary of the candidate's education history.")
    experience: list[dict] | str = Field(..., alias="Experience",
                                         description="Summary of the candidate's professional experience.")
    location: str = Field(..., alias="Location", description="City associated with the candidate.")
    skills: list[str] = Field(
        ..., alias="Skills", description="Key skills separated into a list of short phrases."
    )

    @staticmethod
    def _normalize_text_field(value: object) -> str:
        """
        Normalize flexible resume sections into a single string.

        Accepts raw strings, dicts, or lists of strings/dicts and produces a readable summary.
        """
        if value is None:
            logger.debug("Normalizing None value to empty string")
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                if item is None:
                    continue
                part = str(item).strip()
                if not part:
                    continue
                parts.append(f"{key}: {part}")
            return ", ".join(parts)
        if isinstance(value, list):
            parts = []
            for entry in value:
                normalized = CandidateProfile._normalize_text_field(entry)
                if normalized:
                    parts.append(normalized)
            return " | ".join(parts)
        return str(value).strip()

    @staticmethod
    def _normalize_skills_field(value: object) -> list[str]:
        """Normalize skills as a list of short strings."""
        if value is None:
            logger.debug("Normalizing None skills to empty list")
            return []
        if isinstance(value, list):
            skills: list[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    skill = item.strip()
                else:
                    skill = str(item).strip()
                if skill:
                    skills.append(skill)
            return skills
        if isinstance(value, str):
            # Split on commas or semicolons for forgiving parsing
            raw_items = re.split(r"[;,]", value)
            return [item.strip() for item in raw_items if item.strip()]
        return [str(value).strip()]

    @classmethod
    def _coerce_text_fields(cls, value: object) -> str:
        normalized = cls._normalize_text_field(value)
        if not isinstance(normalized, str):
            logger.error(f"Failed to normalize value into string: {type(value)}")
            raise ValueError("Unable to normalize value into string")
        return normalized

    @classmethod
    def _coerce_skills(cls, value: object) -> list[str]:
        skills = cls._normalize_skills_field(value)
        if not isinstance(skills, list):
            logger.error(f"Failed to normalize skills into list: {type(value)}")
            raise ValueError("Unable to normalize skills into list")
        return skills


def completion(
        model: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        safe_prompt: bool = False,
        timeout: float = 30.0,
) -> str:
    """
    Call Mistral chat completions API with the specified model.

    Args:
        model: The model tag to use (e.g., "mistral-medium", "mistral-large-latest")
        system_prompt: The system prompt string
        prompt: The user prompt string
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum number of tokens to generate (optional)
        safe_prompt: Whether to use safe prompt filtering (default: False)
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        The assistant message content as a string.

    Raises:
        MistralAPIError: If the API call fails or returns an error.
    """
    logger.info(f"Calling Mistral API with model: {model}, temperature: {temperature}")
    api_key = _get_mistral_api_key()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
        logger.debug(f"Max tokens set to: {max_tokens}")
    if safe_prompt:
        payload["safe_prompt"] = True
        logger.debug("Safe prompt filtering enabled")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{MISTRAL_API_BASE_URL}/chat/completions"
    try:
        logger.debug(f"Sending request to: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        logger.error(f"Network error calling Mistral API: {exc}")
        raise MistralAPIError(f"Network error calling Mistral API: {exc}") from None

    if response.status_code >= 400:
        logger.error(f"Mistral API error {response.status_code}: {response.text}")
        raise MistralAPIError(
            f"Mistral API error {response.status_code}: {response.text}"
        )

    logger.info(f"Mistral API call successful with status: {response.status_code}")
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        logger.error("Mistral API returned no choices in response")
        raise MistralAPIError("Mistral API returned no choices in response.") from None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        logger.error("Mistral API response missing message content")
        raise MistralAPIError("Mistral API response missing message content.") from None

    logger.debug(f"Received response content length: {len(content)}")
    return content


def chat_completion(
        messages: list[dict[str, str]],
        *,
        model: str = "mistral-small-latest",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        safe_prompt: bool = False,
        timeout: float = 30.0,
) -> str:
    """
    Call Mistral chat completions API with a conversation history.

    This function is designed for multi-turn conversations where message history
    needs to be maintained. It accepts a list of messages with roles (system, user, assistant).

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
                 Roles should be 'system', 'user', or 'assistant'.
                 Example: [
                     {"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": "Hello!"},
                     {"role": "assistant", "content": "Hi there!"},
                     {"role": "user", "content": "How are you?"}
                 ]
        model: The model tag to use (default: "mistral-small-latest")
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum number of tokens to generate (optional)
        safe_prompt: Whether to use safe prompt filtering (default: False)
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        The assistant message content as a string.

    Raises:
        MistralAPIError: If the API call fails or returns an error.
    """
    logger.info(f"Calling Mistral chat completion with model: {model}, {len(messages)} messages")
    
    if not isinstance(messages, list) or not messages:
        raise MistralAPIError("Messages must be a non-empty list of message dictionaries.")
    
    # Validate message format
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise MistralAPIError(f"Message at index {idx} must be a dictionary.")
        if "role" not in msg or "content" not in msg:
            raise MistralAPIError(f"Message at index {idx} must have 'role' and 'content' keys.")
        if msg["role"] not in ["system", "user", "assistant"]:
            raise MistralAPIError(
                f"Message at index {idx} has invalid role '{msg['role']}'. "
                "Must be 'system', 'user', or 'assistant'."
            )
    
    api_key = _get_mistral_api_key()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
        logger.debug(f"Max tokens set to: {max_tokens}")
    if safe_prompt:
        payload["safe_prompt"] = True
        logger.debug("Safe prompt filtering enabled")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{MISTRAL_API_BASE_URL}/chat/completions"
    try:
        logger.debug(f"Sending chat completion request to: {url}")
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        logger.error(f"Network error calling Mistral API: {exc}")
        raise MistralAPIError(f"Network error calling Mistral API: {exc}") from None

    if response.status_code >= 400:
        logger.error(f"Mistral API error {response.status_code}: {response.text}")
        raise MistralAPIError(
            f"Mistral API error {response.status_code}: {response.text}"
        )

    logger.info(f"Mistral API call successful with status: {response.status_code}")
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        logger.error("Mistral API returned no choices in response")
        raise MistralAPIError("Mistral API returned no choices in response.") from None

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        logger.error("Mistral API response missing message content")
        raise MistralAPIError("Mistral API response missing message content.") from None

    logger.debug(f"Received response content length: {len(content)}")
    return content


def embed_words(
        words: list[str],
        *,
        model: str = "mistral-embed",
        timeout: float = 30.0,
) -> list[list[float]]:
    """
    Generate embedding vectors for a list of words using the Mistral embeddings API.

    Args:
        words: List of words or short phrases to embed.
        model: Embedding model identifier (default: "mistral-embed").
        timeout: Request timeout in seconds (default: 30.0).

    Returns:
        List of embedding vectors in the same order as the input words.

    Raises:
        MistralAPIError: If the API call fails or the response payload is invalid.
    """
    if not isinstance(words, list):
        raise MistralAPIError("`words` must be provided as a list of strings.")
    if not words:
        logger.info("No words provided for embedding; returning empty list.")
        return []

    normalized_words: list[str] = []
    for index, word in enumerate(words):
        if not isinstance(word, str):
            raise MistralAPIError(f"All items must be strings. Invalid item at index {index}.")
        sanitized = word.strip()
        if not sanitized:
            raise MistralAPIError(f"Word at index {index} is empty after stripping.")
        normalized_words.append(sanitized)

    logger.info(f"Requesting embeddings for {len(normalized_words)} words using model: {model}")
    api_key = _get_mistral_api_key()
    payload = {
        "model": model,
        "inputs": normalized_words,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{MISTRAL_API_BASE_URL}/embeddings"

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        logger.error(f"Network error calling Mistral embeddings API: {exc}")
        raise MistralAPIError(f"Network error calling Mistral embeddings API: {exc}") from None

    if response.status_code >= 400:
        logger.error(f"Mistral embeddings API error {response.status_code}: {response.text}")
        raise MistralAPIError(
            f"Mistral embeddings API error {response.status_code}: {response.text}"
        )

    data = response.json()
    embeddings = data.get("data")
    if not isinstance(embeddings, list):
        logger.error("Embeddings API response missing `data` list")
        raise MistralAPIError("Embeddings API response missing `data` list.")
    if len(embeddings) != len(normalized_words):
        logger.error("Embeddings API returned a mismatched number of vectors")
        raise MistralAPIError("Embeddings API returned a mismatched number of vectors.")

    vectors: list[list[float]] = []
    for idx, item in enumerate(embeddings):
        vector = item.get("embedding")
        if not isinstance(vector, list):
            logger.error(f"Embedding at index {idx} missing or invalid")
            raise MistralAPIError("Embeddings API response contains invalid embedding values.")
        vectors.append(vector)

    logger.info("Successfully retrieved embeddings from Mistral API")
    return vectors


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object found in a text blob."""
    logger.debug("Extracting JSON object from response text")
    candidate = text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        logger.debug("Response is already a clean JSON object")
        return candidate

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        logger.error("Model response does not contain a JSON object")
        raise MistralAPIError("Model response does not contain a JSON object.") from None

    logger.debug("Successfully extracted JSON object from response")
    return match.group(0)


def generate_candidate_profile(
        extracted_text: str,
        *,
        model: str = "mistral-large-latest",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
) -> CandidateProfile:
    """
    Generate a structured candidate profile from extracted resume text.

    Args:
        extracted_text: Raw textual content extracted from the document.
        model: Mistral model identifier to use for completion.
        temperature: Sampling temperature for generation.
        max_tokens: Optional token limit for the response.

    Returns:
        CandidateProfile populated from the model output.
    """
    logger.info(f"Generating candidate profile using model: {model}")
    logger.debug(f"Extracted text length: {len(extracted_text)} characters")

    system_prompt = (
        "You are an assistant that extracts resume information and responds strictly in JSON. "
        "Return keys exactly as Education, Experience, Location, Skills. "
        "Skills must be a JSON array of short strings. All skills mentioned in the resume should be included exactly in this section. "
        "Location should contain only the county and city names, separated by comma."
        "Education should have only degree names, institutions and graduation years."
        "Experience should have job titles, companies and durations."
    )
    user_prompt = (
        f"{extracted_text}"
    )

    response = completion(
        model=model,
        system_prompt=system_prompt,
        prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    json_blob = _extract_json_object(response)
    try:
        profile = CandidateProfile.model_validate_json(json_blob)
        logger.info("Successfully generated and validated candidate profile")
        return profile
    except ValidationError as exc:
        logger.error(f"Unable to parse model response into CandidateProfile: {exc}")
        raise MistralAPIError(f"Unable to parse model response into CandidateProfile: {exc}") from None
