import json
import os
import re
from typing import Optional

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
        raise MistralAPIError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Set it to your Mistral API key."
        )
    return api_key


class CandidateProfile(BaseModel):
    """Structured representation of extracted resume information."""

    model_config = ConfigDict(populate_by_name=True)

    education: list[dict] | str = Field(..., alias="Education", description="Summary of the candidate's education history.")
    experience: list[dict] | str = Field(..., alias="Experience", description="Summary of the candidate's professional experience.")
    location: str = Field(..., alias="Location", description="Primary location associated with the candidate.")
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
            raise ValueError("Unable to normalize value into string")
        return normalized


    @classmethod
    def _coerce_skills(cls, value: object) -> list[str]:
        skills = cls._normalize_skills_field(value)
        if not isinstance(skills, list):
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
    if safe_prompt:
        payload["safe_prompt"] = True

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{MISTRAL_API_BASE_URL}/chat/completions"
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise MistralAPIError(f"Network error calling Mistral API: {exc}") from exc

    if response.status_code >= 400:
        raise MistralAPIError(
            f"Mistral API error {response.status_code}: {response.text}"
        )

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise MistralAPIError("Mistral API returned no choices in response.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise MistralAPIError("Mistral API response missing message content.")
    return content


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object found in a text blob."""
    candidate = text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        return candidate

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        raise MistralAPIError("Model response does not contain a JSON object.")
    return match.group(0)


def generate_candidate_profile(
    extracted_text: str,
    *,
    model: str = "mistral-small-latest",
    temperature: float = 0.0,
    max_tokens: Optional[int] =None,
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
    system_prompt = (
        "You are an assistant that extracts resume information and responds strictly in JSON. "
        "Return keys exactly as Education, Experience, Location, Skills. "
        "Skills must be a JSON array of short strings."
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
        return CandidateProfile.model_validate_json(json_blob)
    except ValidationError as exc:
        raise MistralAPIError(f"Unable to parse model response into CandidateProfile: {exc}") from exc

