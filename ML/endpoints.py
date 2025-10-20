import os
from typing import List, Optional, Union

import requests


MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"


class MistralAPIError(Exception):
    pass


def _get_mistral_api_key() -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise MistralAPIError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Set it to your Mistral API key."
        )
    return api_key


def mistral_medium_2509_completion(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    safe_prompt: bool = False,
    timeout: float = 30.0,
) -> str:
    """
    Call Mistral chat completions for model "mistral-medium-2509".

    Returns the assistant message content as a string.
    """
    api_key = _get_mistral_api_key()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "mistral-medium-2509",
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


def mistral_embed(
    inputs: Union[str, List[str]],
    model: str = "mistral-embed",
    timeout: float = 30.0,
) -> List[List[float]]:
    """
    Get embeddings from Mistral for a single string or a list of strings.

    Returns a list of embeddings (one per input). For a single string input,
    the returned list will have length 1.
    """
    api_key = _get_mistral_api_key()

    input_payload: Union[str, List[str]]
    if isinstance(inputs, str):
        input_payload = inputs
    else:
        input_payload = list(inputs)

    payload = {
        "model": model,
        "input": input_payload,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{MISTRAL_API_BASE_URL}/embeddings"
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise MistralAPIError(f"Network error calling Mistral API: {exc}") from exc

    if response.status_code >= 400:
        raise MistralAPIError(
            f"Mistral API error {response.status_code}: {response.text}"
        )

    data = response.json()
    items = data.get("data") or []
    if not items:
        raise MistralAPIError("Mistral API returned no embeddings data.")

    embeddings: List[List[float]] = []
    for item in items:
        emb = item.get("embedding")
        if emb is None:
            raise MistralAPIError("Mistral API item missing 'embedding'.")
        embeddings.append(emb)

    return embeddings


