from __future__ import annotations

from typing import Final

MISTRAL_API_BASE_URL: Final[str] = "https://api.mistral.ai/v1"

# Chat/completion model tags currently used by the application.
COMPLETION_MODEL_TAGS: Final[dict[str, str]] = {
    "default": "mistral-small-latest",
    "resume_extraction": "mistral-large-latest",
    "general_medium": "mistral-medium-2509",
}

# Embedding model tags (kept for completeness if embeddings are needed later).
EMBEDDING_MODEL_TAGS: Final[dict[str, str]] = {
    "default": "mistral-embed",
}

__all__ = ["MISTRAL_API_BASE_URL", "COMPLETION_MODEL_TAGS", "EMBEDDING_MODEL_TAGS"]
