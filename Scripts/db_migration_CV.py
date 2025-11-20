from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from loguru import logger
from sqlalchemy import select

from ML.completion import MistralAPIError, generate_candidate_profile
from data.db_interaction import session_scope, upsert_cv_text
from data.data_models import Entity
from pipelines.Extraction import extract_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset_1"


@dataclass(frozen=True)
class CvEntity:
    """Lightweight representation of an entity persisted in the database."""

    name: str
    tag: str


def fetch_entities(*, tags: Sequence[str] | None = None) -> List[CvEntity]:
    """
    Retrieve all entity records (name + tag) from the database.

    Parameters
    ----------
    tags:
        Optional sequence of tag filters. When provided, only entities whose tags
        appear in this list will be returned.
    """

    with session_scope() as session:
        stmt = select(Entity)
        if tags:
            stmt = stmt.where(Entity.tag.in_(tags))
        rows = session.execute(stmt).scalars().all()

        return [CvEntity(name=row.name.strip(), tag=row.tag.strip()) for row in rows]


def build_dataset_index(dataset_dir: Path) -> Dict[str, Path]:
    """
    Build a lookup table of {filename_stem: file_path} for the dataset directory.
    """

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    index: Dict[str, Path] = {}
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_file():
            continue
        stem = entry.stem.strip()
        if stem in index:
            logger.warning(
                "Duplicate CV filename stem detected; later file will be ignored",
                stem=stem,
                kept=index[stem],
                ignored=entry,
            )
            continue
        index[stem] = entry

    return index


def _stringify_section(value: object) -> str:
    """
    Normalize CandidateProfile sections into a flat string representation.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            normalized = _stringify_section(item)
            if normalized:
                parts.append(f"{key}: {normalized}")
        return "; ".join(parts)
    if isinstance(value, list):
        parts = []
        for entry in value:
            normalized = _stringify_section(entry)
            if normalized:
                parts.append(normalized)
        return " | ".join(parts)
    return str(value).strip()


def process_entity(
    entity: CvEntity,
    dataset_index: Dict[str, Path],
    *,
    dry_run: bool = False,
) -> bool:
    """
    Process a single entity by extracting its CV text, generating a profile, and persisting it.
    """

    cv_path = dataset_index.get(entity.name)
    if cv_path is None:
        logger.error("No CV file found for entity; skipping", name=entity.name, tag=entity.tag)
        return False

    try:
        extracted_text = extract_text(cv_path)
    except Exception as exc:  # pragma: no cover - extractor errors are runtime specific
        logger.exception(
            "Failed to extract text from CV; skipping",
            exception=exc,
            path=cv_path,
            tag=entity.tag,
        )
        return False

    try:
        profile = generate_candidate_profile(extracted_text)
    except MistralAPIError as exc:
        logger.exception(
            "Candidate profile generation failed; skipping",
            exception=exc,
            tag=entity.tag,
            path=cv_path,
        )
        return False

    payload = {
        "skills": ", ".join(profile.skills),
        "experience": _stringify_section(profile.experience),
        "education": _stringify_section(profile.education),
        "location": profile.location.strip(),
    }

    if dry_run:
        logger.info(
            "Dry-run: generated payload for entity",
            tag=entity.tag,
            name=entity.name,
            payload=payload,
        )
        return True

    upsert_cv_text(tag=entity.tag, **payload)
    logger.info(
        "Persisted candidate profile",
        tag=entity.tag,
        name=entity.name,
        path=cv_path.name,
    )
    return True


def iter_entities_to_process(
    entities: Iterable[CvEntity],
    *,
    limit: int | None = None,
) -> Iterable[CvEntity]:
    """
    Yield entities up to the optional limit in the order they were provided.
    """

    if limit is None or limit <= 0:
        yield from entities
        return

    count = 0
    for entity in entities:
        if count >= limit:
            break
        yield entity
        count += 1


def main(  # pragma: no cover - script-style entry point
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    tags: Sequence[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    logger.info("Starting CV processing", dataset_dir=str(dataset_dir), tag_filters=tags, limit=limit)

    dataset_dir = dataset_dir.expanduser().resolve()
    dataset_index = build_dataset_index(dataset_dir)

    entities = fetch_entities(tags=tags)
    if not entities:
        logger.warning("No entities found matching the provided criteria")
        return

    processed = 0
    succeeded = 0

    for entity in iter_entities_to_process(entities, limit=limit):
        processed += 1
        if process_entity(entity, dataset_index, dry_run=dry_run):
            succeeded += 1

    logger.info(
        "Completed CV processing run",
        processed=processed,
        succeeded=succeeded,
        failed=processed - succeeded,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate CvText entries from stored CV files.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the CV files (default: data/dataset_1).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Optional list of tags to process. When omitted, all entities are processed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for the number of CVs to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process CVs but skip database writes.",
    )

    args = parser.parse_args()
    main(
        dataset_dir=args.dataset_dir,
        tags=args.tags,
        limit=args.limit,
        dry_run=args.dry_run,
    )

