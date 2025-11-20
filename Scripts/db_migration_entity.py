from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from loguru import logger

from data.db_interaction import upsert_entity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_1"


@dataclass(frozen=True)
class EntityRecord:
    """
    Lightweight representation of a CV entity.

    Only two values are surfaced as requested:
    - name: the filename without its extension
    - tag: a unique slug that can serve as the canonical identifier
    """

    name: str
    tag: str

    def as_dict(self) -> Dict[str, str]:
        return {"name": self.name, "tag": self.tag}


def _iter_dataset_files(dataset_dir: Path) -> Iterable[Path]:
    """
    Yield files within the dataset directory, ignoring sub-directories.
    """

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for entry in sorted(dataset_dir.iterdir()):
        if entry.is_file():
            yield entry


def _slugify_name(raw_name: str) -> str:
    """
    Create a filesystem/database friendly slug that can serve as the base tag.
    """

    normalized = (
        unicodedata.normalize("NFKD", raw_name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = []
    previous_dash = False
    for char in normalized:
        if char.isalnum():
            slug.append(char)
            previous_dash = False
        else:
            if not previous_dash:
                slug.append("-")
            previous_dash = True

    cleaned = "".join(slug).strip("-")
    return cleaned or "cv"


def _ensure_unique_tag(base_tag: str, used_tags: Dict[str, int]) -> str:
    """
    Guarantee tag uniqueness by appending a numeric suffix when needed.
    """

    count = used_tags.get(base_tag, 0)
    if count == 0:
        used_tags[base_tag] = 1
        return base_tag

    while True:
        candidate = f"{base_tag}-{count + 1}"
        if candidate not in used_tags:
            used_tags[candidate] = 1
            used_tags[base_tag] = count + 1
            return candidate
        count += 1


def build_entity_records(dataset_dir: Path) -> List[EntityRecord]:
    """
    Parse dataset filenames into EntityRecord objects.
    """

    used_tags: Dict[str, int] = {}
    records: List[EntityRecord] = []

    for file_path in _iter_dataset_files(dataset_dir):
        name = file_path.stem.strip()
        base_tag = _slugify_name(name)
        tag = _ensure_unique_tag(base_tag, used_tags)
        records.append(EntityRecord(name=name, tag=tag))

    return records


def sync_entities(records: Iterable[EntityRecord]) -> None:
    """
    Write entity records to the database via the upsert helper.
    """

    for record in records:
        upsert_entity(name=record.name, tag=record.tag)
        logger.info(
            "Upserted entity | name={name} tag={tag}",
            name=record.name,
            tag=record.tag,
        )


def main(output_json: bool = False) -> None:
    records = build_entity_records(DATASET_DIR)

    if output_json:
        payload = [record.as_dict() for record in records]
        print(json.dumps(payload, indent=2))
    else:
        for record in records:
            logger.info(
                "Parsed entity | name={name} tag={tag}",
                name=record.name,
                tag=record.tag,
            )

    sync_entities(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse dataset filenames into entity records."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the resulting dictionaries as JSON.",
    )

    args = parser.parse_args()
    main(output_json=args.json)

