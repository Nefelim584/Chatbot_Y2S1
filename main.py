from __future__ import annotations
from pathlib import Path
from pipelines.Extraction import extract_text
from ML.completion import MistralAPIError, generate_candidate_profile
from loguru import logger


def _pick_first_pdf(db_directory: Path) -> Path:
    """
    Locate the first PDF file in the provided directory.
    """
    for candidate in db_directory.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".pdf":
            return candidate
    raise FileNotFoundError(f"No PDF files found in directory: {db_directory}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    db_directory = project_root / "data/dataset_1"

    if not db_directory.exists() or not db_directory.is_dir():
        raise FileNotFoundError(f"Database directory not found: {db_directory}")

    pdf_path = _pick_first_pdf(db_directory)
    extracted_text = extract_text(pdf_path)

    try:
        profile = generate_candidate_profile(extracted_text)
    except MistralAPIError as exc:
        raise RuntimeError(f"Failed to generate candidate profile: {exc}") from exc

    logger.info(f"Candidate profile extracted from '{pdf_path.name}':")
    logger.info(profile.model_dump_json(by_alias=True, indent=2))


if __name__ == "__main__":
    main()


