from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from parsing import PathLike, extract_docx_text, extract_pdf_text, extract_txt_text

ExtractorFn = Callable[[PathLike], str]

_EXTRACTORS: Dict[str, ExtractorFn] = {
    ".pdf": extract_pdf_text,
    ".txt": extract_txt_text,
    ".docx": extract_docx_text,
}


def extract_text(file_path: PathLike) -> str:
    """
    Extract textual content from a file, delegating to a format-specific parser.

    Parameters
    ----------
    file_path: PathLike
        Path to the document. The file extension determines the parser that will be
        used. Currently, only PDF files are supported.

    Returns
    -------
    str
        Extracted text content.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If no extractor is registered for the file's extension.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    extractor = _EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        supported = ", ".join(sorted(_EXTRACTORS))
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. "
            f"Supported formats: {supported or 'none'}."
        )

    return extractor(path)


__all__ = ["extract_text"]

