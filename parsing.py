from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union
from zipfile import ZipFile
from xml.etree.ElementTree import XML, ParseError

from PyPDF2 import PdfReader


PathLike = Union[str, Path]


def _coalesce_text(chunks: Iterable[str]) -> str:
    """
    Join text chunks, skipping empty fragments while preserving line breaks.
    """
    normalized_chunks = []
    for chunk in chunks:
        if chunk is None:
            continue
        stripped = chunk.strip()
        if stripped:
            normalized_chunks.append(stripped)
    return "\n".join(normalized_chunks)


def extract_pdf_text(pdf_path: PathLike) -> str:
    """
    Extract text from a PDF file and return it as a single string.

    Parameters
    ----------
    pdf_path: PathLike
        Path to the PDF CV file.

    Returns
    -------
    str
        All textual content extracted from the PDF. Pages are separated by
        newline characters.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    with path.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)
        page_texts = [page.extract_text() for page in reader.pages]
        return _coalesce_text(page_texts)


def extract_txt_text(txt_path: PathLike, encoding: str = "utf-8") -> str:
    """
    Extract text from a plain text file.

    Parameters
    ----------
    txt_path: PathLike
        Path to the text file.
    encoding: str
        Encoding to use for decoding the file. Defaults to UTF-8.
    """
    path = Path(txt_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    with path.open("r", encoding=encoding, errors="ignore") as txt_file:
        return txt_file.read()


def extract_docx_text(docx_path: PathLike) -> str:
    """
    Extract text content from a DOCX document.

    Parameters
    ----------
    docx_path: PathLike
        Path to the DOCX file.
    """
    path = Path(docx_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []

    try:
        with ZipFile(path) as archive:
            with archive.open("word/document.xml") as document_xml:
                xml_content = document_xml.read()
    except KeyError as exc:
        raise ValueError(f"Invalid DOCX structure: missing document.xml in {path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read DOCX file: {path}") from exc

    try:
        root = XML(xml_content)
    except ParseError as exc:
        raise ValueError(f"Failed to parse DOCX XML content: {path}") from exc

    for paragraph in root.iter(f"{namespace}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
        paragraph_text = "".join(runs).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return _coalesce_text(paragraphs)


__all__ = ["extract_pdf_text", "extract_txt_text", "extract_docx_text"]

