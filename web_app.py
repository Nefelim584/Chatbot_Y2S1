import os
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rank_cvs import rank_cvs_for_offer_text  # uses your existing ranking logic


# ---------------- CONFIG ----------------
TOP_K = 10

# Prefer env var for absolute path (recommended),
# fallback to project-relative path next to this file.
DEFAULT_DIR = Path(__file__).resolve().parent / "data" / "CV" / "CV_Brut"
CV_FILES_DIR = Path(os.getenv("CV_FILES_DIR", str(DEFAULT_DIR))).resolve()
# --------------------------------------


app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


class RankRequest(BaseModel):
    message: str


def _safe_resolve_cv_path(file_name: str) -> Path:
    """
    Prevent path traversal. We allow only a plain filename (no folders).
    If not found directly, we optionally try to find it recursively under CV_FILES_DIR.
    """
    if not file_name or file_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid file name.")

    # Reject any attempt to include directories
    if "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name (no folders allowed).")

    base = CV_FILES_DIR
    direct = (base / file_name).resolve()

    # Must remain inside base
    if base not in direct.parents and direct != base:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    if direct.exists() and direct.is_file():
        return direct

    # Fallback: search recursively (helps if CVs are stored in subfolders)
    matches = list(base.rglob(file_name))
    matches = [m for m in matches if m.is_file()]

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=f"Multiple files with same name found under CV_FILES_DIR: {file_name}. "
                   f"Please rename files to be unique."
        )

    raise HTTPException(status_code=404, detail=f"File not found: {file_name}")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "top_k": TOP_K,
            "cv_dir": str(CV_FILES_DIR),
        },
    )


@app.post("/rank")
def rank(req: RankRequest):
    offer_text = (req.message or "").strip()
    if not offer_text:
        raise HTTPException(status_code=400, detail="Empty job offer text.")

    data = rank_cvs_for_offer_text(offer_text, top_k=TOP_K)

    # Add open & download links (URL-encoded)
    for r in data.get("results", []):
        fn = r["file_name"]
        enc = quote(fn, safe="")  # encode spaces + special chars safely
        r["open_url"] = f"/cv/{enc}"
        r["download_url"] = f"/cv/{enc}?download=1"

    return data


@app.get("/cv/{file_name}")
def get_cv(file_name: str, download: int = 0):
    # Important: FastAPI already URL-decodes path params
    path = _safe_resolve_cv_path(file_name)

    suffix = path.suffix.lower()
    media_type = "application/pdf" if suffix == ".pdf" else "application/octet-stream"

    disposition = "attachment" if download else "inline"
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        headers={"Content-Disposition": f'{disposition}; filename="{path.name}"'},
    )
