from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import app as api_app

DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
INDEX_FILE = DIST_DIR / "index.html"

if not INDEX_FILE.is_file():
    raise RuntimeError(f"Production frontend not found: {INDEX_FILE}")

# Reuse the existing API and its original routes.
app = api_app

# Serve the built frontend assets without changing API routes.
assets_dir = DIST_DIR / "assets"
if assets_dir.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=assets_dir),
        name="frontend-assets",
    )

@app.get("/", include_in_schema=False)
def frontend_home():
    return FileResponse(INDEX_FILE)

@app.get("/{path:path}", include_in_schema=False)
def frontend_fallback(path: str):
    candidate = (DIST_DIR / path).resolve()

    if candidate.is_relative_to(DIST_DIR.resolve()) and candidate.is_file():
        return FileResponse(candidate)

    return FileResponse(INDEX_FILE)
