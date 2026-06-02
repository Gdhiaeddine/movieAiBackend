from __future__ import annotations

import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent


def _resolve_model_path() -> Path:
    env = os.getenv("MODEL_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    candidates = [
        BACKEND_DIR / "movies_and_tfidf.pkl",
        PROJECT_DIR / "movies_and_tfidf.pkl",
        WORKSPACE_DIR / "movies_and_tfidf.pkl",
    ]
    for c in candidates:
        if c.is_file():
            return c

    return PROJECT_DIR / "movies_and_tfidf.pkl"


MODEL_PATH: Path = _resolve_model_path()

CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]
CORS_ALLOW_ALL: bool = "*" in CORS_ORIGINS

DEFAULT_LIMIT: int = int(os.getenv("DEFAULT_LIMIT", "10"))
MAX_LIMIT: int = int(os.getenv("MAX_LIMIT", "50"))

ENABLE_POSTER_FETCH: bool = os.getenv("ENABLE_POSTER_FETCH", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
POSTER_CACHE_PATH: Path = Path(
    os.getenv("POSTER_CACHE_PATH", str(BACKEND_DIR / "poster_cache.json"))
).expanduser()
POSTER_FETCH_TIMEOUT: float = float(os.getenv("POSTER_FETCH_TIMEOUT", "5.0"))
POSTER_FETCH_CONCURRENCY: int = int(os.getenv("POSTER_FETCH_CONCURRENCY", "8"))
