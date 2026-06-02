from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    CORS_ALLOW_ALL,
    CORS_ORIGINS,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MODEL_DOWNLOAD_SHA256,
    MODEL_DOWNLOAD_URL,
    MODEL_PATH,
)
from .poster_resolver import (
    enrich_movie,
    enrich_recommendation_response,
    resolver as poster_resolver,
)
from .recommender import recommender
from .schemas import (
    HealthResponse,
    Movie,
    PingResponse,
    RecommendRequest,
    RecommendResponse,
    SearchResponse,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("api")

APP_VERSION = "1.0.0"
_STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — loading recommender model...")
    try:
        await _ensure_model_on_disk()
        recommender.load()
    except Exception as exc:
        log.exception("Failed to load model: %s", exc)
    poster_resolver.load()
    yield
    poster_resolver.save()
    log.info("Shutting down.")


async def _ensure_model_on_disk() -> None:
    """If MODEL_DOWNLOAD_URL is set and the pickle is missing, fetch it.

    Lets us deploy to hosts that don't ship the 42 MB .pkl in the repo
    (e.g. Render free tier, Fly.io). Verify SHA-256 if provided.
    """
    if MODEL_PATH.is_file():
        return
    if not MODEL_DOWNLOAD_URL:
        return
    log.info(
        "Model missing at %s — downloading from %s",
        MODEL_PATH, MODEL_DOWNLOAD_URL,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
    hasher = hashlib.sha256() if MODEL_DOWNLOAD_SHA256 else None
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", MODEL_DOWNLOAD_URL) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                written = 0
                with open(tmp, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        written += len(chunk)
                        if hasher is not None:
                            hasher.update(chunk)
                        if total:
                            pct = 100 * written / total
                            log.info(
                                "  %5.1f%%  %.2f / %.2f MB",
                                pct, written / 1e6, total / 1e6,
                            )
        if hasher is not None:
            digest = hasher.hexdigest()
            if digest.lower() != MODEL_DOWNLOAD_SHA256.lower():
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Model SHA-256 mismatch "
                    f"(expected {MODEL_DOWNLOAD_SHA256}, got {digest})"
                )
            log.info("Model SHA-256 verified: %s", digest)
        tmp.replace(MODEL_PATH)
        log.info("Model saved to %s (%d bytes)", MODEL_PATH, MODEL_PATH.stat().st_size)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


app = FastAPI(
    title="MovieAI Recommendation API",
    description=(
        "Content-based movie recommendation backend powered by a "
        "pre-fitted TF-IDF cosine-similarity model."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ALLOW_ALL else CORS_ORIGINS,
    allow_credentials=not CORS_ALLOW_ALL,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": "MovieAI Recommendation API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
        "ping": "/api/ping",
    }


@app.head("/", include_in_schema=False)
def root_head() -> Response:
    return Response(status_code=200)


@app.get("/api/ping", response_model=PingResponse, tags=["meta"])
def ping() -> PingResponse:
    """Cheap heartbeat endpoint. Use this for uptime monitors."""
    return PingResponse(pong=True, timestamp_ms=int(time.time() * 1000))


@app.head("/api/ping", include_in_schema=False)
def ping_head() -> Response:
    """HEAD support so UptimeRobot can probe with HEAD as well."""
    return Response(status_code=200)


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    rows, cols = recommender.tfidf_shape
    model_ok = recommender.loaded
    cache = poster_resolver.stats()
    overall = "ok" if model_ok else "loading"
    return HealthResponse(
        status=overall,
        model_loaded=model_ok,
        movies_count=recommender.movies_count,
        tfidf_shape=[rows, cols],
        model_path=str(recommender.model_path),
        version=APP_VERSION,
        uptime_seconds=round(time.time() - _STARTED_AT, 3),
        poster_cache=cache,
    )


@app.head("/api/health", include_in_schema=False)
def health_head() -> Response:
    return Response(status_code=200 if recommender.loaded else 503)


@app.get("/api/movies/search", response_model=SearchResponse, tags=["movies"])
def search_movies(
    q: str = Query(..., min_length=1, description="Title fragment to search"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> SearchResponse:
    _ensure_loaded()
    results = recommender.search(q, limit=limit)
    return SearchResponse(query=q, results=results)


@app.get("/api/movies/random", response_model=Movie, tags=["movies"])
async def random_movie() -> Movie:
    _ensure_loaded()
    movie = recommender.random_movie()
    await enrich_movie(movie)
    return movie


@app.get("/api/movies/{movie_id}", response_model=Movie, tags=["movies"])
async def get_movie(movie_id: int) -> Movie:
    _ensure_loaded()
    movie = recommender.get_movie(movie_id=movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail=f"Movie id={movie_id} not found")
    await enrich_movie(movie)
    return movie


@app.post("/api/recommend", response_model=RecommendResponse, tags=["recommend"])
async def recommend(req: RecommendRequest) -> RecommendResponse:
    _ensure_loaded()
    if req.title is None and req.movie_id is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'title' or 'movie_id' in the request body.",
        )
    try:
        source, recs = recommender.recommend(
            title=req.title,
            movie_id=req.movie_id,
            limit=req.limit,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await enrich_recommendation_response(source, recs)
    return RecommendResponse(source=source, recommendations=recs)


@app.get("/api/recommend", response_model=RecommendResponse, tags=["recommend"])
async def recommend_get(
    title: Optional[str] = Query(None),
    movie_id: Optional[int] = Query(None),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> RecommendResponse:
    return await recommend(
        RecommendRequest(title=title, movie_id=movie_id, limit=limit)
    )


@app.get(
    "/api/recommend/random",
    response_model=RecommendResponse,
    tags=["recommend"],
)
async def recommend_random(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> RecommendResponse:
    """Pick a random highly-rated movie and return its recommendations."""
    _ensure_loaded()
    source, recs = recommender.recommend_random(limit=limit)
    await enrich_recommendation_response(source, recs)
    return RecommendResponse(source=source, recommendations=recs)


def _ensure_loaded() -> None:
    if not recommender.loaded:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model is not loaded yet. Try again shortly.",
        )
