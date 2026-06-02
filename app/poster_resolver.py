"""Resolve IMDb poster URLs via the public suggestion endpoint.

We hit `https://v3.sg.media-imdb.com/suggestion/t/<imdb_id>.json` (the same
endpoint that powers IMDb autocomplete) which returns a JSON payload
including `i.imageUrl` — a direct CDN link to the title's poster.

Results are cached:
  * in-memory for the lifetime of the process
  * on disk as JSON (`POSTER_CACHE_PATH`) so restarts stay fast

Missing posters are cached as empty strings to avoid retrying constantly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from .config import (
    ENABLE_POSTER_FETCH,
    POSTER_CACHE_PATH,
    POSTER_FETCH_CONCURRENCY,
    POSTER_FETCH_TIMEOUT,
)
from .schemas import Movie, Recommendation

log = logging.getLogger("posters")

_BASE_URL = "https://v3.sg.media-imdb.com/suggestion/t/{imdb_id}.json"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.imdb.com/",
}

_CACHE_VERSION = 1


class PosterResolver:
    def __init__(self, cache_path: Path) -> None:
        self._cache_path = cache_path
        self._cache: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._loaded = False

    # ---------- cache lifecycle ----------

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._cache_path.is_file():
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Poster cache unreadable, starting fresh: %s", exc)
            return
        if isinstance(payload, dict) and payload.get("v") == _CACHE_VERSION:
            data = payload.get("data") or {}
            if isinstance(data, dict):
                self._cache = {str(k): str(v) for k, v in data.items()}
                log.info("Loaded %d poster cache entries", len(self._cache))

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {"v": _CACHE_VERSION, "data": self._cache},
                    f,
                    ensure_ascii=False,
                )
            tmp.replace(self._cache_path)
            self._dirty = False
            log.debug("Saved %d poster cache entries", len(self._cache))
        except OSError as exc:
            log.warning("Could not persist poster cache: %s", exc)

    # ---------- public API ----------

    def get_cached(self, imdb_id: Optional[str]) -> Optional[str]:
        if not imdb_id:
            return None
        v = self._cache.get(imdb_id)
        return v or None

    def stats(self) -> Dict[str, Any]:
        """Snapshot of cache state for the /api/health endpoint."""
        return {
            "enabled": ENABLE_POSTER_FETCH,
            "path": str(self._cache_path),
            "entries": len(self._cache),
            "loaded": self._loaded,
        }

    async def resolve_many(self, imdb_ids: Iterable[str]) -> Dict[str, Optional[str]]:
        if not ENABLE_POSTER_FETCH:
            return {i: self.get_cached(i) for i in imdb_ids}

        uniq = [i for i in {x for x in imdb_ids if x} if i not in self._cache]
        if uniq:
            sem = asyncio.Semaphore(max(1, POSTER_FETCH_CONCURRENCY))
            async with httpx.AsyncClient(
                timeout=POSTER_FETCH_TIMEOUT,
                headers=_HEADERS,
                follow_redirects=True,
                http2=False,
            ) as client:
                await asyncio.gather(
                    *(self._fetch_one(client, sem, i) for i in uniq),
                    return_exceptions=True,
                )
            if self._dirty:
                # persist opportunistically; tiny file so it's cheap
                self.save()

        return {i: self.get_cached(i) for i in imdb_ids}

    # ---------- internal ----------

    async def _fetch_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        imdb_id: str,
    ) -> None:
        async with sem:
            if imdb_id in self._cache:
                return
            try:
                resp = await client.get(_BASE_URL.format(imdb_id=imdb_id))
                if resp.status_code != 200:
                    self._store(imdb_id, "")
                    return
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.debug("Poster fetch failed for %s: %s", imdb_id, exc)
                # cache empty briefly so we don't hammer the endpoint
                self._store(imdb_id, "")
                return

            url = _extract_image_url(data, imdb_id)
            self._store(imdb_id, url or "")

    def _store(self, imdb_id: str, url: str) -> None:
        self._cache[imdb_id] = url
        self._dirty = True


def _extract_image_url(payload: Any, imdb_id: str) -> Optional[str]:
    """Pull `i.imageUrl` from an IMDb suggestion payload."""
    if not isinstance(payload, dict):
        return None
    items = payload.get("d") or []
    if not isinstance(items, list):
        return None
    # prefer the exact id match, fall back to the first item
    match = next(
        (it for it in items if isinstance(it, dict) and it.get("id") == imdb_id),
        None,
    )
    if match is None and items:
        match = items[0] if isinstance(items[0], dict) else None
    if not isinstance(match, dict):
        return None
    image = match.get("i")
    if not isinstance(image, dict):
        return None
    url = image.get("imageUrl")
    if isinstance(url, str) and url.startswith("http"):
        return url
    return None


# ---------- enrichment helpers ----------

resolver = PosterResolver(POSTER_CACHE_PATH)


async def enrich_movie(movie: Movie) -> Movie:
    await enrich_movies([movie])
    return movie


async def enrich_movies(movies: List[Movie]) -> List[Movie]:
    ids = [m.imdb_id for m in movies if m.imdb_id]
    if not ids:
        return movies
    mapping = await resolver.resolve_many(ids)
    for m in movies:
        if m.imdb_id and not m.poster_url:
            url = mapping.get(m.imdb_id)
            if url:
                m.poster_url = url
    return movies


async def enrich_recommendation_response(
    source: Movie, recs: List[Recommendation]
) -> None:
    all_movies: List[Movie] = [source] + [r.movie for r in recs]
    await enrich_movies(all_movies)
