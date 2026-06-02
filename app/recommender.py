from __future__ import annotations

import ast
import logging
import math
import pickle
import random
import re
import threading
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import norm as sparse_norm

from .config import MODEL_PATH
from .schemas import Movie, MovieDNA, MovieSummary, Recommendation


log = logging.getLogger("recommender")


def _to_list(value: Any) -> List[str]:
    """Normalize a cell that may be a Python list, a stringified list, or NaN."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (ValueError, SyntaxError):
                pass
        return [p.strip() for p in s.split(",") if p.strip()]
    return [str(value)]


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _to_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    return s or None


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_title(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return " ".join(_WORD_RE.findall(str(text).lower()))


def _l2_normalize_rows(matrix: csr_matrix) -> csr_matrix:
    """Return a copy of the sparse matrix with each row L2-normalized."""
    if not issparse(matrix):
        raise TypeError("expected scipy sparse matrix")
    matrix = matrix.tocsr().astype(np.float32, copy=True)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0.0] = 1.0
    inv = 1.0 / norms
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        if end > start:
            matrix.data[start:end] *= inv[i]
    return matrix


class Recommender:
    """Content-based movie recommender backed by a pre-fitted TF-IDF matrix."""

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._matrix: Optional[csr_matrix] = None
        self._matrix_norm: Optional[csr_matrix] = None
        self._title_index: dict[str, int] = {}
        self._id_index: dict[int, int] = {}
        self._model_path: Path = MODEL_PATH
        self._lock = threading.Lock()
        self._loaded = False

    # ---------- lifecycle ----------

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def movies_count(self) -> int:
        return 0 if self._df is None else int(len(self._df))

    @property
    def tfidf_shape(self) -> Tuple[int, int]:
        if self._matrix is None:
            return (0, 0)
        r, c = self._matrix.shape
        return (int(r), int(c))

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            path = self._model_path
            log.info("Loading recommender model from %s", path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Model file not found at {path}. "
                    "Set MODEL_PATH env var or place movies_and_tfidf.pkl in the project."
                )

            with open(path, "rb") as f:
                payload = pickle.load(f)

            if not (isinstance(payload, tuple) and len(payload) == 2):
                raise ValueError(
                    "Unexpected pickle format: expected (DataFrame, csr_matrix) tuple."
                )

            df, matrix = payload
            if not isinstance(df, pd.DataFrame):
                raise ValueError("First pickle element must be a pandas DataFrame.")
            if not issparse(matrix):
                raise ValueError("Second pickle element must be a scipy sparse matrix.")
            if matrix.shape[0] != len(df):
                raise ValueError(
                    f"Row count mismatch: df={len(df)} matrix={matrix.shape[0]}"
                )

            df = df.reset_index(drop=True).copy()

            log.info("Normalizing TF-IDF matrix (%s nnz)...", matrix.nnz)
            matrix_norm = _l2_normalize_rows(matrix)

            title_index: dict[str, int] = {}
            for i, t in enumerate(df["title"].astype(str).tolist()):
                key = _normalize_title(t)
                if key and key not in title_index:
                    title_index[key] = i
            if "original_title" in df.columns:
                for i, t in enumerate(df["original_title"].astype(str).tolist()):
                    key = _normalize_title(t)
                    if key and key not in title_index:
                        title_index[key] = i

            id_index: dict[int, int] = {}
            if "id" in df.columns:
                for i, mid in enumerate(df["id"].tolist()):
                    try:
                        id_index[int(mid)] = i
                    except (TypeError, ValueError):
                        continue

            self._df = df
            self._matrix = matrix.tocsr()
            self._matrix_norm = matrix_norm
            self._title_index = title_index
            self._id_index = id_index
            self._loaded = True
            log.info(
                "Model loaded: %d movies, TF-IDF %s",
                len(df),
                tuple(matrix.shape),
            )

    # ---------- lookup helpers ----------

    def _require_loaded(self) -> None:
        if not self._loaded or self._df is None or self._matrix_norm is None:
            raise RuntimeError("Recommender model is not loaded yet.")

    def _row_to_movie(self, idx: int) -> Movie:
        self._require_loaded()
        assert self._df is not None
        row = self._df.iloc[idx]
        imdb_id = _to_optional_str(row.get("imdb_id"))
        genres = _to_list(row.get("genres"))
        keywords = _to_list(row.get("keywords"))
        overview = _to_optional_str(row.get("overview"))
        vote = _to_optional_float(row.get("vote_average"))
        popularity = _to_optional_float(row.get("popularity"))
        return Movie(
            id=int(row["id"]),
            imdb_id=imdb_id,
            title=str(row["title"]),
            original_title=_to_optional_str(row.get("original_title")),
            original_language=_to_optional_str(row.get("original_language")),
            overview=overview,
            genres=genres,
            keywords=keywords,
            cast=_to_list(row.get("cast")),
            crew=_to_list(row.get("crew")),
            vote_average=vote,
            popularity=popularity,
            poster_url=_build_poster_url(imdb_id),
            moods=_compute_moods(genres, keywords),
            dna=_compute_dna(genres, keywords, overview, popularity, vote),
        )

    def _row_to_summary(self, idx: int) -> MovieSummary:
        self._require_loaded()
        assert self._df is not None
        row = self._df.iloc[idx]
        return MovieSummary(
            id=int(row["id"]),
            title=str(row["title"]),
            vote_average=_to_optional_float(row.get("vote_average")),
            genres=_to_list(row.get("genres"))[:3],
            poster_url=_build_poster_url(_to_optional_str(row.get("imdb_id"))),
        )

    def find_index(
        self,
        title: Optional[str] = None,
        movie_id: Optional[int] = None,
    ) -> Optional[int]:
        self._require_loaded()
        if movie_id is not None:
            return self._id_index.get(int(movie_id))
        if title is None:
            return None
        key = _normalize_title(title)
        if not key:
            return None
        if key in self._title_index:
            return self._title_index[key]
        # fall back to a stricter "title contains query" match (min 3 chars)
        if len(key) >= 3:
            best: Optional[int] = None
            best_len = float("inf")
            for stored_key, idx in self._title_index.items():
                if key in stored_key and len(stored_key) < best_len:
                    best = idx
                    best_len = len(stored_key)
            if best is not None:
                return best
        return None

    # ---------- public API ----------

    def get_movie(
        self,
        title: Optional[str] = None,
        movie_id: Optional[int] = None,
    ) -> Optional[Movie]:
        idx = self.find_index(title=title, movie_id=movie_id)
        if idx is None:
            return None
        return self._row_to_movie(idx)

    def search(self, query: str, limit: int = 10) -> List[MovieSummary]:
        self._require_loaded()
        assert self._df is not None
        q = _normalize_title(query)
        if not q:
            return []

        titles = self._df["title"].astype(str).tolist()
        votes = self._df["vote_average"].fillna(0.0).astype(float).to_numpy()

        starts: list[tuple[float, int]] = []
        contains: list[tuple[float, int]] = []
        for i, t in enumerate(titles):
            nt = _normalize_title(t)
            if not nt:
                continue
            if nt == q:
                starts.append((1e9 + votes[i], i))
            elif nt.startswith(q):
                starts.append((1e6 + votes[i], i))
            elif q in nt:
                contains.append((votes[i], i))

        ranked = sorted(starts + contains, key=lambda x: x[0], reverse=True)
        seen: set[int] = set()
        out: List[MovieSummary] = []
        for _, idx in ranked:
            if idx in seen:
                continue
            seen.add(idx)
            out.append(self._row_to_summary(idx))
            if len(out) >= limit:
                break
        return out

    def random_movie(self, min_vote: float = 6.5) -> Movie:
        self._require_loaded()
        assert self._df is not None
        votes = self._df["vote_average"].fillna(0.0).astype(float).to_numpy()
        pop = (
            self._df["popularity"]
            .apply(_to_optional_float)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )
        mask = (votes >= min_vote) & (pop >= 5.0)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            candidates = np.arange(len(self._df))
        idx = int(random.choice(candidates))
        return self._row_to_movie(idx)

    def recommend(
        self,
        title: Optional[str] = None,
        movie_id: Optional[int] = None,
        limit: int = 10,
    ) -> Tuple[Movie, List[Recommendation]]:
        self._require_loaded()
        assert self._matrix_norm is not None and self._df is not None

        idx = self.find_index(title=title, movie_id=movie_id)
        if idx is None:
            raise LookupError("Movie not found in the dataset.")
        return self._recommend_for_index(idx, limit=limit)

    def recommend_random(self, limit: int = 10) -> Tuple[Movie, List[Recommendation]]:
        self._require_loaded()
        assert self._df is not None
        # only pick a "good" random source so the demo always looks great
        votes = self._df["vote_average"].fillna(0.0).astype(float).to_numpy()
        pop = (
            self._df["popularity"]
            .apply(_to_optional_float)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )
        mask = (votes >= 7.0) & (pop >= 10.0)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            candidates = np.arange(len(self._df))
        idx = int(random.choice(candidates))
        return self._recommend_for_index(idx, limit=limit)

    def _recommend_for_index(
        self, idx: int, limit: int = 10
    ) -> Tuple[Movie, List[Recommendation]]:
        assert self._matrix_norm is not None and self._df is not None
        source = self._row_to_movie(idx)
        row = self._matrix_norm.getrow(idx)
        scores = np.asarray(row.dot(self._matrix_norm.T).todense()).ravel()
        scores[idx] = -1.0

        k = max(1, min(limit, len(scores) - 1))
        top_idx = np.argpartition(-scores, k)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        src_genres = {g.lower() for g in source.genres}
        src_keywords = {k.lower() for k in source.keywords}
        src_cast = {c.lower() for c in source.cast}

        recs: List[Recommendation] = []
        for j in top_idx:
            j = int(j)
            score = float(max(0.0, min(1.0, scores[j])))
            movie = self._row_to_movie(j)
            shared_g = [g for g in movie.genres if g.lower() in src_genres]
            shared_k = [k for k in movie.keywords if k.lower() in src_keywords]
            shared_c = [c for c in movie.cast if c.lower() in src_cast]
            reason = _build_reason(source, movie, shared_g, shared_k, shared_c)
            recs.append(
                Recommendation(
                    movie=movie,
                    match=int(round(score * 100)),
                    score=score,
                    shared_genres=shared_g,
                    shared_keywords=shared_k,
                    shared_cast=shared_c,
                    reason=reason,
                )
            )
        return source, recs


def _build_poster_url(imdb_id: Optional[str]) -> Optional[str]:
    if not imdb_id:
        return None
    return None


_MIND_BENDING_KEYS = {
    "dream", "subconsciousness", "subconscious", "virtual reality",
    "time travel", "time", "memory", "philosophy", "twist", "identity",
    "reality", "simulation", "parallel universe", "alternate reality",
    "mind", "hallucination", "psychological", "perception",
}
_FUTURISTIC_KEYS = {
    "cyberpunk", "virtual reality", "artificial intelligence",
    "robot", "space", "future", "spaceship", "alien", "dystopia",
}
_DARK_KEYS = {
    "murder", "death", "dystopia", "violence", "revenge", "war",
    "serial killer", "blood", "gore",
}


def _has_any(values: Iterable[str], needles: set[str]) -> bool:
    vs = {str(v).lower() for v in values}
    return any(n in vs for n in needles)


def _count_any(values: Iterable[str], needles: set[str]) -> int:
    vs = {str(v).lower() for v in values}
    return sum(1 for n in needles if n in vs)


def _clamp01_100(x: float) -> int:
    return int(max(0.0, min(100.0, round(x))))


def _compute_dna(
    genres: List[str],
    keywords: List[str],
    overview: Optional[str],
    popularity: Optional[float],
    vote: Optional[float],
) -> MovieDNA:
    gset = {g.lower() for g in genres}
    kset = {k.lower() for k in keywords}
    kw_count = len(keywords)
    overview_len = len(overview or "")
    pop = float(popularity or 0.0)
    vote_score = float(vote or 0.0)

    mb_hits = _count_any(keywords, _MIND_BENDING_KEYS)
    mind_bending = 30 + mb_hits * 18
    if "Science Fiction" in genres or "Mystery" in genres:
        mind_bending += 10

    complex_story = 25 + min(kw_count, 12) * 4 + min(overview_len // 60, 15)
    if "Mystery" in genres or "Drama" in genres:
        complex_story += 8

    sci_fi = 10
    if "Science Fiction" in genres:
        sci_fi += 70
    if "Fantasy" in genres:
        sci_fi += 25
    if _has_any(keywords, _FUTURISTIC_KEYS):
        sci_fi += 15

    thriller = 10
    if "Thriller" in genres:
        thriller += 55
    if "Action" in genres:
        thriller += 25
    if "Horror" in genres:
        thriller += 25
    if "Mystery" in genres:
        thriller += 15
    if "Crime" in genres:
        thriller += 15

    emotional = 15
    if "Drama" in genres:
        emotional += 50
    if "Romance" in genres:
        emotional += 30
    if "Family" in genres:
        emotional += 20
    if "Animation" in genres:
        emotional += 10
    emotional += min(int(vote_score * 4), 20)

    visual = 25
    if "Animation" in genres:
        visual += 35
    if "Adventure" in genres:
        visual += 20
    if "Fantasy" in genres:
        visual += 20
    if "Action" in genres:
        visual += 15
    if "Science Fiction" in genres:
        visual += 15
    visual += min(int(pop / 4), 25)

    return MovieDNA(
        mindBending=_clamp01_100(mind_bending),
        complexStory=_clamp01_100(complex_story),
        sciFi=_clamp01_100(sci_fi),
        thriller=_clamp01_100(thriller),
        emotionalDepth=_clamp01_100(emotional),
        visualStyle=_clamp01_100(visual),
    )


def _compute_moods(genres: List[str], keywords: List[str]) -> List[str]:
    moods: List[str] = []
    if (
        _has_any(genres, {"horror", "thriller", "crime"})
        or _has_any(keywords, _DARK_KEYS)
    ):
        moods.append("Dark")
    if _has_any(genres, {"action", "thriller", "war"}):
        moods.append("Intense")
    if _has_any(genres, {"drama", "mystery", "science fiction"}) or _has_any(
        keywords, {"philosophy", "subconsciousness"}
    ):
        moods.append("Thought-Provoking")
    if _has_any(genres, {"drama", "romance", "family"}):
        moods.append("Emotional")
    if _has_any(genres, {"thriller", "mystery", "horror"}):
        moods.append("Suspenseful")
    if _has_any(genres, {"mystery", "science fiction"}) or _has_any(
        keywords, {"philosophy"}
    ):
        moods.append("Smart")
    if _has_any(genres, {"science fiction"}) or _has_any(
        keywords, _FUTURISTIC_KEYS
    ):
        moods.append("Futuristic")
    if _has_any(genres, {"comedy", "family", "animation"}):
        moods.append("Feel-Good")
    if _has_any(genres, {"adventure", "fantasy"}):
        moods.append("Epic")
    # de-dup preserving order, cap at 6
    seen: set[str] = set()
    out: List[str] = []
    for m in moods:
        if m not in seen:
            seen.add(m)
            out.append(m)
        if len(out) >= 6:
            break
    return out


def _build_reason(
    source: Movie,
    target: Movie,
    shared_genres: Iterable[str],
    shared_keywords: Iterable[str],
    shared_cast: Iterable[str],
) -> str:
    sg = list(shared_genres)
    sk = list(shared_keywords)
    sc = list(shared_cast)
    parts: List[str] = []
    if sg:
        parts.append(
            f"shares the {', '.join(sg[:2]).lower()} tone of {source.title}"
        )
    if sk:
        parts.append(f"explores {', '.join(sk[:2])}")
    if sc and not sk:
        parts.append(f"features {', '.join(sc[:2])}")
    if not parts:
        return f"Stylistically similar to {source.title}."
    return (" — ".join(parts)).capitalize() + "."


recommender = Recommender()
