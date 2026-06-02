from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MovieDNA(BaseModel):
    mindBending: int = 0
    complexStory: int = 0
    sciFi: int = 0
    thriller: int = 0
    emotionalDepth: int = 0
    visualStyle: int = 0


class Movie(BaseModel):
    id: int
    imdb_id: Optional[str] = None
    title: str
    original_title: Optional[str] = None
    original_language: Optional[str] = None
    overview: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    cast: List[str] = Field(default_factory=list)
    crew: List[str] = Field(default_factory=list)
    vote_average: Optional[float] = None
    popularity: Optional[float] = None
    poster_url: Optional[str] = None
    moods: List[str] = Field(default_factory=list)
    dna: MovieDNA = Field(default_factory=MovieDNA)


class MovieSummary(BaseModel):
    id: int
    title: str
    vote_average: Optional[float] = None
    genres: List[str] = Field(default_factory=list)
    poster_url: Optional[str] = None


class Recommendation(BaseModel):
    movie: Movie
    match: int = Field(ge=0, le=100, description="Similarity match score 0-100")
    score: float = Field(ge=0.0, le=1.0, description="Raw cosine similarity 0-1")
    shared_genres: List[str] = Field(default_factory=list)
    shared_keywords: List[str] = Field(default_factory=list)
    shared_cast: List[str] = Field(default_factory=list)
    reason: str = ""


class RecommendRequest(BaseModel):
    title: Optional[str] = None
    movie_id: Optional[int] = None
    limit: int = Field(default=10, ge=1, le=50)


class RecommendResponse(BaseModel):
    source: Movie
    recommendations: List[Recommendation]


class SearchResponse(BaseModel):
    query: str
    results: List[MovieSummary]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    movies_count: int
    tfidf_shape: List[int]
    model_path: str
    version: str
    uptime_seconds: float
    poster_cache: Dict[str, Any] = Field(default_factory=dict)


class PingResponse(BaseModel):
    pong: bool
    timestamp_ms: int
