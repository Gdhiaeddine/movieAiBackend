"""
Evaluation harness for the content-based recommender.

Reports (across a fixed pool of well-known movies):
  1. Top-K score distribution (mean, median, min, max) of cosine similarity
  2. Genre overlap (Jaccard) of recommendations with the source
  3. Tag overlap (Jaccard) of recommendations with the source
  4. Intra-list diversity (mean pairwise distance of recs)
  5. Sequel / prequel recovery: how often the known follow-up is in the top-10
  6. Catalog coverage: # unique movies surfaced across N queries
  7. Per-query latency

Run from the backend folder:
    ./.venv/Scripts/python.exe eval.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.recommender import Recommender  # noqa: E402

# (source title, expected sequel fragment to look for in rec titles)
GROUND_TRUTH = [
    ("The Matrix",                "reloaded"),
    ("The Matrix Reloaded",       "revolutions"),
    ("The Matrix Revolutions",    "reloaded"),
    ("Inception",                 "interstellar"),  # Nolan's sci-fi
    ("The Dark Knight",           "batman begins"),
    ("Batman Begins",             "dark knight"),
    ("Iron Man",                  "iron man 2"),
    ("Toy Story",                 "toy story 2"),
    ("Avatar",                    "way of water"),
    ("Pirates of the Caribbean: The Curse of the Black Pearl", "dead man"),
    ("Harry Potter and the Sorcerer's Stone",                 "chamber of secrets"),
    ("Harry Potter and the Chamber of Secrets",                "prisoner of azkaban"),
    ("The Lord of the Rings: The Fellowship of the Ring",     "two towers"),
    ("Star Wars",                 "empire strikes back"),
    ("Back to the Future",        "part ii"),
    ("The Terminator",            "judgment day"),
    ("Aliens",                    "alien"),
    ("Indiana Jones and the Raiders of the Lost Ark",         "temple of doom"),
    ("Shrek",                     "shrek 2"),
    ("Spider-Man",                "spider-man 2"),
    ("Die Hard",                  "die hard 2"),
    ("Jurassic Park",             "lost world"),
    ("Rocky",                     "rocky ii"),
    ("Jaws",                      "jaws 2"),
    ("Batman",                    "batman returns"),
]

GENERAL_TITLES = [
    "The Godfather", "Pulp Fiction", "Forrest Gump", "Fight Club",
    "Goodfellas", "The Shawshank Redemption", "The Silence of the Lambs",
    "Gladiator", "Titanic", "Dune", "Blade Runner", "The Lion King",
]


def jaccard(a, b) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    print("Loading recommender model...")
    t0 = time.perf_counter()
    r = Recommender()
    r.load()
    print(f"  loaded in {(time.perf_counter() - t0) * 1000:.0f} ms "
          f"({r.movies_count:,} movies, TF-IDF {r.tfidf_shape})\n")

    # Resolve sources
    sources = []
    for title, frag in GROUND_TRUTH + [(t, "") for t in GENERAL_TITLES]:
        idx = r.find_index(title=title)
        if idx is None:
            print(f"  [skip] source not found: {title}")
            continue
        sources.append((title, idx, frag))
    print(f"Running recommend(k=10) on {len(sources)} sources...\n")

    top1_scores: list[float] = []
    top10_scores: list[float] = []
    genre_jaccards: list[float] = []
    tag_jaccards: list[float] = []
    diversities: list[float] = []
    covered: set[int] = set()
    sequel_in_top10: list[tuple[str, str]] = []
    sequel_in_top3: list[tuple[str, str]] = []
    latencies: list[float] = []

    for title, idx, frag in sources:
        t = time.perf_counter()
        source, recs = r.recommend(movie_id=None, title=title, limit=10)  # type: ignore[arg-type]
        latencies.append((time.perf_counter() - t) * 1000)

        if not recs:
            continue

        top1_scores.append(recs[0].score)
        top10_scores.extend(rec.score for rec in recs)
        genre_jaccards.append(statistics.mean(
            jaccard(source.genres, rec.movie.genres) for rec in recs
        ))
        tag_jaccards.append(statistics.mean(
            jaccard(source.keywords, rec.movie.keywords) for rec in recs
        ))

        # intra-list diversity: 1 - mean pairwise cosine of the rec rows
        idxs = [int(np.where(r._df["id"] == rec.movie.id)[0][0]) for rec in recs]  # type: ignore[attr-defined]
        rows = r._matrix_norm[idxs]  # type: ignore[attr-defined]
        sim = (rows @ rows.T).toarray()
        iu = np.triu_indices_from(sim, k=1)
        pairwise = sim[iu]
        diversities.append(1.0 - float(pairwise.mean()))

        covered.update(rec.movie.id for rec in recs)

        if frag:
            for rank, rec in enumerate(recs, start=1):
                if frag in rec.movie.title.lower():
                    if rank <= 3:
                        sequel_in_top3.append((title, rec.movie.title))
                    sequel_in_top10.append((title, rec.movie.title))
                    break

    def block(title, lines):
        print(title)
        for ln in lines:
            print(ln)
        print()

    total_with_sequel = sum(1 for s in sources if s[2])

    block(
        "── 1. Top-K cosine similarity score distribution ──",
        [
            f"  queries evaluated:    {len(sources)}",
            f"  top-1  mean:          {statistics.mean(top1_scores):.4f}",
            f"  top-1  median:        {statistics.median(top1_scores):.4f}",
            f"  top-1  min / max:     {min(top1_scores):.4f} / {max(top1_scores):.4f}",
            f"  top-10 mean:          {statistics.mean(top10_scores):.4f}",
            f"  top-10 stdev:         {statistics.stdev(top10_scores):.4f}",
        ],
    )

    block(
        "── 2. Genre overlap with source (Jaccard) ──",
        [
            f"  per-query mean:       {statistics.mean(genre_jaccards):.4f}",
            f"  median:               {statistics.median(genre_jaccards):.4f}",
            f"  range:                {min(genre_jaccards):.4f} – {max(genre_jaccards):.4f}",
            "  (1.0 = perfect genre match, 0.0 = no shared genres)",
        ],
    )

    block(
        "── 3. Keyword (tag) overlap with source (Jaccard) ──",
        [
            f"  per-query mean:       {statistics.mean(tag_jaccards):.4f}",
            f"  median:               {statistics.median(tag_jaccards):.4f}",
            f"  range:                {min(tag_jaccards):.4f} – {max(tag_jaccards):.4f}",
            "  (compares Movie.keywords, the curated TMDB tag-like terms)",
        ],
    )

    block(
        "── 4. Intra-list diversity (mean 1−cosine between recs) ──",
        [
            f"  per-query mean:       {statistics.mean(diversities):.4f}",
            f"  median:               {statistics.median(diversities):.4f}",
            f"  range:                {min(diversities):.4f} – {max(diversities):.4f}",
            "  (0.0 = all recs near-duplicates, 1.0 = orthogonal; good recsys ~0.6–0.8)",
        ],
    )

    block(
        "── 5. Sequel / prequel recovery ──",
        [
            f"  sources with a known follow-up:  {total_with_sequel}",
            f"  follow-up found in top-10:       {len(sequel_in_top10):>2} / {total_with_sequel}"
            f"   ({100 * len(sequel_in_top10) / max(1, total_with_sequel):.1f}%)",
            f"  follow-up found in top-3:        {len(sequel_in_top3):>2} / {total_with_sequel}"
            f"   ({100 * len(sequel_in_top3) / max(1, total_with_sequel):.1f}%)",
        ] + [f"    {s:55s} → {f}" for s, f in sequel_in_top10],
    )

    block(
        "── 6. Catalog coverage ──",
        [
            f"  total queries:          {len(sources)}",
            f"  unique movies surfaced: {len(covered):,}",
            f"  % of catalog (46,628):  {100 * len(covered) / r.movies_count:.2f}%",
        ],
    )

    block(
        "── 7. Latency (recommend only) ──",
        [
            f"  mean:    {statistics.mean(latencies):.1f} ms",
            f"  median:  {statistics.median(latencies):.1f} ms",
            f"  p95:     {sorted(latencies)[int(0.95 * len(latencies))]:.1f} ms",
            f"  max:     {max(latencies):.1f} ms",
        ],
    )

    print("── quick sample (first 3 sources) ──")
    for title, idx, _ in sources[:3]:
        source, recs = r.recommend(title=title, limit=5)
        print(f"  {title}:")
        for rec in recs:
            print(f"    {rec.score:5.3f}  {rec.movie.title}")
    print()


if __name__ == "__main__":
    main()
