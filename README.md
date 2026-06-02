# MovieAI Backend

FastAPI service that serves content-based movie recommendations from the
pre-trained TF-IDF model bundled in `movies_and_tfidf.pkl` (a tuple of a
pandas `DataFrame` with 46,628 movies and a 46,628 × 57,971 `csr_matrix`
TF-IDF feature matrix).

## Layout

```
backend/
  app/
    __init__.py
    config.py        # env + model path resolution
    schemas.py       # Pydantic request/response models
    recommender.py   # loads the pickle, builds L2-normalized matrix,
                     # exposes search / recommend / random helpers
    poster_resolver.py  # IMDb suggestion fetcher + on-disk cache
    main.py          # FastAPI app, routes, CORS, lifespan loader
  movies_and_tfidf.pkl  # the trained model (DataFrame + TF-IDF csr_matrix)
  poster_cache.json  # generated on shutdown (gitignored)
  requirements.txt
  .env.example
  run.ps1            # convenience launcher (venv + install + uvicorn)
```

## Quick start (Windows PowerShell)

```powershell
cd site\backend
./run.ps1
```

This creates a `.venv`, installs requirements, and starts the API at
http://127.0.0.1:8000. Interactive docs are at http://127.0.0.1:8000/docs.

## Quick start (manual)

```powershell
cd site\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Deploying to Render / Fly.io / Heroku / Vercel

The pickled model is **42 MB and gitignored**, so platforms that build
from a git repo (Render free tier, Heroku, Vercel, Fly.io without a
volume) won't ship it. Two options:

### Option A — Auto-download on startup (recommended)

1. Upload `movies_and_tfidf.pkl` somewhere public. Free options:
   - **GitHub Releases** — attach the file to a release, copy the URL
   - **Cloudflare R2** — free egress up to 10 GB/month
   - **S3** — pay-as-you-go

2. Compute the SHA-256 once locally:
   ```powershell
   Get-FileHash backend\movies_and_tfidf.pkl -Algorithm SHA256
   ```

3. In your host's dashboard, set environment variables:
   ```
   MODEL_DOWNLOAD_URL=https://github.com/<you>/<repo>/releases/download/v1/movies_and_tfidf.pkl
   MODEL_DOWNLOAD_SHA256=<the hash from step 2>
   ```

4. The backend will fetch the pickle on first request, verify the hash,
   and persist it to the resolved `MODEL_PATH`. On subsequent restarts
   the file is already there and the download is skipped.

Health-check path on Render: set it to **`/api/ping`** (or `/`). The
backend supports both `GET` and `HEAD` on those paths.

### Option B — Docker image with the model baked in

Add a line to your `Dockerfile`:
```dockerfile
COPY movies_and_tfidf.pkl /app/movies_and_tfidf.pkl
```
Image grows by ~42 MB; no network dependency at startup. Use this if
you're deploying via a container registry and want a single self-
contained artifact.

### Manual download (one-off)

```powershell
# download the resolved model path
.\.venv\Scripts\python.exe download_model.py --url https://example.com/movies_and_tfidf.pkl

# with integrity check
.\.venv\Scripts\python.exe download_model.py --url https://example.com/movies_and_tfidf.pkl --sha256 <hash>

# force re-download
.\.venv\Scripts\python.exe download_model.py --url https://example.com/movies_and_tfidf.pkl --force
```

## Monitoring endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` / `HEAD` | `/api/ping` | Tiny heartbeat. Returns `{ "pong": true, "timestamp_ms": ... }` in < 1 ms. **Use this for UptimeRobot / Pingdom / Better Uptime.** |
| `GET` / `HEAD` | `/api/health` | Rich diagnostic snapshot. Returns `200` if the model is loaded, `503` otherwise. Includes version, uptime, and poster-cache stats. |

### UptimeRobot one-liner

Add a new **HTTP(s) monitor** with:

- **URL:** `https://your-domain.com/api/ping`
- **Monitoring Interval:** 5 minutes
- **Keyword:** `pong` (UptimeRobot only marks it up if the body contains this string)
- **HTTP Method:** GET (HEAD also works)

Example responses:

```bash
$ curl -s http://127.0.0.1:8000/api/ping
{"pong":true,"timestamp_ms":1780428252618}

$ curl -s http://127.0.0.1:8000/api/health
{
  "status": "ok",
  "model_loaded": true,
  "movies_count": 46628,
  "tfidf_shape": [46628, 57971],
  "version": "1.0.0",
  "uptime_seconds": 60.69,
  "poster_cache": { "enabled": true, "entries": 99, "loaded": true }
}
```

## Model location

The pickled model `movies_and_tfidf.pkl` lives **inside this backend folder**
(`site/backend/movies_and_tfidf.pkl`). At startup the backend resolves it in
this order — the first existing path wins:

1. `MODEL_PATH` env variable (if set and existing).
2. `./movies_and_tfidf.pkl` ← **default — next to this README**.
3. `../movies_and_tfidf.pkl` (the `site/` folder, legacy fallback).
4. `../../movies_and_tfidf.pkl` (workspace root, legacy fallback).

The pickle must be a tuple `(pandas.DataFrame, scipy.sparse.csr_matrix)`
where the matrix has one TF-IDF row per movie in the DataFrame, aligned
by index.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET    | `/`                          | Service info |
| GET    | `/api/ping`                  | Lightweight heartbeat (use for UptimeRobot) |
| GET    | `/api/health`                | Health + model status |
| GET    | `/api/movies/search?q=...&limit=` | Search by title |
| GET    | `/api/movies/random`         | Random highly-rated movie |
| GET    | `/api/movies/{id}`           | Full movie details by TMDB id |
| GET    | `/api/recommend?title=...&limit=` | Recommend (query string) |
| POST   | `/api/recommend`             | Recommend (JSON body) |
| GET    | `/api/recommend/random`      | Random recommendations (curated pool) |

### Example

```powershell
# Health
curl http://127.0.0.1:8000/api/health

# Search
curl "http://127.0.0.1:8000/api/movies/search?q=inception&limit=5"

# Recommend by title
curl "http://127.0.0.1:8000/api/recommend?title=Inception&limit=8"

# Recommend by body
curl -Method POST http://127.0.0.1:8000/api/recommend `
     -ContentType application/json `
     -Body '{"title":"The Matrix","limit":6}'
```

### Response shape (recommend)

```json
{
  "source": { "id": 27205, "title": "Inception", "genres": ["Action","Science Fiction"], ... },
  "recommendations": [
    {
      "movie": { "id": 157336, "title": "Interstellar", ... },
      "match": 42,
      "score": 0.4231,
      "shared_genres": ["Science Fiction"],
      "shared_keywords": ["dream"],
      "shared_cast": [],
      "reason": "Shares the science fiction tone of Inception — explores dream."
    }
  ]
}
```

## How recommendations are computed

1. At startup, the pickle is unpickled into `(df, tfidf)` and each row of
   the sparse TF-IDF matrix is L2-normalized in float32.
2. On request, the source row vector is multiplied against the
   transposed normalized matrix — yielding cosine similarities in one
   sparse matmul.
3. The source row is masked out, `np.argpartition` picks the top-k, and
   the result is enriched with shared genres/keywords/cast and a
   human-readable reason string.

## Environment variables

Copy `.env.example` to `.env` (or export in your shell) to override
defaults:

- `MODEL_PATH` — absolute path to the `.pkl` file.
- `MODEL_DOWNLOAD_URL` — public URL of the `.pkl` to fetch on startup if
  the file is missing on disk. Use this when deploying to hosts that
  don't ship the 42 MB pickle with the repo (Render free tier, Fly.io,
  Heroku, Vercel, …). Combine with `MODEL_DOWNLOAD_SHA256` to verify
  integrity. You can host the file on GitHub Releases, Cloudflare R2,
  S3, or any HTTPS-accessible URL.
- `MODEL_DOWNLOAD_SHA256` — expected SHA-256 of the downloaded pickle.
  Recommended for production.
- `CORS_ORIGINS` — comma-separated list of allowed origins (default `*`).
  Setting it to `*` opens the API to any origin (e.g. for monitoring tools
  hitting from different IPs). For a production frontend, prefer an
  explicit list like `http://localhost:3000,https://your-domain.com`.
- `DEFAULT_LIMIT`, `MAX_LIMIT` — pagination defaults for recommendations.
- `ENABLE_POSTER_FETCH` — `true` (default) enriches responses with IMDb
  poster URLs. Set `false` to disable.
- `POSTER_CACHE_PATH` — file path for the on-disk poster cache
  (default `poster_cache.json` next to the backend).
- `POSTER_FETCH_TIMEOUT` — per-request timeout in seconds (default `5.0`).
- `POSTER_FETCH_CONCURRENCY` — max parallel IMDb fetches (default `8`).

## Posters

`movie.poster_url` is populated lazily by `app/poster_resolver.py` using
IMDb's public autocomplete suggestion endpoint:

```
https://v3.sg.media-imdb.com/suggestion/t/{imdb_id}.json
```

This is the same endpoint IMDb's own search box hits — it requires no API
key and returns a direct `m.media-amazon.com` CDN URL. The resolver:

1. Looks up the in-memory cache; returns immediately on hit.
2. For misses, fires concurrent `httpx` requests (bounded by
   `POSTER_FETCH_CONCURRENCY`) and gathers results.
3. Persists the cache to `POSTER_CACHE_PATH` on shutdown and re-loads
   it on startup, so warm restarts are instant.
4. Caches missing posters as empty strings (no retry storm).

Typical latencies on a warm process:
- Cold query (9 uncached movies): ~1.5 s
- Warm query (cache hit): ~20–40 ms
