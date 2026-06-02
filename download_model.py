"""Download the trained model (movies_and_tfidf.pkl) from a public URL.

Use this when deploying to environments that don't ship the pickle with
the repo (e.g. Render free tier, Fly.io, etc.). The backend will also
auto-download at startup if `MODEL_DOWNLOAD_URL` is set and the model
file is missing — this script is for one-off setup or local refreshes.

Examples
--------
    # download the resolved model path
    python download_model.py --url https://example.com/movies_and_tfidf.pkl

    # verify a SHA-256 after download (recommended for production)
    python download_model.py \\
        --url https://example.com/movies_and_tfidf.pkl \\
        --sha256 9f1c...a2b4
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from app.config import (  # noqa: E402
    MODEL_DOWNLOAD_SHA256,
    MODEL_DOWNLOAD_URL,
    MODEL_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
log = logging.getLogger("download_model")

CHUNK = 1024 * 1024  # 1 MB


def _is_http(url: str) -> bool:
    scheme = urlparse(url).scheme.lower()
    return scheme in ("http", "https")


def _download(url: str, dest: Path, sha256: str | None) -> None:
    log.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    hasher = hashlib.sha256() if sha256 else None
    bytes_written = 0

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        with open(tmp, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=CHUNK):
                f.write(chunk)
                bytes_written += len(chunk)
                if hasher is not None:
                    hasher.update(chunk)
                if total:
                    pct = 100 * bytes_written / total
                    print(
                        f"\r  {bytes_written / 1e6:7.2f} MB / {total / 1e6:7.2f} MB  ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
    if total:
        print()  # newline after progress line

    if hasher is not None:
        digest = hasher.hexdigest()
        if digest.lower() != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"SHA-256 mismatch!\n  expected: {sha256}\n  got:      {digest}"
            )
        log.info("SHA-256 verified: %s", digest)

    tmp.replace(dest)
    log.info("Saved %s (%d bytes)", dest, dest.stat().st_size)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default=MODEL_DOWNLOAD_URL,
        help="Public URL of movies_and_tfidf.pkl (or env MODEL_DOWNLOAD_URL)",
    )
    p.add_argument(
        "--dest",
        default=str(MODEL_PATH),
        help="Destination path (default: resolved MODEL_PATH)",
    )
    p.add_argument(
        "--sha256",
        default=MODEL_DOWNLOAD_SHA256,
        help="Optional expected SHA-256 to verify integrity",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination already exists",
    )
    args = p.parse_args()

    if not args.url:
        log.error(
            "No URL provided. Pass --url or set MODEL_DOWNLOAD_URL in the env."
        )
        return 2
    if not _is_http(args.url):
        log.error("Only http(s) URLs are supported (got: %s)", args.url)
        return 2

    dest = Path(args.dest)
    if dest.is_file() and not args.force:
        log.info("Already present at %s (%d bytes) — use --force to redownload",
                 dest, dest.stat().st_size)
        return 0

    _download(args.url, dest, args.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
