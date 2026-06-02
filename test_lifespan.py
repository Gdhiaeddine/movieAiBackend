"""Smoke test: HEAD /, HEAD /api/ping, GET /, GET /api/ping, plus the
auto-download path (set MODEL_PATH to a temp file + MODEL_DOWNLOAD_URL)."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# fake server-side env before importing app
TMP = Path("C:/Users/pc/AppData/Local/Temp/opencode/auto_dest.pkl")
TMP.parent.mkdir(parents=True, exist_ok=True)
if TMP.exists():
    TMP.unlink()
os.environ["MODEL_PATH"] = str(TMP)
os.environ["MODEL_DOWNLOAD_URL"] = "http://127.0.0.1:18999/movies_and_tfidf.pkl"
# (no SHA → downloader will skip verification)

# re-import config now that env is set
import importlib
import app.config
importlib.reload(app.config)
import app.main
importlib.reload(app.main)

print(f"MODEL_PATH resolves to: {app.config.MODEL_PATH}")
print(f"  exists before start?  {app.config.MODEL_PATH.is_file()}")
print(f"MODEL_DOWNLOAD_URL:     {app.config.MODEL_DOWNLOAD_URL}")

# Re-create app because reload may have left a stale one
from fastapi.testclient import TestClient

print("\n--- start app (TestClient triggers lifespan) ---")
with TestClient(app.main.app) as c:
    print(f"  MODEL_PATH exists after start? {app.config.MODEL_PATH.is_file()}")
    if app.config.MODEL_PATH.is_file():
        print(f"  size: {app.config.MODEL_PATH.stat().st_size} bytes")

    print("\n--- HEAD / (was 405) ---")
    r = c.head("/")
    print(f"  status: {r.status_code}")

    print("\n--- GET / ---")
    r = c.get("/")
    print(f"  status: {r.status_code}  body: {r.json()}")

    print("\n--- GET /api/ping ---")
    r = c.get("/api/ping")
    print(f"  status: {r.status_code}  body: {r.json()}")

    print("\n--- HEAD /api/ping ---")
    r = c.head("/api/ping")
    print(f"  status: {r.status_code}")

    print("\n--- GET /api/health ---")
    r = c.get("/api/health")
    print(f"  status: {r.status_code}  body: {r.json()}")

# clean up the temp download
if TMP.exists():
    TMP.unlink()
print("\n--- done ---")
