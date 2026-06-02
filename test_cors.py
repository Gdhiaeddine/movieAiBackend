"""Verify the CORS middleware behavior without needing a real port."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def show(label, resp):
    acao = resp.headers.get("access-control-allow-origin", "")
    acac = resp.headers.get("access-control-allow-credentials", "")
    acam = resp.headers.get("access-control-allow-methods", "")
    print(f"  {label}")
    print(f"    status:  {resp.status_code}")
    print(f"    ACAO:    '{acao}'")
    print(f"    ACAC:    '{acac}'")
    if acam:
        print(f"    ACAM:    '{acam}'")


print("--- 1) plain GET /api/ping (no Origin) ---")
r = c.get("/api/ping")
show("GET /api/ping", r)

print("\n--- 2) cross-origin GET, Origin: https://monitoring.example.com ---")
r = c.get("/api/ping", headers={"Origin": "https://monitoring.example.com"})
show("GET with foreign origin", r)

print("\n--- 3) CORS preflight OPTIONS from a different origin ---")
r = c.options(
    "/api/ping",
    headers={
        "Origin": "https://monitoring.example.com",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "content-type",
    },
)
show("OPTIONS preflight", r)

print("\n--- 4) regression: recommend endpoint still works ---")
r = c.get("/api/recommend", params={"title": "Inception", "limit": 2})
print(f"  status: {r.status_code}, source: {r.json()['source']['title']}")

print("\n--- 5) health ---")
r = c.get("/api/health")
h = r.json()
print(f"  status: {r.status_code}, model: {h['status']}, movies: {h['movies_count']}")
