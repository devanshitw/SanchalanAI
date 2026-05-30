import os
from pathlib import Path
import requests

BASE_URL = os.environ.get("SERVICE_URL", "http://localhost:8000")
SAMPLE_DIR = Path(__file__).resolve().parent / ".." / "sample"
SAMPLE_DIR = SAMPLE_DIR.resolve()

if not SAMPLE_DIR.exists():
    raise SystemExit(f"Sample directory not found: {SAMPLE_DIR}")

identities = sorted([p for p in SAMPLE_DIR.iterdir() if p.name.startswith("id") and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
if len(identities) < 5:
    raise SystemExit("Need 5 enroll sample identities in ./sample")

print("Using service at", BASE_URL)
for idx, path in enumerate(identities[:5]):
    identity_id = f"id{idx}"
    print(f"Enrolling {identity_id} <- {path.name}")
    with open(path, "rb") as f:
        r = requests.post(f"{BASE_URL}/enroll", files={"image": f}, data={"id": identity_id})
    print(r.status_code, r.text)

query_path = SAMPLE_DIR / "query_should_match_id0.jpg"
print(f"Querying with {query_path}")
with open(query_path, "rb") as f:
    r = requests.post(f"{BASE_URL}/search", files={"image": f})
print(r.status_code, r.text)
