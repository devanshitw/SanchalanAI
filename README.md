# Task 4 — Face Match + Vector Search (starter)

Fill in the two endpoints in `app/main.py`. Deploy it live. Report numbers in `RESULTS.md`.

## What to build
A FastAPI service with:
- `POST /enroll` — accepts a face image, computes an **ArcFace / InsightFace** embedding (detect + align first), stores it in **Qdrant** with an `id`.
- `POST /search` — accepts a query image, returns the **top match** `id` + cosine score, using Qdrant ANN search (not a brute-force Python loop).

## Get sample faces
```bash
python scripts/get_sample_faces.py   # downloads 5 identities + 1 query from LFW into ./sample
```
(Or use any 5 public face images + 1 query of your own — note the source.)

## Run locally
```bash
pip install -r requirements.txt
# start Qdrant (docker): docker run -p 6333:6333 qdrant/qdrant
uvicorn app.main:app --reload
```

## Run with Qdrant Cloud or native server
If you are using Qdrant Cloud or a native binary, set:
```bash
export QDRANT_URL="https://your-qdrant-endpoint"
export QDRANT_API_KEY="your-api-key"
uvicorn app.main:app --reload
```

## Test the service
1. Download sample images:
```bash
python scripts/get_sample_faces.py
```
2. Enroll the 5 sample identities and search the query image:
```bash
python scripts/test_service.py
```

## Deliver
1. This repo (public GitHub, real commit history).
2. A **live URL** where `/enroll` and `/search` work (Render/Railway/Fly/HF Spaces/cloud).
3. `RESULTS.md` filled in (see template there).
4. `AI_LOG.md` filled in.

## Hard requirements
- Detect + **align** the face before embedding. Tilted/off-center faces must still work.
- Use **Qdrant ANN** for search. A brute-force loop over all vectors will not pass.
- Show **one hard negative** (a look-alike) and how your threshold handles it.

Time: ~1 day. Questions → reply to the email.
