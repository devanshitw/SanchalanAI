import os
import time
import uuid
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from insightface.app import FaceAnalysis
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, SearchParams, VectorParams


QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "faces")
VECTOR_SIZE = 512

app = FastAPI(title="Face Match + Vector Search")
face_analyzer: Optional[FaceAnalysis] = None
qdrant: Optional[QdrantClient] = None


def init_face_analyzer() -> FaceAnalysis:
    global face_analyzer
    if face_analyzer is None:
        face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
    return face_analyzer


def init_qdrant() -> QdrantClient:
    global qdrant
    if qdrant is None:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        try:
            if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
                qdrant.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.DOT),
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Qdrant. Check QDRANT_URL, QDRANT_API_KEY, and collection permissions."
            ) from exc
    return qdrant


def load_image_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def detect_and_embed(image: np.ndarray) -> np.ndarray:
    analyzer = init_face_analyzer()
    faces = analyzer.get(image)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image")

    # Choose the most confident face if multiple are found.
    def face_score(f):
        if hasattr(f, "det_score"):
            return f.det_score
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    face = max(faces, key=face_score)
    embedding = np.asarray(face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0 or np.isnan(norm):
        raise HTTPException(status_code=500, detail="Invalid face embedding")
    return embedding / norm


@app.on_event("startup")
def startup():
    init_face_analyzer()
    init_qdrant()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/enroll")
async def enroll(id: str = Form(...), image: UploadFile = File(...)):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image upload")

    embedding = detect_and_embed(load_image_bytes(content))
    client = init_qdrant()
    point_id = uuid.uuid5(uuid.NAMESPACE_URL, id)
    point = PointStruct(id=point_id, vector=embedding.tolist(), payload={"user_id": id})
    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return {"id": id, "stored": True}


@app.post("/search")
async def search(image: UploadFile = File(...)):
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image upload")

    query_vector = detect_and_embed(load_image_bytes(content))
    client = init_qdrant()
    start = time.perf_counter()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=1,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128),
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    if not results.points:
        raise HTTPException(status_code=404, detail="No enrolled faces found")

    top = results.points[0]
    match_id = None
    if top.payload:
        match_id = top.payload.get("user_id")
    if match_id is None:
        match_id = str(top.id)

    cosine_score = float(top.score)
    return {
        "match_id": match_id,
        "cosine": cosine_score,
        "search_latency_ms": round(latency_ms, 2),
    }
