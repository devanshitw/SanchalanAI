import numpy as np
from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis

SampleDir = Path(__file__).resolve().parent.parent / "sample"

if not SampleDir.exists():
    raise SystemExit("Sample directory not found")

analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
analyzer.prepare(ctx_id=0, det_size=(320, 320))


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def get_embedding(path: Path) -> np.ndarray:
    image = load_image(path)
    faces = analyzer.get(image)
    if not faces:
        raise RuntimeError(f"No face detected in {path}")
    face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
    emb = np.asarray(face.embedding, dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb


embeddings = {}
for path in sorted(SampleDir.glob("id*.jpg")):
    embeddings[path.name] = get_embedding(path)

print("Embeddings computed for:")
for name in embeddings:
    print(" -", name)

print("\nPairwise cosine similarities:")
keys = list(embeddings.keys())
for i, a in enumerate(keys):
    for b in keys[i + 1 :]:
        sim = float(np.dot(embeddings[a], embeddings[b]))
        print(f"{a} vs {b}: {sim:.6f}")

query_path = SampleDir / "query_should_match_id0.jpg"
query_emb = get_embedding(query_path)
print("\nQuery similarity to enrolled identities:")
for name, emb in embeddings.items():
    sim = float(np.dot(query_emb, emb))
    print(f"query vs {name}: {sim:.6f}")
