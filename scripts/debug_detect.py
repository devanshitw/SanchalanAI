from pathlib import Path
import numpy as np
from app.main import init_face_analyzer, load_image_bytes

sample = Path(__file__).resolve().parent.parent / "sample" / "id0_Atal_Bihari_Vajpayee.jpg"
print('sample path:', sample)
with open(sample, 'rb') as f:
    img = load_image_bytes(f.read())
print('img shape', img.shape, 'dtype', img.dtype, 'min/max', img.min(), img.max())
ana = init_face_analyzer()
for label, test_img in [('rgb', img), ('bgr', img[..., ::-1])]:
    print('--- testing', label)
    faces = ana.get(test_img)
    print(label, 'faces len', len(faces))
    for i, face in enumerate(faces):
        print('face', i, 'bbox', getattr(face, 'bbox', None), 'det_score', getattr(face, 'det_score', None), 'embedding_len', len(getattr(face, 'embedding', [])))

from PIL import Image
for scale in [2, 4, 6]:
    new_img = np.asarray(Image.fromarray(img).resize((img.shape[1]*scale, img.shape[0]*scale)))
    print('--- testing scale', scale, 'shape', new_img.shape)
    faces = ana.get(new_img)
    print('scaled', scale, 'faces len', len(faces))
    for i, face in enumerate(faces):
        print('face', i, 'bbox', getattr(face, 'bbox', None), 'det_score', getattr(face, 'det_score', None), 'embedding_len', len(getattr(face, 'embedding', [])))
