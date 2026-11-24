"""Extract the seal from the phone photo and rebuild it as a clean RGBA asset.

The glyph strokes are never redrawn: only the paper is keyed out and the ink
density is mapped to alpha, so the carved characters stay exactly as stamped.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build" / "seal_top.png"
SIZE = 1200
CINNABAR = (206, 62, 46)  # RGB, traditional seal paste red

im = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
up = cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
b, g, r = [c.astype(np.float32) for c in cv2.split(up)]

# ink is far redder than the yellowed paper; density becomes alpha
d = r - b
strength = np.clip((d - 70.0) / 70.0, 0, 1)
strength = cv2.medianBlur((strength * 255).astype(np.uint8), 5).astype(np.float32) / 255
strength = np.clip((strength - 0.12) / 0.88, 0, 1)

ys, xs = np.where(strength > 0.25)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
alpha = strength[y0:y1 + 1, x0:x1 + 1]

# square canvas, seal occupies 86%
side = int(max(alpha.shape) / 0.86)
scale = SIZE / side
alpha = cv2.resize(alpha, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
canvas = np.zeros((SIZE, SIZE), np.float32)
oy, ox = (SIZE - alpha.shape[0]) // 2, (SIZE - alpha.shape[1]) // 2
canvas[oy:oy + alpha.shape[0], ox:ox + alpha.shape[1]] = alpha
canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)

rgba = np.zeros((SIZE, SIZE, 4), np.uint8)
rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = CINNABAR[2], CINNABAR[1], CINNABAR[0]
rgba[:, :, 3] = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)

for out in (ROOT / "assets" / "seal.png", ROOT / "preview" / "assets" / "seal.png"):
    cv2.imencode(".png", rgba)[1].tofile(str(out))
    print("saved", out)

for name, bg in (("seal_on_white", (245, 243, 238)), ("seal_on_navy", (79, 38, 17))):
    base = np.full((SIZE, SIZE, 3), bg, np.float32)
    a = canvas[:, :, None]
    ink = np.array(CINNABAR[::-1], np.float32)
    comp = (ink * a + base * (1 - a)).astype(np.uint8)
    small = cv2.resize(comp, (520, 520), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(ROOT / "build" / (name + ".jpg")))
print("previews written")
