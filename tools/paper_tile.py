"""Build a seamless rice-paper tile from the blank area of the seal photo."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "name.jpg"
TILE = 600
CREAM = np.array([234, 243, 247], np.float32)  # BGR target paper colour

im = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
patch = im[600:1500, 1300:2200]
patch = cv2.resize(patch, (TILE, TILE), interpolation=cv2.INTER_AREA).astype(np.float32)

# remove uneven lighting, keep the fibre detail
blur = cv2.GaussianBlur(patch, (0, 0), 30)
flat = patch / np.maximum(blur, 1.0)
detail = patch - blur

paper = CREAM[None, None, :] * flat + detail * 0.55
paper = np.clip(paper, 0, 255)

# mirror-tile so the texture repeats without a visible seam
top = np.concatenate([paper, paper[:, ::-1]], axis=1)
bottom = top[::-1]
tile = np.concatenate([top, bottom], axis=0).astype(np.uint8)

for out in (ROOT / "assets" / "paper.jpg", ROOT / "preview" / "assets" / "paper.jpg"):
    cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(out))
    print("saved", out)

small = cv2.resize(tile, (520, 520), interpolation=cv2.INTER_AREA)
cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(ROOT / "build" / "paper_preview.jpg"))
