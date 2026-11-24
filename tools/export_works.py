"""Export review-ready files from build/crop/ into assets/works/.

Rotation decisions (judged from the images, see build/crop/_rot.jpg):
  05 and 06 were placed sideways on the desk -> 90 deg clockwise.
Everything else was already upright.

Naming keeps the sort order of 作品/: 001 … 021, so the size grouping stays
stable: 004, 006, 009, 010, 016, 017 are the 8.5 x 20.5 cm long pieces.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build" / "crop"
FULL = ROOT / "assets" / "works"
THUMB = FULL / "th"

ROTATE = {5: cv2.ROTATE_90_CLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}
LONG = 1800
THUMB_LONG = 640


def fit(im: np.ndarray, long_edge: int) -> np.ndarray:
    if max(im.shape[:2]) <= long_edge:
        return im
    k = long_edge / max(im.shape[:2])
    return cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def save(im: np.ndarray, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(stem.with_suffix(".jpg")))
    cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, 86])[1].tofile(str(stem.with_suffix(".webp")))


def main() -> None:
    files = sorted(p for p in SRC.glob("[0-9][0-9].jpg"))
    for p in files:
        n = int(p.stem)
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if n in ROTATE:
            im = cv2.rotate(im, ROTATE[n])
        full = fit(im, LONG)
        save(full, FULL / f"{n:03d}")
        save(fit(im, THUMB_LONG), THUMB / f"{n:03d}")
        print(f"{n:03d} {full.shape[1]}x{full.shape[0]} ratio={full.shape[1] / full.shape[0]:.3f}")


if __name__ == "__main__":
    main()
