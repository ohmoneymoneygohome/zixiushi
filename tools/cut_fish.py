"""Cut the nine pinned fish paintings out of assets/background.jpg.

The photo is a dark board with nine small paintings pinned on it. We keep the
cards exactly where they are (angles, spacing, pins) and drop the board, so the
group can float on the site's navy. Output: assets/nine_fish.png (RGBA).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "background.jpg"
OUT = ROOT / "assets" / "nine_fish.png"


def imread(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    bgr = imread(SRC)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2].astype(np.int16), hsv[:, :, 1].astype(np.int16)

    # cards are bright and saturated; the board is dark olive
    fg = (((v > 88) & (s > 30)) | (v > 140)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    min_area = 0.004 * h * w
    cards = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    cards.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    print("components kept:", len(cards), "areas:", [int(stats[i, 4]) for i in cards[:12]])

    alpha = np.zeros((h, w), np.uint8)
    for i in cards:
        m = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.01, 0.02, 0.03, 0.045, 0.06):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2)
                break
        if quad is not None:
            cv2.fillConvexPoly(alpha, quad, 255)
        else:
            cv2.drawContours(alpha, [hull], -1, 255, -1)

    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.6)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tofile(str(OUT))

    cov = cv2.countNonZero(alpha) / (h * w)
    print(f"saved {OUT.name} {w}x{h} coverage={cov:.3f}")

    # preview over the site navy so the edge can be judged by eye
    bg = np.zeros_like(bgr)
    bg[:] = (79, 38, 17)
    a = alpha.astype(np.float32)[..., None] / 255.0
    comp = (bgr * a + bg * (1 - a)).astype(np.uint8)
    cv2.imencode(".jpg", comp, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
        str(ROOT / "build" / "nine_fish_check.jpg"))


if __name__ == "__main__":
    main()
