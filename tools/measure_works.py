"""Measure paper vs. painted area for each snapshot, to decide the crop.

For every file in 作品/ we locate the white sheet (bright, low-saturation, far
brighter than the grey desk), warp it flat, then measure where the painted
area sits inside it. Output goes to build/measure/.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "作品"
OUT = ROOT / "build" / "measure"
EDGE = 1400


def imread(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def order_quad(pts) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def quad_of(mask: np.ndarray, min_frac: float) -> tuple[np.ndarray | None, str]:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = [c for c in contours if cv2.contourArea(c) > min_frac * h * w]
    if not keep:
        return None, "none"
    pts = np.vstack(keep)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.06):
        q = cv2.approxPolyDP(hull, eps * peri, True)
        if len(q) == 4:
            return order_quad(q), "quad"
    return order_quad(cv2.boxPoints(cv2.minAreaRect(pts))), "rect"


def sheet_mask(im: np.ndarray) -> np.ndarray:
    """The sheet differs in colour from the grey desk, whatever tint it is."""
    h, w = im.shape[:2]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(6, int(min(h, w) * 0.03))
    strips = np.concatenate([
        lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
        lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    ref = np.median(strips, axis=0)
    dist = np.linalg.norm(lab - ref, axis=2)
    m = (dist > 20).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return m


def paint_mask(im: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    m = (((s > 62) & (v > 40)) | (v < 55)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    return m


def warp(im: np.ndarray, q: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = q
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(im, cv2.getPerspectiveTransform(q, src),
                               (int(round(w)), int(round(h))), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(p for p in SRC.glob("*.jpg") if not p.name.startswith("_"))
    for i, f in enumerate(files, 1):
        im = imread(f)
        h0, w0 = im.shape[:2]
        k = EDGE / max(h0, w0)
        sm = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else im

        pq, how = quad_of(paper_mask(sm), 0.12)
        info = {"ord": i, "file": f.name, "paper": how}
        if pq is None:
            cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(
                str(OUT / f"{i:02d}.jpg"))
            info.update(paper_aspect=None)
            print(f"{i:02d} paper=none")
            rows.append(info)
            continue

        touches = (pq[:, 0].min() < 2 or pq[:, 0].max() > sm.shape[1] - 3
                   or pq[:, 1].min() < 2 or pq[:, 1].max() > sm.shape[0] - 3)
        flat = warp(sm, pq)
        ph, pw = flat.shape[:2]

        pm = paint_mask(flat)
        ys, xs = np.where(pm > 0)
        box = None
        if len(xs):
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        info.update({
            "paper_size": [pw, ph],
            "paper_aspect": round(pw / ph, 3),
            "touches_frame": bool(touches),
            "paint_box": box,
        })
        if box:
            bw, bh = box[2] - box[0], box[3] - box[1]
            info.update({"paint_aspect": round(bw / bh, 3),
                         "paint_share_w": round(bw / pw, 3),
                         "paint_share_h": round(bh / ph, 3)})

        cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(
            str(OUT / f"{i:02d}.jpg"))
        print(f"{i:02d} {f.name[:10]} paper={how} {pw}x{ph} a={info['paper_aspect']} "
              f"edge={touches} paint_a={info.get('paint_aspect')} "
              f"share={info.get('paint_share_w')}x{info.get('paint_share_h')}")
        rows.append(info)

    (OUT / "report.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
