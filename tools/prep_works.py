"""Crop and flatten the phone snapshots in 作品/.

Strategy
  * The desk is everything reachable from the image border by flood fill with a
    small colour tolerance. Whatever is left is the sheet of paper, painted area
    included -- this works for white, cream and yellow paper alike.
  * If the whole sheet is inside the frame, warp its quadrilateral flat and keep
    the sheet, so the deckled edge and any seal in the margin survive.
  * If the sheet runs past the frame edge, fall back to warping the painted area
    only, so no desk is left in the picture.

Output: build/crop/NN.jpg (long edge 1600) + report.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "作品"
OUT = ROOT / "build" / "crop"
WORK_EDGE = 900
OUT_EDGE = 1600


def imread(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def order_quad(pts) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.015, 0.02, 0.03, 0.045, 0.06):
        q = cv2.approxPolyDP(hull, eps * peri, True)
        if len(q) == 4:
            return order_quad(q)
    return order_quad(cv2.boxPoints(cv2.minAreaRect(cnt)))


def desk_mask(im: np.ndarray, tol: int = 16) -> np.ndarray:
    h, w = im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    d = (tol, tol, tol)
    step = max(8, min(h, w) // 40)
    seeds = ([(2, y) for y in range(2, h - 2, step)]
             + [(w - 3, y) for y in range(2, h - 2, step)]
             + [(x, 2) for x in range(2, w - 2, step)]
             + [(x, h - 3) for x in range(2, w - 2, step)])
    for s in seeds:
        if mask[s[1] + 1, s[0] + 1] == 0:
            cv2.floodFill(im.copy(), mask, s, 255, d, d,
                          cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
    desk = mask[1:-1, 1:-1].copy()
    desk = cv2.morphologyEx(desk, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return desk


def paint_mask(im: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    m = (((s > 62) & (v > 40)) | (v < 55)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return m


def warp(im: np.ndarray, q: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = q
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(im, cv2.getPerspectiveTransform(q, src),
                               (int(round(w)), int(round(h))), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def bleed(im: np.ndarray, px: int) -> np.ndarray:
    return cv2.copyMakeBorder(im, px, px, px, px, cv2.BORDER_REPLICATE)


def process(path: Path):
    src = imread(path)
    H0, W0 = src.shape[:2]
    k = WORK_EDGE / max(H0, W0)
    sm = cv2.resize(src, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else src
    h, w = sm.shape[:2]

    sheet = (desk_mask(sm) == 0).astype(np.uint8) * 255
    sheet = cv2.morphologyEx(sheet, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no-sheet"
    cnt = max(contours, key=cv2.contourArea)
    sheet = np.zeros((h, w), np.uint8)
    cv2.drawContours(sheet, [cnt], -1, 255, -1)

    touches = bool(sheet[0].any() or sheet[-1].any() or sheet[:, 0].any() or sheet[:, -1].any())
    area_frac = cv2.countNonZero(sheet) / (h * w)
    if area_frac > 0.985:
        touches = True

    if not touches:
        q = quad_from_mask(sheet)
        mode = "sheet"
    else:
        q = quad_from_mask(paint_mask(sm))
        if q is None:
            return None, "no-paint"
        mode = "paint"
    flat = warp(src, q / k)

    # shave the bright fringe the flood fill leaves on the boundary
    pad = max(2, int(min(flat.shape[:2]) * (0.004 if mode == "sheet" else 0.002)))
    ph, pw = flat.shape[:2]
    if ph > pad * 8 and pw > pad * 8:
        flat = flat[pad:ph - pad, pad:pw - pad]
    flat = bleed(flat, 2)

    scale = OUT_EDGE / max(flat.shape[:2])
    if scale < 1:
        flat = cv2.resize(flat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return flat, mode


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    files = sorted(p for p in SRC_DIR.glob("*.jpg") if not p.name.startswith("_"))
    for i, f in enumerate(files, 1):
        flat, mode = process(f)
        if flat is None:
            print(f"{i:02d} {f.name[:10]} FAIL {mode}")
            report.append({"ord": i, "file": f.name, "fail": mode})
            continue
        cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile(
            str(OUT / f"{i:02d}.jpg"))
        r = round(flat.shape[1] / flat.shape[0], 3)
        report.append({"ord": i, "file": f.name, "mode": mode,
                       "size": [flat.shape[1], flat.shape[0]], "ratio": r})
        print(f"{i:02d} {f.name[:10]} {mode:5s} {flat.shape[1]}x{flat.shape[0]} a={r}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
