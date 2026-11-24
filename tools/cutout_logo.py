"""Extract the rabbit from logo.jpg and write a transparent PNG.

The painted background is a flat blue-grey, so it separates cleanly from the
rabbit in Lab space. We keep only background regions that connect to the
border, then keep the largest remaining object and feather its edge.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "logo.jpg"
OUT_DIR = ROOT / "assets"
OUT = OUT_DIR / "rabbit.png"
CHECK = ROOT / "preview" / "_cutout_check.png"

DIST = 20.0


def main() -> None:
    img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {SRC}")
    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    def patch(y0, y1, x0, x1):
        return np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)

    ref = np.median(np.array([
        patch(0, 40, 0, 40), patch(0, 40, w - 40, w),
        patch(h - 40, h, 0, 40), patch(0, 40, w // 2 - 20, w // 2 + 20),
    ]), axis=0)

    near = (np.linalg.norm(lab - ref, axis=2) < DIST).astype(np.uint8)

    # close the paper speckles, then keep only background touching the border
    near = cv2.morphologyEx(
        near, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, labels = cv2.connectedComponents(near, connectivity=8)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    bg = np.isin(labels, list(border)).astype(np.uint8)

    fg = ((1 - bg) * 255).astype(np.uint8)
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise SystemExit("nothing left after masking")
    biggest = max(contours, key=cv2.contourArea)
    solid = np.zeros((h, w), np.uint8)
    cv2.drawContours(solid, [biggest], -1, 255, thickness=cv2.FILLED)

    alpha = cv2.GaussianBlur(solid, (3, 3), 0.6)

    # paint the cut-away pixels with the nearest edge colour: without this the
    # browser's downscaling bleeds pale RGB into the rim and leaves a halo
    visible = (alpha > 8).astype(np.uint8)
    kernel3 = np.ones((3, 3), np.uint8)
    bgr = img.copy()
    for _ in range(16):
        grown = cv2.dilate(bgr, kernel3)
        bgr = np.where((visible == 0)[..., None], grown, bgr)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    ys, xs = np.where(alpha > 8)
    pad = 8
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    cropped = rgba[y0:y1, x0:x1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", cropped)[1].tofile(str(OUT))

    (ROOT / "preview").mkdir(parents=True, exist_ok=True)
    ch, cw = cropped.shape[:2]
    sheet = np.zeros((ch, cw * 3, 3), np.uint8)
    for i, hexcolor in enumerate(("#11264f", "#ffffff", "#f58f98")):
        bgr = tuple(int(hexcolor[j:j + 2], 16) for j in (5, 3, 1))
        patch_bg = np.full((ch, cw, 3), bgr, np.uint8)
        a = cropped[:, :, 3:4].astype(np.float32) / 255.0
        sheet[:, i * cw:(i + 1) * cw] = (
            cropped[:, :, :3].astype(np.float32) * a
            + patch_bg.astype(np.float32) * (1 - a)).astype(np.uint8)
    cv2.imencode(".png", sheet)[1].tofile(str(CHECK))

    print(f"wrote {OUT} ({cw}x{ch}) and {CHECK}")


if __name__ == "__main__":
    main()
