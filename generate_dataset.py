"""Generate a large procedural vessel dataset with parallel workers.

Sample i (0-based) uses version VERSIONS[i % 5] (even split across v1-v5), seed i + 1 and palette (i // 5) % 3
(even palette split within each version). Output, sharded 1,000 samples per folder:

    <out>/images/<i // 1000:04d>/<i:07d>.webp   colored input, lossy WebP (quality 95)
    <out>/masks/<i // 1000:04d>/<i:07d>.png     binary mask, 1-bit PNG (lossless; vessel = 1)
    <out>/manifest.csv                          index, version, seed, shape, width, height, palette, vessel_fraction

Pixels outside the canvas shape are black in the input and 0 in the mask; the canvas (FOV) is recovered in the
loader from `shape_mask(size, shape)` using the manifest.

    python generate_dataset.py --out /path/to/procedural_1M -n 1000000 --workers 30
    python generate_dataset.py --out /path/to/procedural_1M -n 2000 --start 1000000   # extra samples, e.g. validation
    python generate_dataset.py --out /path/to/procedural_hard --res 2 --density 2 --bg-variation 0.15 -n 100000   # harder
Re-running skips samples whose files already exist.
"""

import argparse
import csv
import math
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import generate_vessels as g  # noqa: E402

VERSIONS = list(g.CONFIGS)
FIELDS = ["index", "version", "seed", "shape", "width", "height", "palette", "vessel_fraction"]


def paths(out, i):
    sub = f"{i // 1000:04d}"
    return out / "images" / sub / f"{i:07d}.webp", out / "masks" / sub / f"{i:07d}.png"


def configure_worker(res, density, bg_variation):
    g.RES, g.DENSITY, g.BG_VARIATION = res, density, bg_variation


def make(args):
    out, i, quality = args
    version, seed, palette = VERSIONS[i % len(VERSIONS)], i + 1, (i // len(VERSIONS)) % len(g.PALETTES)
    c = g.CONFIGS[version](seed)
    shape, (w, h) = c.get("shape", "square"), c.get("size", (g.SIZE, g.SIZE))
    img_p, msk_p = paths(out, i)
    inside = g.shape_mask((w, h), shape).astype(bool)
    if img_p.exists() and msk_p.exists():
        with Image.open(img_p) as image, Image.open(msk_p) as labels:
            if image.size != (w, h) or labels.size != (w, h):
                raise ValueError(f"sample {i} dimensions differ from the requested settings; use a new output directory")
            mask = np.asarray(labels, dtype=np.uint8)
    else:
        mask = g.generate(**c)
        inp = g.render_input(np.random.default_rng([9000, seed]), mask, inside, palette)
        msk_p.parent.mkdir(parents=True, exist_ok=True)
        img_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(bool)).save(msk_p, optimize=True)
        tmp = img_p.with_suffix(".tmp.webp")
        Image.fromarray(inp).save(tmp, quality=quality, method=4)
        tmp.replace(img_p)  # image written last and atomically: its presence marks a complete sample
    return [i, version, seed, shape, w, h, palette, round(float(mask[inside].mean()), 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=1_000_000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--res", type=float, default=1.0, help="canvas size multiplier (generate_vessels.RES)")
    ap.add_argument("--density", type=float, default=1.0, help="vessel/branch count multiplier (generate_vessels.DENSITY)")
    ap.add_argument("--bg-variation", type=float, default=0.0, help="background colour blotch amplitude (generate_vessels.BG_VARIATION)")
    a = ap.parse_args()
    if a.n < 0 or a.start < 0 or a.workers < 1 or not 1 <= a.quality <= 100:
        ap.error("n and start must be nonnegative, workers must be positive, and quality must be between 1 and 100")
    if not math.isfinite(a.res) or a.res < 1 / 320:
        ap.error("res must be finite and at least 1/320")
    if not math.isfinite(a.density) or a.density <= 0:
        ap.error("density must be positive and finite")
    if not math.isfinite(a.bg_variation) or a.bg_variation < 0:
        ap.error("bg-variation must be nonnegative and finite")
    configure_worker(a.res, a.density, a.bg_variation)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(out, i, a.quality) for i in range(a.start, a.start + a.n)]
    rows = []
    with Pool(a.workers, initializer=configure_worker, initargs=(a.res, a.density, a.bg_variation)) as pool:
        for k, row in enumerate(pool.imap(make, jobs, chunksize=16), 1):
            rows.append(row)
            if k % 10000 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)}", flush=True)
    with open(out / a.manifest, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(FIELDS)
        wr.writerows(rows)


if __name__ == "__main__":
    main()
