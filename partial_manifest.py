"""Write a manifest for a completed prefix using the samples' actual geometry."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import generate_vessels as g
from generate_dataset import FIELDS, VERSIONS, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("n", type=int)
    parser.add_argument("name", help="output CSV filename relative to out")
    args = parser.parse_args()
    if args.n < 0:
        parser.error("n must be nonnegative")
    rows = []
    for index in range(args.n):
        image_path, mask_path = paths(args.out, index)
        if not image_path.is_file() or not mask_path.is_file():
            parser.error(f"sample {index} is incomplete")
        with Image.open(image_path) as image, Image.open(mask_path) as labels:
            if image.size != labels.size:
                parser.error(f"sample {index} image/mask dimensions differ")
            width, height = image.size
            mask = np.asarray(labels) > 0
        version, seed = VERSIONS[index % len(VERSIONS)], index + 1
        shape, _ = g.random_canvas(seed)
        inside = g.shape_mask((width, height), shape).astype(bool)
        rows.append([index, version, seed, shape, width, height,
                     (index // len(VERSIONS)) % len(g.PALETTES), round(float(mask[inside].mean()), 5)])
    output = args.out / args.name
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print("wrote", output, len(rows))


if __name__ == "__main__":
    main()
