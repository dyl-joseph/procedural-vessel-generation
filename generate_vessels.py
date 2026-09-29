import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from pathlib import Path

SIZE = 512
OUT = Path(__file__).parent / "examples"


def grow_tree(rng, n_attractors, n_roots, step=5.0, influence=75.0, kill=22.0, momentum=0.35, noise=0.25,
              curl=0.14, n_trunks=0, branch_decay=1.0, arc=0.0, max_children=3, size=(SIZE, SIZE)):
    w, h = size
    margin = 8
    attractors = rng.uniform([margin, margin], [w - margin, h - margin], (n_attractors, 2))
    if n_trunks:
        origin = np.array([rng.uniform(0.55, 0.85), rng.uniform(0.3, 0.7)]) * [w, h]
        pos = [origin]
        parent = [-1]
        direction = [np.zeros(2)]
        arcsign = [0.0]
        angles = rng.uniform(0, 2 * np.pi) + np.arange(n_trunks) * 2 * np.pi / n_trunks + rng.normal(0, 0.25, n_trunks)
        for ang in angles:
            d = np.array([np.cos(ang), np.sin(ang)])
            pos.append(origin + step * d)
            parent.append(0)
            direction.append(d)
            arcsign.append(rng.choice([-1.0, 1.0]))
        n_init = len(pos)
    else:
        edge = rng.integers(0, 4, n_roots)
        roots = []
        for e in edge:
            t = rng.uniform(0.15, 0.85)
            roots.append([(t * w, 0), (t * w, h), (0, t * h), (w, t * h)][e])
        pos = [np.array(r, float) for r in roots]
        parent = [-1] * n_roots
        direction = []
        for p in pos:
            d = np.array([w / 2, h / 2]) - p
            direction.append(d / np.linalg.norm(d))
        arcsign = [0.0] * n_roots
        n_init = n_roots
    turn = [0.0] * n_init
    order = [0] * n_init
    n_children = [0] * n_init
    for p in parent:
        if p >= 0:
            n_children[p] += 1

    for _ in range(600):
        if len(attractors) == 0:
            break
        tree = cKDTree(np.array(pos))
        dist, nearest = tree.query(attractors, distance_upper_bound=influence)
        valid = np.isfinite(dist)
        if not valid.any():
            break
        pull = {}
        for a, n in zip(attractors[valid], nearest[valid]):
            v = a - pos[n]
            pull.setdefault(n, []).append(v / (np.linalg.norm(v) + 1e-9))
        for n, vs in pull.items():
            if n_children[n] >= max_children:
                continue
            o = order[n] + (1 if n_children[n] > 0 else 0)
            scale = branch_decay ** o
            d = np.mean(vs, axis=0)
            d = d / (np.linalg.norm(d) + 1e-9) + momentum * direction[n] + noise * scale * rng.normal(size=2)
            d /= np.linalg.norm(d)
            sgn = arcsign[n] if n_children[n] == 0 else 0.0
            t = 0.85 * turn[n] + curl * scale * rng.normal() + arc * sgn
            c, s_ = np.cos(t), np.sin(t)
            d = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])
            new = pos[n] + step * d
            if not (0 <= new[0] < w and 0 <= new[1] < h):
                continue
            pos.append(new)
            parent.append(n)
            direction.append(d)
            turn.append(t)
            arcsign.append(sgn)
            order.append(o)
            n_children.append(0)
            n_children[n] += 1
        d_att, _ = cKDTree(np.array(pos)).query(attractors)
        attractors = attractors[d_att > kill]
    return np.array(pos), np.array(parent)


def rot(d, a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])


def edge_start(rng, size, inset=0.0):
    w, h = size
    t, e = rng.uniform(0.2, 0.8), rng.uniform(0, inset)
    p = np.array([(t * w, 1 + e * h), (t * w, (1 - e) * h - 2), (1 + e * w, t * h), ((1 - e) * w - 2, t * h)][rng.integers(0, 4)], float)
    d = np.array([w / 2, h / 2]) - p
    return p, rot(d / np.linalg.norm(d), rng.uniform(-0.5, 0.5))


def walk(rng, pos, parent, start, d, length, step, turn_sd, decay=0.9, bend=0.0, bend_len=1e9, pull=0.0, stop_wall=False, size=(SIZE, SIZE)):
    w, h = size
    d0, t, prev, path = d.copy(), 0.0, start, [start]
    for i in range(int(length / step)):
        t = decay * t + turn_sd * rng.normal()
        d = rot(d, t + bend * np.exp(-i * step / bend_len))
        if pull:
            d = d + pull * d0
            d /= np.linalg.norm(d)
        ahead = pos[prev] + 30 * d
        m = 0.06 * min(w, h)
        if not (m <= ahead[0] < w - m and m <= ahead[1] < h - m):
            if stop_wall:
                break
            c = np.array([w / 2, h / 2]) - pos[prev]
            d = d + 0.15 * c / np.linalg.norm(c)
            d /= np.linalg.norm(d)
        new = pos[prev] + step * d
        if not (0 <= new[0] < w and 0 <= new[1] < h):
            break
        pos.append(new)
        parent.append(prev)
        prev = len(pos) - 1
        path.append(prev)
    return path


def heading(pos, path, k):
    a, b = path[max(k - 1, 0)], path[k]
    d = pos[b] - pos[a] if a != b else pos[path[min(k + 1, len(path) - 1)]] - pos[b]
    return d / (np.linalg.norm(d) + 1e-9)


def build_coronary(rng, size, trunk_len, n_side, side_len, fork_p, hook_p):
    s = min(size) / SIZE
    for _ in range(20):
        pos, parent = [], []
        p, d = edge_start(rng, size, inset=0.3)
        pos.append(p)
        parent.append(-1)
        walk_ = lambda start, d, length, **kw: walk(rng, pos, parent, start, d, length, size=size, **kw)
        bend_len = rng.uniform(60, 140) * s
        trunk = walk_(0, d, trunk_len * s, step=3, turn_sd=0.003, decay=0.95, stop_wall=True,
                      bend=rng.choice([-1, 1]) * rng.uniform(1.0, 2.2) * 3 / bend_len, bend_len=bend_len)
        if len(trunk) * 3 >= 0.6 * trunk_len * s:
            break

    def branch(k_path, k, length, depth):
        side = rng.choice([-1, 1])
        path = walk_(k_path[k], rot(heading(pos, k_path, k), side * rng.uniform(0.45, 1.0)), length,
                     step=3, turn_sd=0.015, decay=0.9, bend=side * rng.uniform(0, 0.008), bend_len=length)
        if depth < 2 and len(path) > 6:
            for _ in range(int(rng.random() < fork_p) + int(rng.random() < fork_p / 2)):
                branch(path, int(rng.uniform(0.3, 0.9) * len(path)), length * rng.uniform(0.4, 0.7), depth + 1)

    n = len(trunk)
    if n < 10:
        return np.array(pos), np.array(parent)
    for k in np.sort(rng.integers(int(0.5 * n), n - 1, n_side)):
        branch(trunk, k, side_len * s * rng.uniform(0.5, 1.3), 0)
    for _ in range(rng.integers(1, 3)):
        branch(trunk, n - 1, side_len * s * rng.uniform(0.4, 0.9), 1)
    if rng.random() < hook_p:
        k = int(rng.uniform(0.05, 0.15) * n)
        side = rng.choice([-1, 1])
        walk_(trunk[k], rot(heading(pos, trunk, k), side * 0.8), rng.uniform(30, 70) * s,
              step=2, turn_sd=0.01, bend=side * rng.uniform(0.05, 0.1), bend_len=1e9)
    return np.array(pos), np.array(parent)


def build_fan(rng, size, n_main, main_len, split_p, twig_p):
    s = min(size) / SIZE
    pos, parent = [], []
    p, d = edge_start(rng, size)
    pos.append(p)
    parent.append(-1)
    walk_ = lambda start, d, length, **kw: walk(rng, pos, parent, start, d, length, size=size, **kw)
    trunk = walk_(0, d, rng.uniform(60, 120) * s, step=6 * s, turn_sd=0.03, decay=0.5,
                  bend=rng.choice([-1, 1]) * rng.uniform(0.06, 0.14), bend_len=40 * s)
    if len(trunk) < 4:
        return np.array(pos), np.array(parent)
    base = heading(pos, trunk, len(trunk) - 1)

    def twig(path, k):
        side = rng.choice([-1, 1])
        walk_(path[k], rot(heading(pos, path, k), side * rng.uniform(0.6, 1.3)), rng.uniform(15, 50) * s,
              step=rng.uniform(8, 14) * s, turn_sd=0.35, decay=0.0)

    def strand(start, d, length, depth):
        path = walk_(start, d, length, step=rng.uniform(14, 28) * s, turn_sd=0.12, decay=0.0, pull=0.35)
        for k in range(1, len(path)):
            if depth < 2 and rng.random() < split_p:
                strand(path[k], rot(heading(pos, path, k), rng.choice([-1, 1]) * rng.uniform(0.15, 0.4)),
                       length * rng.uniform(0.4, 0.7), depth + 1)
            if rng.random() < twig_p:
                twig(path, k)

    starts = rng.integers(int(0.5 * len(trunk)), len(trunk), n_main)
    for i, a in enumerate(np.linspace(-0.35, 0.35, n_main) + rng.normal(0, 0.08, n_main)):
        strand(trunk[starts[i]], rot(base, a), main_len * s * rng.uniform(0.7, 1.2), 0)
    return np.array(pos), np.array(parent)


def colonize(rng, pos, parent, attractors, step, influence, kill, momentum, noise, size, max_children=3, open_p=1.0):
    w, h = size
    direction = [np.zeros(2) if p < 0 else (pos[i] - pos[p]) / (np.linalg.norm(pos[i] - pos[p]) + 1e-9)
                 for i, p in enumerate(parent)]
    n_children = [0] * len(pos)
    for p in parent:
        if p >= 0:
            n_children[p] += 1
    for i in range(len(pos)):
        if n_children[i] and rng.random() > open_p:
            n_children[i] = max_children
    for _ in range(400):
        d_att, _ = cKDTree(np.array(pos)).query(attractors)
        attractors = attractors[d_att > kill]
        if len(attractors) == 0:
            break
        dist, nearest = cKDTree(np.array(pos)).query(attractors, distance_upper_bound=influence)
        valid = np.isfinite(dist)
        pull = {}
        for a, n in zip(attractors[valid], nearest[valid]):
            v = a - pos[n]
            pull.setdefault(n, []).append(v / (np.linalg.norm(v) + 1e-9))
        if not any(n_children[n] < max_children for n in pull):
            influence *= 1.4
            continue
        for n, vs in pull.items():
            if n_children[n] >= max_children:
                continue
            d = np.mean(vs, axis=0)
            d = d / (np.linalg.norm(d) + 1e-9) + momentum * direction[n] + noise * rng.normal(size=2)
            d /= np.linalg.norm(d)
            new = pos[n] + step * d
            if not (0 <= new[0] < w and 0 <= new[1] < h):
                continue
            pos.append(new)
            parent.append(n)
            direction.append(d)
            n_children.append(0)
            n_children[n] += 1
    return pos, parent


def build_retina(rng, size, density, arcade_len, n_nasal, macula_r, step, influence, kill, noise, open_p):
    w, h = size
    s = min(size) / SIZE
    side = rng.choice([-1, 1])
    disc = np.array([w / 2 + side * rng.uniform(0.25, 0.35) * w, rng.uniform(0.4, 0.6) * h])
    temporal = np.array([-side, 0.0])
    macula = disc + temporal * rng.uniform(0.3, 0.4) * w
    pos, parent = [], []
    walk_ = lambda start, d, length, **kw: walk(rng, pos, parent, start, d, length, size=size, **kw)
    for _ in range(2):
        root = len(pos)
        pos.append(disc + rng.normal(0, 3 * s, 2))
        parent.append(-1)
        for vert in (-1, 1):
            d0 = rot(np.array([0.0, vert]), -vert * side * rng.uniform(0.2, 0.5))
            sign = np.sign(d0[0] * temporal[1] - d0[1] * temporal[0])
            bend_len = rng.uniform(0.2, 0.35) * min(size)
            walk_(root, d0, arcade_len * w, step=3 * s, turn_sd=0.01, decay=0.9, stop_wall=True,
                  bend=sign * rng.uniform(0.9, 1.3) * 3 * s / bend_len, bend_len=bend_len)
        for a in rng.uniform(-1.3, 1.3, n_nasal):
            walk_(root, rot(-temporal, a), rng.uniform(0.1, 0.25) * w, step=3 * s, turn_sd=0.02, decay=0.85, stop_wall=True)
    attractors = rng.uniform([0, 0], [w, h], (int(density * w * h / SIZE**2), 2))
    attractors = attractors[np.linalg.norm(attractors - macula, axis=1) > macula_r * min(size)]
    attractors = attractors[np.linalg.norm(attractors - disc, axis=1) > 12 * s]
    roots = [i for i, p in enumerate(parent) if p < 0] + [len(pos)]
    trees = [(list(range(roots[k], roots[k + 1]))) for k in range(2)]
    halves = rng.random(len(attractors)) < 0.5
    out_pos, out_parent = [], []
    for k, idx in enumerate(trees):
        offset = len(out_pos)
        tp = [pos[i] for i in idx]
        tpar = [parent[i] - idx[0] if parent[i] >= 0 else -1 for i in idx]
        tp, tpar = colonize(rng, tp, tpar, attractors[halves == bool(k)], step * s, influence * s, kill * s,
                            0.5, noise, size, open_p=open_p)
        out_pos += tp
        out_parent += [p + offset if p >= 0 else -1 for p in tpar]
    return np.array(out_pos), np.array(out_parent)


def roughen(rng, mask, amount):
    f = gaussian_filter(mask.astype(np.float32), 0.8)
    n = gaussian_filter(rng.standard_normal(mask.shape).astype(np.float32), 1.0)
    out = (f + amount * n / n.std()) > 0.5
    return ((out | (f > 0.9)) & (f > 0.1)).astype(np.uint8)


BUILDERS = {"coronary": build_coronary, "fan": build_fan, "retina": build_retina}


def add_sprouts(rng, pos, parent, prob, length=(8, 30), step=3.0, size=(SIZE, SIZE)):
    w, h = size
    pos = list(pos)
    parent = list(parent)
    for i in range(len(pos)):
        if parent[i] < 0 or rng.random() > prob:
            continue
        edge = pos[i] - pos[parent[i]]
        edge /= np.linalg.norm(edge) + 1e-9
        ang = rng.choice([-1, 1]) * rng.uniform(0.9, 1.9)
        d = np.array([np.cos(ang) * edge[0] - np.sin(ang) * edge[1], np.sin(ang) * edge[0] + np.cos(ang) * edge[1]])
        prev, turn = i, 0.0
        for _ in range(int(rng.uniform(*length) / step)):
            turn = 0.7 * turn + 0.12 * rng.normal()
            c, s_ = np.cos(turn), np.sin(turn)
            d = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])
            new = pos[prev] + step * d
            if not (0 <= new[0] < w and 0 <= new[1] < h):
                break
            pos.append(new)
            parent.append(prev)
            prev = len(pos) - 1
    return np.array(pos), np.array(parent)


def murray_radii(parent, root_r, gamma, r_min, rng=None, thick_var=0.0):
    n = len(parent)
    weight = np.zeros(n)
    has_child = np.zeros(n, bool)
    has_child[parent[parent >= 0]] = True
    weight[~has_child] = 1.0
    for i in range(n - 1, -1, -1):
        if parent[i] >= 0:
            weight[parent[i]] += weight[i]
    total = weight[parent < 0].sum()
    radii = root_r * (weight / total) ** (1 / gamma)
    if thick_var:
        log_f = np.zeros(n)
        for i in range(n):
            p = parent[i]
            if p < 0:
                continue
            step_sd = thick_var * (3.0 if has_child[p] and weight[p] > weight[i] * 1.5 else 0.3)
            log_f[i] = 0.9 * log_f[p] + step_sd * rng.normal()
            radii[i] = min(radii[i] * np.exp(log_f[i]), radii[p])
    return np.maximum(radii, r_min)


def rasterize(pos, parent, radii, size=(SIZE, SIZE), ss=3):
    w, h = size
    img = Image.new("1", (w * ss, h * ss), 0)
    draw = ImageDraw.Draw(img)
    for i, p in enumerate(parent):
        r = radii[i] * ss
        x, y = pos[i] * ss
        if p >= 0:
            draw.line([tuple(pos[p] * ss), (x, y)], fill=1, width=max(1, round(2 * r)))
        if r > 1.2:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=1)
    big = np.array(img, dtype=np.float32).reshape(h, ss, w, ss).mean(axis=(1, 3))
    return (big >= 0.5).astype(np.uint8)


def shape_mask(size, shape):
    w, h = size
    if shape not in ("circle", "oval"):
        return np.ones((h, w), np.uint8)
    y, x = np.mgrid[:h, :w]
    return ((((x + 0.5) / w * 2 - 1) ** 2 + ((y + 0.5) / h * 2 - 1) ** 2) <= 1).astype(np.uint8)


def generate(seed, n_attractors=3000, n_roots=2, root_r=7.0, gamma=2.5, r_min=1.5,
             sprout_p=0.0, sprout_len=(8, 30), thick_var=0.0, size=(SIZE, SIZE), shape="square", kind=None, rough=0.0, **grow):
    rng = np.random.default_rng(seed)
    if kind:
        pos, parent = BUILDERS[kind](rng, size, **grow)
    else:
        pos, parent = grow_tree(rng, n_attractors, n_roots, size=size, **grow)
    if sprout_p:
        pos, parent = add_sprouts(rng, pos, parent, sprout_p, sprout_len, size=size)
    radii = murray_radii(parent, root_r, gamma, r_min, rng, thick_var)
    mask = rasterize(pos, parent, radii, size)
    if rough:
        mask = roughen(rng, mask, rough)
    return mask * shape_mask(size, shape)


PALETTES = [((47, 47, 47), (62, 62, 62)), ((210, 86, 62), (148, 48, 30)), ((170, 85, 50), (164, 100, 58))]


def render_input(rng, mask, inside, palette, blur=0.1):
    lo, hi = (np.array(c, np.float32) for c in PALETTES[palette])
    t = gaussian_filter(rng.standard_normal(mask.shape).astype(np.float32), min(mask.shape) / 8)
    t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    span = rng.uniform(0.3, 1.0)
    t = t * span + rng.uniform(0, 1 - span)
    bg = lo + t[..., None] * (hi - lo)
    sigma = blur * 2 * distance_transform_edt(mask).max()
    soft = gaussian_filter(mask.astype(np.float32), sigma)
    contrast = rng.choice([-1, 1]) * rng.uniform(*((0.15, 0.20) if palette == 0 else (0.10, 0.15)))
    img = bg * (1 + contrast * soft[..., None])
    return (np.clip(np.rint(img), 0, 255) * inside[..., None]).astype(np.uint8)


def contact_sheet(tiles, path, mode):
    cols, cell = 4, 512
    rows = -(-len(tiles) // cols)
    sheet = Image.new(mode, (cols * (cell + 4) - 4, rows * (cell + 4) - 4), (128,) * len(mode))
    for i, img in enumerate(tiles):
        img = img.copy()
        img.thumbnail((cell, cell))
        r, c = divmod(i, cols)
        sheet.paste(img, (c * (cell + 4) + (cell - img.width) // 2, r * (cell + 4) + (cell - img.height) // 2))
    sheet.save(path)


def random_canvas(seed):
    rng = np.random.default_rng(2000 + seed)
    shape = str(rng.choice(["circle", "oval", "square", "rectangle"]))
    if shape in ("circle", "square"):
        side = int(rng.integers(384, 769))
        return shape, (side, side)
    while True:
        w, h = (int(v) for v in rng.integers(320, 769, 2))
        if max(w, h) / min(w, h) >= 1.3:
            return shape, (w, h)


def random_config(seed):
    rng = np.random.default_rng(1000 + seed)
    cfg = dict(
        seed=seed,
        n_trunks=int(rng.integers(2, 8)),
        root_r=rng.uniform(3.5, 11),
        r_min=rng.uniform(0.5, 2.2),
        gamma=rng.uniform(1.8, 3.4),
        thick_var=rng.uniform(0.03, 0.09),
        branch_decay=rng.uniform(0.15, 0.6),
    )
    cfg.update(
        branch_decay=rng.uniform(0.5, 0.85),
        n_attractors=int(rng.integers(1000, 2200)),
        step=rng.uniform(4, 6),
        curl=rng.uniform(0.05, 0.2),
        momentum=rng.uniform(0.4, 0.7),
        noise=rng.uniform(0.05, 0.2),
        influence=rng.uniform(45, 80),
        kill=rng.uniform(8, 13),
        arc=rng.uniform(0.005, 0.02),
        r_min=rng.uniform(0.5, 0.9),
        root_r=rng.uniform(4, 7),
        sprout_p=rng.uniform(0.03, 0.12),
        sprout_len=(rng.uniform(6, 12), rng.uniform(25, 60)),
    )
    shape, size = random_canvas(seed)
    density = size[0] * size[1] / SIZE**2 * 0.45
    cfg.update(
        shape=shape,
        size=size,
        n_attractors=int(cfg["n_attractors"] * density),
        kill=cfg["kill"] * 1.7,
        sprout_p=cfg["sprout_p"] * 0.5,
        root_r=cfg["root_r"] * 0.8,
    )
    return cfg


def v1_config(seed):
    rng = np.random.default_rng(4000 + seed)
    shape, size = random_canvas(seed)
    area = size[0] * size[1] / SIZE**2
    return dict(seed=seed, n_attractors=int(rng.integers(30, 111) * area), n_roots=int(rng.integers(1, 4)),
                root_r=rng.uniform(6.0, 8.0), influence=rng.uniform(140, 200), max_children=2,
                shape=shape, size=size)


def v3_config(seed):
    rng = np.random.default_rng(6000 + seed)
    shape, size = random_canvas(seed)
    return dict(seed=seed, kind="coronary", trunk_len=rng.uniform(400, 620), n_side=int(rng.integers(2, 6)),
                side_len=rng.uniform(80, 160), fork_p=rng.uniform(0.3, 0.8), hook_p=0.6,
                root_r=rng.uniform(5.0, 8.0), gamma=rng.uniform(2.6, 3.4), r_min=rng.uniform(1.0, 1.6),
                thick_var=rng.uniform(0.0, 0.04), shape=shape, size=size)


def v4_config(seed):
    rng = np.random.default_rng(7000 + seed)
    shape, size = random_canvas(seed)
    return dict(seed=seed, kind="fan", n_main=int(rng.integers(3, 7)), main_len=rng.uniform(250, 420),
                split_p=rng.uniform(0.03, 0.1), twig_p=rng.uniform(0.08, 0.2),
                root_r=rng.uniform(6.0, 10.0), gamma=rng.uniform(3.5, 4.5), r_min=rng.uniform(1.8, 2.6),
                thick_var=rng.uniform(0.0, 0.05), shape=shape, size=size)


def v5_config(seed):
    rng = np.random.default_rng(8000 + seed)
    shape, size = random_canvas(seed)
    return dict(seed=seed, kind="retina", density=rng.uniform(500, 900), arcade_len=rng.uniform(0.35, 0.55),
                n_nasal=int(rng.integers(2, 4)), macula_r=rng.uniform(0.05, 0.09), step=rng.uniform(4, 5.5),
                influence=rng.uniform(60, 90), kill=rng.uniform(20, 28), noise=rng.uniform(0.25, 0.45),
                open_p=rng.uniform(0.06, 0.12),
                sprout_p=rng.uniform(0.01, 0.03), sprout_len=(6, 18),
                root_r=rng.uniform(5.0, 7.0), gamma=rng.uniform(2.6, 3.2), r_min=rng.uniform(0.8, 1.1),
                thick_var=rng.uniform(0.02, 0.05), rough=rng.uniform(0.12, 0.2), shape=shape, size=size)


CONFIGS = {"v1": v1_config, "v2": random_config, "v3": v3_config, "v4": v4_config, "v5": v5_config}


def sample_plan(name, n):
    if name in CONFIGS:
        return [(name, CONFIGS[name](seed)) for seed in range(1, 13)]
    names = list(CONFIGS)
    versions = np.random.default_rng(5000 + n).permutation([names[i % len(names)] for i in range(n)])
    return [(v, CONFIGS[v](i)) for i, v in enumerate(versions, 1)]


def main():
    parser = argparse.ArgumentParser(description="Generate binary vessel masks (vessels=1, background=0) and colored input images.")
    parser.add_argument("preset", choices=[*CONFIGS, "mix"], help="mix = even split across all versions")
    parser.add_argument("-n", type=int, default=12, help="samples for mix")
    args = parser.parse_args()
    out = OUT / args.preset
    out.mkdir(parents=True, exist_ok=True)
    tiles, inputs = [], []
    for i, (version, c) in enumerate(sample_plan(args.preset, args.n)):
        mask = generate(**c)
        shape = c.get("shape", "square")
        stem = f"{args.preset}_seed{c['seed']}" if args.preset != "mix" else f"mix_{c['seed']:02d}_{version}"
        if "shape" in c:
            stem += f"_{shape}"
        np.save(out / f"{stem}.npy", mask)
        img = Image.fromarray(mask * 255)
        img.save(out / f"{stem}.png")
        inside = shape_mask(c.get("size", (SIZE, SIZE)), shape).astype(bool)
        inp = Image.fromarray(render_input(np.random.default_rng(9000 + c["seed"]), mask, inside, i % len(PALETTES)))
        inp.save(out / f"{stem}_input.png")
        tiles.append(img)
        inputs.append(inp)
        print(stem, mask.shape, np.unique(mask), f"vessel fraction {mask[inside].mean():.3f}", f"palette {i % len(PALETTES)}")
    contact_sheet(tiles, out / "contact_sheet.png", "L")
    contact_sheet(inputs, out / "contact_sheet_input.png", "RGB")

if __name__ == "__main__":
    main()
