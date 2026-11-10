"""chain2-curves (CVPR'27 Sparse-rethinking, task 014): the last step's near-frame reads of DreamZero's R arm, built per call from the
agentview body mask (joint values + calibration, ``body_fk``), on DreamZero's 5 x 10 near-frame grid (agentview columns 0-4, wrist 5-9).

DreamZero's view of a 256 x 256 observation: torchvision ``CenterCrop(243)`` (offset round(6.5) = 6), resize, 5 x 5 tokens per view, so
a token covers 243 / 5 = 48.6 observation pixels. The body set is the cover of the projected surface points, dilated 3 x 3 inside the
view unless the dilated set would hold >= 80 % of the 25 agentview tokens (then not dilated).

Reads (``spec["read"]``), the wrist fixed for the agentview curves and the agentview fixed for the wrist curves:

- ``full`` / ``bg`` / ``bd`` / ``bdnb`` (wrist: fingertips + every other wrist token): agentview background G = agentview - body -
  sinks off the body. ``bg`` reads the body and G without its first round(x |G|) (half up) in a random order drawn from the
  generator 3 + the call's seed; ``bd`` reads G, the sinks inside the body and the body without its first round(x |B'|) (B' = body -
  sinks, same generator); ``full`` = ``bg`` at x = 0. Every agentview read except the sinks off the body: key + log(25 / its count)
  (``bdnb``: none).
- ``s`` / ``c`` (agentview: the body, key + log(25 / |body|)): ``s`` n wrist tokens stratified over the wrist view's quadrants (rows 0-2
  / 3-4 x columns 0-2 / 3-4, largest remainder, ties to the earlier stratum), uniform within a stratum (generator 1); ``c`` the n
  nearest (squared Euclidean on the wrist grid, ties by token) to a centre drawn from the pool with the generator 1 + the call's seed.
  Key + log(pool / n).
- The sinks off the body and the fingertips: no group bias. Every key + log 2 (the block's 100 tokens over the near frame's 50).
"""

from __future__ import annotations

import math

import numpy as np

RES, CROP0, CROP, GRID, WIDTH, HALF = 256, 6, 243, 5, 10, 5
AG = [r * WIDTH + c for r in range(GRID) for c in range(HALF)]
WR = [r * WIDTH + HALF + c for r in range(GRID) for c in range(HALF)]
READS = ("full", "bg", "bd", "bdnb", "s", "c")
LG2 = math.log(2.0)


def cover(xy: np.ndarray) -> np.ndarray:
    """Observation pixels (x from the left, y from the top, 256 x 256 as the benchmark returns it) -> 5 x 5 bool tokens of DreamZero's
    centre-cropped view that some pixel lands in."""
    out = np.zeros((GRID, GRID), dtype=bool)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    cell = CROP / GRID
    u, v = (xy[:, 0] + 0.5 - CROP0) / cell, (xy[:, 1] + 0.5 - CROP0) / cell
    keep = (u >= 0) & (u < GRID) & (v >= 0) & (v < GRID)
    out[np.floor(v[keep]).astype(int), np.floor(u[keep]).astype(int)] = True
    return out


def dilate(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            sh = np.zeros_like(m)
            rs, re_ = max(dr, 0), GRID + min(dr, 0)
            cs, ce = max(dc, 0), GRID + min(dc, 0)
            sh[rs:re_, cs:ce] = m[rs - dr:re_ - dr, cs - dc:ce - dc]
            out |= sh
    return out


def body(q: np.ndarray, calib: dict) -> dict:
    """The call's agentview body set on the DreamZero grid (near-frame indices r * 10 + c) from joint values and the calibration."""
    import body_fk as bf  # noqa: PLC0415 - next to this file; the adapter puts its folder on sys.path

    cov = cover(bf.project(bf.world_points(q, calib), calib["camera"]))
    dil = dilate(cov)
    use = dil if int(dil.sum()) < 0.8 * GRID * GRID else cov
    rows = lambda m: [int(r) * WIDTH + int(c) for r, c in zip(*np.nonzero(m))]  # noqa: E731
    return {"body": rows(use), "cover": rows(cov), "dilated": bool(use is dil)}


def spread(pool: list, n: int) -> list:
    """``n`` of the wrist ``pool`` stratified over the view's quadrants (largest remainder), uniform within (generator 1)."""
    import torch  # noqa: PLC0415

    n = min(int(n), len(pool))
    key = lambda j: int((j // WIDTH) >= 3) * 2 + int(((j % WIDTH) - HALF) >= 3)  # noqa: E731
    strata = [[j for j in pool if key(j) == k] for k in range(4)]
    strata = [s for s in strata if s]
    quota = [len(s) * n / len(pool) for s in strata]
    take = [int(math.floor(x)) for x in quota]
    for i in sorted(range(len(strata)), key=lambda i: (-(quota[i] - take[i]), i))[:n - sum(take)]:
        take[i] += 1
    g = torch.Generator().manual_seed(1)
    out = []
    for s, t in zip(strata, take):
        out += [s[i] for i in torch.randperm(len(s), generator=g)[:t].tolist()]
    return sorted(out)


def clump(pool: list, n: int, seed: int) -> list:
    import torch  # noqa: PLC0415

    g = torch.Generator().manual_seed(1 + int(seed))
    centre = pool[int(torch.randint(len(pool), (1,), generator=g))]
    d = sorted((((j // WIDTH) - (centre // WIDTH)) ** 2 + ((j % WIDTH) - (centre % WIDTH)) ** 2, j) for j in pool)
    return sorted(j for _, j in d[:n])


def rows_bias(spec: dict, body_rows: list, tips: list, sinks: list, seed: int) -> tuple[list, list, dict]:
    """(sorted near-frame rows, their key biases, info) of one arm for one call (``seed`` = the call's noise seed)."""
    import torch  # noqa: PLC0415

    read = str(spec["read"])
    if read not in READS:
        raise ValueError(f"chain2 read must be one of {READS}, got {read!r}")
    b = sorted({int(j) for j in body_rows})
    if not b or any(j not in AG for j in b):
        raise ValueError(f"chain2: the body set must be non-empty agentview tokens, got {b}")
    tip = sorted(int(j) for j in tips)
    sink = [int(s) for s in sinks if int(s) not in b]
    pool_w = [j for j in WR if j not in tip]
    info: dict = {"read": read}
    if read in ("full", "bg", "bd", "bdnb"):
        x = 0.0 if read == "full" else float(spec["x"])
        g_bg = [j for j in AG if j not in b and j not in sink]
        b_sink = [j for j in b if j in {int(s) for s in sinks}]
        b_del = [j for j in b if j not in b_sink]
        pool = g_bg if read in ("full", "bg") else b_del
        m = int(math.floor(x * len(pool) + 0.5))
        drop: list = []
        if m:
            g = torch.Generator().manual_seed(3 + int(seed))
            drop = [pool[i] for i in torch.randperm(len(pool), generator=g)[:m].tolist()]
        kept = [j for j in pool if j not in drop]
        a_read = sorted(b + kept) if read in ("full", "bg") else sorted(g_bg + b_sink + kept)
        lg_a = 0.0 if read == "bdnb" else math.log(len(AG) / len(a_read))
        wp = list(pool_w)
        info.update(x=x, pool=len(pool), dropped=m)
    else:
        n = int(spec["n"])
        a_read, lg_a = b, math.log(len(AG) / len(b))
        wp = spread(pool_w, n) if read == "s" else clump(pool_w, n, seed)
        info.update(n=n)
    lg_w = math.log(len(pool_w) / len(wp))
    bias = {}
    for group, lg in ((a_read, lg_a), (sink, 0.0), (tip, 0.0), (wp, lg_w)):
        for j in group:
            if j in bias:
                raise RuntimeError(f"chain2: token {j} in two groups")
            bias[j] = LG2 + lg
    rows = sorted(bias)
    info.update(agent_read=len(a_read), wrist_read=len(wp), body=b, sinks_off_body=sink, tips=tip)
    return rows, [bias[j] for j in rows], info
