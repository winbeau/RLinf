"""CPU checks of chain2_reads (task 014): DreamZero's 5 x 10 near-frame reads. Run with this folder on PYTHONPATH:

    PYTHONPATH=evaluation/action_eval <python with numpy + torch + pytest> -m pytest evaluation/action_eval/test_chain2_reads.py
"""

import math

import numpy as np
import pytest

import chain2_reads as c2

BODY = [1 * 10 + 1, 1 * 10 + 2, 2 * 10 + 1, 2 * 10 + 2, 2 * 10 + 3, 3 * 10 + 2]  # 6 agentview tokens
TIPS = [3 * 10 + 6, 3 * 10 + 8]
SINK = [40]


def test_cover_crop_and_cells():
    xy = np.array([[0.0, 0.0], [5.0, 5.0], [6.0, 6.0], [128.0, 128.0], [248.0, 248.0], [249.0, 10.0], [np.nan, 3.0]])
    m = c2.cover(xy)
    # (0, 0) and (5, 5) fall outside the 243 crop starting at 6; 6.5 / 48.6 -> cell 0; 128.5 - 6 = 122.5 -> cell 2; 248.5 - 6 = 242.5 -> 4
    assert m[0, 0] and m[2, 2] and m[4, 4] and int(m.sum()) == 3


def test_dilate_and_body_cap():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    assert int(c2.dilate(m).sum()) == 9
    m[0, 0] = m[4, 4] = True
    assert int(c2.dilate(m).sum()) == 9 + 4 + 4 - 2  # (1, 1) and (3, 3) are shared with the centre patch


@pytest.mark.parametrize("read,x", (("full", 0.0), ("bg", 0.5), ("bg", 1.0), ("bd", 0.5), ("bd", 1.0), ("bdnb", 1.0)))
def test_agentview_reads(read, x):
    rows, bias, info = c2.rows_bias({"read": read, "x": x}, BODY, TIPS, SINK, seed=7)
    b = dict(zip(rows, bias))
    a = [j for j in rows if j in c2.AG and j not in SINK]
    g = [j for j in c2.AG if j not in BODY and j not in SINK]  # 18
    assert set(TIPS) | set(SINK) <= set(rows) and set(c2.WR) <= set(rows)  # the wrist is read whole
    if read in ("full", "bg"):
        assert set(BODY) <= set(a) and len(set(g) - set(a)) == int(math.floor(x * 18 + 0.5))
    else:
        assert set(g) <= set(a) and len(set(BODY) - set(a)) == int(math.floor(x * 6 + 0.5))
    la = 0.0 if read == "bdnb" else math.log(25 / len(a))
    assert all(b[j] == pytest.approx(math.log(2) + la) for j in a)
    assert all(b[j] == pytest.approx(math.log(2)) for j in TIPS + SINK + [w for w in c2.WR if w not in TIPS])


def test_nested_drops_and_seed():
    r50 = set(c2.rows_bias({"read": "bg", "x": 0.5}, BODY, TIPS, SINK, seed=3)[0])
    r100 = set(c2.rows_bias({"read": "bg", "x": 1.0}, BODY, TIPS, SINK, seed=3)[0])
    assert r100 <= r50
    assert set(c2.rows_bias({"read": "bg", "x": 0.5}, BODY, TIPS, SINK, seed=4)[0]) != r50


@pytest.mark.parametrize("n", (4, 8, 23))
def test_wrist_reads(n):
    pool = [j for j in c2.WR if j not in TIPS]
    rows, bias, _ = c2.rows_bias({"read": "s", "n": n}, BODY, TIPS, SINK, seed=5)
    b = dict(zip(rows, bias))
    w = [j for j in rows if j in pool]
    assert len(w) == n and set(BODY) <= set(rows) and all(b[j] == pytest.approx(math.log(2) + math.log(25 / 6)) for j in BODY)
    assert all(b[j] == pytest.approx(math.log(2) + math.log(23 / n)) for j in w)
    if n < 23:
        rows, bias, _ = c2.rows_bias({"read": "c", "n": n}, BODY, TIPS, SINK, seed=5)
        w = [j for j in rows if j in pool]
        d2 = lambda a, b_: (a // 10 - b_ // 10) ** 2 + (a % 10 - b_ % 10) ** 2  # noqa: E731
        out = [p for p in pool if p not in w]
        assert len(w) == n and any(max(d2(c, j) for j in w) <= min(d2(c, p) for p in out) for c in w)


def test_bad_specs():
    with pytest.raises(ValueError):
        c2.rows_bias({"read": "zz"}, BODY, TIPS, SINK, 0)
    with pytest.raises(ValueError):
        c2.rows_bias({"read": "s", "n": 4}, [7], TIPS, SINK, 0)  # 7 is a wrist token
