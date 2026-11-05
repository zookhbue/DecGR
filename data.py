"""FDCompCN loader + the split used in the paper.

The dataset ships as a DGL binary file (`datasets/FDCompCN_dataset/comp.dgl`).
DGL is not required: the file is a sequence of records

    [u64 name_len][name][magic 8B][u64 0][u64 1][u32 ndim][u32 dtype]
    [ndim x u64 dims][u64 nbytes][raw little-endian data]

and is parsed directly below (same reader as analysis/dgl_fast.py).

Graph: 5,317 companies, 57 features, 559 fraudulent (10.51 %).
Edges:  three relations (C-I-C invest, C-P-C sale, C-S-C provide); the model uses
        their union, symmetrised (3,675 unique links; 7,489 directed records).

Split (given in the manuscript):
        train 2,658 (2,379 benign / 279 fraud)
        val   1,064 (  953 benign / 111 fraud)
        test  1,595 (1,426 benign / 169 fraud)
"""
from __future__ import annotations

import mmap
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp

MAGIC = bytes.fromhex("3fa1b496f0405edd")
KIND = {0x00010801: "u", 0x00014000: "i", 0x00012000: "f",
        0x00012002: "f", 0x00014002: "f", 0x00014001: "i"}

DEFAULT_DGL = Path(__file__).resolve().parents[1] / "datasets/FDCompCN_dataset/comp.dgl"

# manuscript split: (benign, fraud) per part
SPLIT_COUNTS = {"train": (2379, 279), "val": (953, 111), "test": (1426, 169)}


# --------------------------------------------------------------------------- #
# DGL binary reader
# --------------------------------------------------------------------------- #
def _scan(buf) -> list[dict]:
    u64 = lambda o: struct.unpack_from("<Q", buf, o)[0]
    u32 = lambda o: struct.unpack_from("<I", buf, o)[0]
    recs, pos = [], 0
    while True:
        m = buf.find(MAGIC, pos)
        if m < 0:
            break
        name = ""
        for L in range(1, 25):
            s = m - L
            if s - 8 >= 0 and u64(s - 8) == L and all(32 <= c < 127 for c in buf[s:m]):
                name = buf[s:m].decode()
                break
        o = m + 8
        nd, dt = u32(o + 16), u32(o + 20)
        dims = [u64(o + 24 + 8 * i) for i in range(nd)]
        p = o + 24 + 8 * nd
        nb = u64(p)
        p += 8
        recs.append({"name": name, "ndim": nd, "dt": dt, "dims": dims,
                     "nbytes": nb, "offset": p})
        pos = m + 8
    return recs


def _read(path: Path, rec: dict) -> np.ndarray:
    numel = int(np.prod(rec["dims"])) if rec["dims"] else 0
    itemsize = rec["nbytes"] // max(numel, 1)
    kind = KIND.get(rec["dt"], "i")
    dt = {"f": {4: "<f4", 8: "<f8"}, "i": {1: "<i1", 4: "<i4", 8: "<i8"},
          "u": {1: "<u1"}}[kind].get(itemsize, "<i8")
    a = np.fromfile(path, dtype=dt, count=numel, offset=rec["offset"])
    return a.reshape(rec["dims"]) if rec["ndim"] > 1 else a


def load_fdcompcn(path: str | Path = DEFAULT_DGL) -> dict:
    """Return features, labels and the union adjacency of the three relations."""
    path = Path(path)
    with open(path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        recs = _scan(buf)
        named = {}
        for r in recs:
            if r["name"] and r["name"] not in named:      # 5317-node graph comes first
                named[r["name"]] = r
        x = _read(path, named["feature"]).astype(np.float32)
        y = _read(path, named["label"]).ravel().astype(np.int64)
        chunks = [_read(path, r) for r in recs if not r["name"] and r["dims"] and r["dims"][0] > 100]
        buf.close()
    # chunks = [homo, invest, provide, sale]; the first pair is the union graph
    a, b = chunks[0], chunks[1]
    m = a != b                                            # drop the added self-loops
    e = np.unique(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1), axis=0)
    return {"x": x, "y": y, "edges": e.astype(np.int64)}


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def make_split(y: np.ndarray, seed: int = 1) -> dict:
    """Exact per-class counts of the manuscript, drawn from a seeded permutation."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in SPLIT_COUNTS}
    for cls, j in ((0, 0), (1, 1)):
        idx = rng.permutation(np.flatnonzero(y == cls))
        p = 0
        for part, cnt in SPLIT_COUNTS.items():
            n = cnt[j]
            out[part].append(idx[p:p + n])
            p += n
    return {k: np.sort(np.concatenate(v)).astype(np.int64) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #
def adjacency(n: int, edges: np.ndarray) -> sp.csr_matrix:
    """Symmetric binary adjacency (no self-loops)."""
    r = np.concatenate([edges[:, 0], edges[:, 1]])
    c = np.concatenate([edges[:, 1], edges[:, 0]])
    a = sp.coo_matrix((np.ones(r.size, np.float32), (r, c)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    return a


def row_normalize(a: sp.csr_matrix) -> sp.csr_matrix:
    """A_hat = D^{-1} A (+I), as used by Eq. (10)."""
    a = a + sp.eye(a.shape[0], dtype=np.float32, format="csr")
    d = np.asarray(a.sum(1)).ravel()
    return (sp.diags(1.0 / np.maximum(d, 1e-12)) @ a).tocsr()


def standardize(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mu = x[train_idx].mean(0, keepdims=True)
    sd = x[train_idx].std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return ((x - mu) / sd).astype(np.float32)
