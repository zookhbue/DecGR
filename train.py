from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import data as DATA
from model import DecGR, build_hypergraph

ROOT = Path(__file__).resolve().parent


def sparse_tensor(a: sp.csr_matrix) -> torch.Tensor:
    a = a.tocoo()
    idx = torch.from_numpy(np.vstack((a.row, a.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(a.data.astype(np.float32)),
                                   a.shape).coalesce()


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    """Fraud metrics at the argmax decision (threshold 0.5), as stated in the paper."""
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"auc": float(roc_auc_score(y, p)),
            "gmean": float(np.sqrt(tpr * tnr)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "ap": float(average_precision_score(y, p)),
            "tpr": float(tpr), "tnr": float(tnr)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device(args.device)
    t0 = time.time()

    ds = DATA.load_fdcompcn()
    x_np, y_np = ds["x"], ds["y"]
    n = x_np.shape[0]
    split = DATA.make_split(y_np, cfg["seed"])
    print(f"nodes {n}  fraud {int(y_np.sum())}  edges {ds['edges'].shape[0]}")
    print("split:", {k: int(v.size) for k, v in split.items()},
          "fraud:", {k: int(y_np[v].sum()) for k, v in split.items()})

    x_np = DATA.standardize(x_np, split["train"])
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    adj_np = DATA.row_normalize(DATA.adjacency(n, ds["edges"]))
    adj = sparse_tensor(adj_np).to(device)
    edge_index = adj.indices()

    model = DecGR(x.shape[1], hidden=cfg["hidden"], layers=cfg["layers"], k=cfg["k"],
                  dropout=cfg["dropout"], lam=cfg["lam"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    tr = torch.from_numpy(split["train"]).to(device)
    va = torch.from_numpy(split["val"]).to(device)
    te = torch.from_numpy(split["test"]).to(device)

    epochs = args.epochs or cfg["epochs"]
    best = {"auc": -1.0, "state": None, "epoch": -1}
    wait = 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, out = model(x, adj, edge_index, None)      # H rebuilt per layer
        loss_cls = F.cross_entropy(logits[tr], y[tr])                  # Eq.(24)
        loss = loss_cls + cfg.get("dec_w", 1.0) * model.decoupling_loss(out)   # Eq.(21)-(25)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % cfg["eval_every"] == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                logits_e, _ = model(x, adj, edge_index, None)
                p = torch.softmax(logits_e, -1)[:, 1].cpu().numpy()
            auc = roc_auc_score(y_np[split["val"]], p[split["val"]])
            if auc > best["auc"]:
                best = {"auc": auc, "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                wait = 0
            else:
                wait += 1
            print(f"  ep {ep:>3d} loss {loss.item():.4f} val AUC {auc:.4f} "
                  f"(best {best['auc']:.4f})  {time.time() - t0:.0f}s", flush=True)
            if wait * cfg["eval_every"] >= cfg["patience"]:
                print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, adj, edge_index, None)
        p = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    res = evaluate(y_np[split["test"]], p[split["test"]])
    res.update({"best_epoch": int(best["epoch"]), "val_auc": float(best["auc"]),
                "seconds": round(time.time() - t0, 1), "config": cfg})

    print("\nTest:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

    ck = ROOT / "checkpoints"; ck.mkdir(exist_ok=True)
    torch.save({"state_dict": best["state"], "config": cfg,
                "in_dim": int(x.shape[1]), "metrics": res}, ck / "decgr_fdcompcn.pt")
    lg = ROOT / "logs"; lg.mkdir(exist_ok=True)
    (lg / "decgr_fdcompcn.json").write_text(json.dumps(res, indent=2) + "\n")
    print("saved checkpoint and log")


if __name__ == "__main__":
    main()
