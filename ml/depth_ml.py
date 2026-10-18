"""Sensor-teaches-camera: DINOv2 embeddings -> ridge regression on log distance.

Runs on the board (in ~/venv). Data layout (written by collector_server.py):
    ~/depth_data/<session>/frames/*.jpg
    ~/depth_data/<session>/labels.csv   (idx, file, ..., distance_mm, ...)

Commands:
    python depth_ml.py embed SESSION [SESSION ...]        # cache embeddings per session
    python depth_ml.py bench                              # single-image CPU latency
    python depth_ml.py train --train S1 S2 --test S3 S4   # fit head, evaluate, save model

Setup on the board (once):  pip install pillow
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

DATA = os.path.expanduser("~/depth_data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MIN_MM, MAX_MM = 50, 1200          # valid label range; everything else is dropped
WIDTH = 308                        # resize width, a multiple of the 14 px patch size
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- features

class Embedder:
    """DINOv2-small on CPU. Feature = [CLS token, mean of the centre patch tokens]."""

    def __init__(self, variant="dinov2_vits14", height=None):
        self.variant = variant
        self.height = height            # fixed after the first image if None
        self.model = torch.hub.load("facebookresearch/dinov2", variant).eval()

    def _size_for(self, img):
        if self.height is None:
            w, h = img.size
            self.height = max(14, round(WIDTH * h / w / 14) * 14)
        return WIDTH, self.height

    def preprocess(self, img):
        img = img.convert("RGB").resize(self._size_for(img), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    @torch.no_grad()
    def embed(self, images):
        x = torch.stack([self.preprocess(im) for im in images])
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]                       # (B, D)
        gh, gw = x.shape[2] // 14, x.shape[3] // 14
        patches = out["x_norm_patchtokens"].reshape(x.shape[0], gh, gw, -1)
        ch, cw = max(2, gh // 3), max(2, gw // 3)          # roughly the sensor's cone
        r0, c0 = (gh - ch) // 2, (gw - cw) // 2
        centre = patches[:, r0:r0 + ch, c0:c0 + cw].mean(dim=(1, 2))
        return torch.cat([cls, centre], dim=1).numpy()

    def config(self):
        return {"variant": self.variant, "width": WIDTH, "height": self.height}


def load_labels(session):
    import csv
    rows = []
    with open(os.path.join(DATA, session, "labels.csv")) as f:
        for r in csv.DictReader(f):
            d = float(r["distance_mm"])
            if MIN_MM <= d <= MAX_MM:
                rows.append((r["file"], d))
    return rows


def cmd_embed(args):
    emb = Embedder()
    for s in args.sessions:
        rows = load_labels(s)
        if not rows:
            print(f"{s}: no valid frames, skipped")
            continue
        feats, t0 = [], time.perf_counter()
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            imgs = [Image.open(os.path.join(DATA, s, "frames", f)) for f, _ in chunk]
            feats.append(emb.embed(imgs))
            print(f"\r{s}: {min(i + args.batch, len(rows))}/{len(rows)}", end="", flush=True)
        dt = time.perf_counter() - t0
        X = np.concatenate(feats)
        y = np.array([d for _, d in rows], dtype=np.float32)
        np.savez(os.path.join(DATA, s, "emb.npz"), X=X, y=y, files=np.array([f for f, _ in rows]))
        print(f"\r{s}: {len(rows)} frames embedded, {1000 * dt / len(rows):.1f} ms/frame (batched)")
    with open(os.path.join(DATA, "embed_config.json"), "w") as f:
        json.dump(emb.config(), f)


def cmd_bench(args):
    emb = Embedder()
    img = Image.new("RGB", (320, 180), (128, 128, 128))
    for _ in range(5):
        emb.embed([img])
    ts = []
    for _ in range(args.n):
        t = time.perf_counter()
        emb.embed([img])
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print(f"{emb.variant} {emb.config()['width']}x{emb.config()['height']}, "
          f"threads={torch.get_num_threads()}: p50 {ts[len(ts) // 2]:.1f} ms, "
          f"p95 {ts[int(len(ts) * 0.95)]:.1f} ms over {args.n} runs")


# ---------------------------------------------------------------- head

def load_sessions(names):
    Xs, ys, groups = [], [], []
    for g, s in enumerate(names):
        d = np.load(os.path.join(DATA, s, "emb.npz"))
        Xs.append(d["X"]); ys.append(d["y"]); groups.append(np.full(len(d["y"]), g))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)


def fit_ridge(X, y, alpha):
    """Ridge on log(distance), features standardised with training statistics."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    t = np.log(y)
    b = t.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (t - b))
    return {"mean": mu, "std": sd, "w": w, "b": b, "alpha": alpha}


def predict(head, X, calibrated=True):
    t = ((X - head["mean"]) / head["std"]) @ head["w"] + head["b"]
    if calibrated and "cal_a" in head:
        t = head["cal_a"] * t + head["cal_b"]
    return np.exp(t)


def out_of_fold_log_preds(X, y, groups, alpha):
    """Predictions for every training frame from a head that never saw that frame's session."""
    t = np.zeros(len(y))
    sessions = np.unique(groups)
    if len(sessions) > 1:
        folds = [(groups != g, groups == g) for g in sessions]
    else:
        cut = int(len(y) * 0.8)
        tr = np.arange(len(y)) < cut
        folds = [(tr, ~tr)]
    covered = np.zeros(len(y), dtype=bool)
    for tr, va in folds:
        h = fit_ridge(X[tr], y[tr], alpha)
        t[va] = np.log(predict(h, X[va]))
        covered |= va
    return t[covered], np.log(y[covered])


def metrics(y, p, baseline):
    err = np.abs(p - y) / 10.0                                  # cm
    return {"n": len(y), "mae_cm": err.mean(), "median_cm": np.median(err),
            "within_5cm": (err <= 5).mean() * 100,
            "baseline_mae_cm": np.abs(baseline - y).mean() / 10.0}


def choose_alpha(X, y, groups, grid):
    """Leave-one-session-out over the training sessions; contiguous 80/20 split if only one."""
    folds = []
    if len(np.unique(groups)) > 1:
        folds = [(groups != g, groups == g) for g in np.unique(groups)]
    else:
        cut = int(len(y) * 0.8)
        tr = np.arange(len(y)) < cut
        folds = [(tr, ~tr)]
    scores = {}
    for a in grid:
        maes = [np.abs(predict(fit_ridge(X[tr], y[tr], a), X[va]) - y[va]).mean() for tr, va in folds]
        scores[a] = float(np.mean(maes))
    return min(scores, key=scores.get), scores


def cmd_train(args):
    X, y, groups = load_sessions(args.train)
    grid = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    alpha, scores = choose_alpha(X, y, groups, grid)
    print("validation MAE (mm) by alpha:", {a: round(v, 1) for a, v in scores.items()})
    print(f"chosen alpha = {alpha}")

    head = fit_ridge(X, y, alpha)

    # Calibration: ridge shrinks predictions toward the mean (too far when close, too near when
    # far). Fit log(true) = a * log(pred) + b on out-of-fold training predictions only.
    tp, tt = out_of_fold_log_preds(X, y, groups, alpha)
    a, b = np.polyfit(tp, tt, 1)
    head["cal_a"], head["cal_b"] = float(a), float(b)
    print(f"calibration from out-of-fold training predictions: slope {a:.2f} (1.00 = no compression)")

    baseline = float(np.median(y))
    print(f"\ntrained on {len(y)} frames from {args.train}; baseline = always predict {baseline:.0f} mm\n")
    print(f"{'test set':<16}{'frames':>8}{'MAE cm':>9}{'uncal cm':>10}{'median cm':>11}"
          f"{'<=5 cm %':>10}{'bias cm':>9}{'baseline cm':>13}")
    results = {}
    for s in args.test:
        Xt, yt, _ = load_sessions([s])
        p = predict(head, Xt)
        m = metrics(yt, p, baseline)
        m["mae_uncal_cm"] = np.abs(predict(head, Xt, calibrated=False) - yt).mean() / 10
        m["bias_cm"] = (p - yt).mean() / 10
        results[s] = m
        print(f"{s:<16}{m['n']:>8}{m['mae_cm']:>9.1f}{m['mae_uncal_cm']:>10.1f}{m['median_cm']:>11.1f}"
              f"{m['within_5cm']:>10.0f}{m['bias_cm']:>9.1f}{m['baseline_mae_cm']:>13.1f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, "head.npz"), **head)
    cfg = json.load(open(os.path.join(DATA, "embed_config.json")))
    cfg.update({"train_sessions": args.train, "alpha": alpha, "min_mm": MIN_MM, "max_mm": MAX_MM,
                "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}})
    json.dump(cfg, open(os.path.join(MODEL_DIR, "config.json"), "w"), indent=2)
    print(f"\nsaved model to {MODEL_DIR}")


# ---------------------------------------------------------------- live use

class EtEmbedder(Embedder):
    """Same features, computed by the exported ExecuTorch program instead of PyTorch."""

    def __init__(self, pte_path, height):
        from executorch.runtime import Runtime
        self.variant, self.height = os.path.basename(pte_path), height
        self.method = Runtime.get().load_program(pte_path).load_method("forward")

    def embed(self, images):
        return np.concatenate([self.method.execute([self.preprocess(im).unsqueeze(0)])[0].numpy()
                               for im in images])


class DepthPredictor:
    """For the live demo: jpeg bytes in, predicted distance in mm out.
    Uses the ExecuTorch program if export_et.py has produced one, else PyTorch."""

    def __init__(self, model_dir=MODEL_DIR, backend="auto"):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        pte = os.path.join(model_dir, cfg.get("pte", "dinov2_fp32.pte"))
        if backend in ("auto", "et") and os.path.exists(pte):
            self.embedder = EtEmbedder(pte, cfg["height"])
        else:
            self.embedder = Embedder(cfg["variant"], cfg["height"])
        print("DepthPredictor using:", self.embedder.variant)
        d = np.load(os.path.join(model_dir, "head.npz"))
        self.head = {k: d[k] for k in d.files if k != "alpha"}

    def predict_jpeg(self, data):
        import io
        img = Image.open(io.BytesIO(data))
        return float(predict(self.head, self.embedder.embed([img]))[0])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("sessions", nargs="+"); e.add_argument("--batch", type=int, default=16)
    b = sub.add_parser("bench"); b.add_argument("--n", type=int, default=50)
    t = sub.add_parser("train"); t.add_argument("--train", nargs="+", required=True); t.add_argument("--test", nargs="+", required=True)
    args = ap.parse_args()
    {"embed": cmd_embed, "bench": cmd_bench, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
