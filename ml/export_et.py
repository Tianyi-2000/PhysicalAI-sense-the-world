"""Export the DINOv2 feature extractor to ExecuTorch (XNNPACK), fp32 and optionally int8,
then benchmark and re-evaluate accuracy through the exported programs.

Run on the board after `depth_ml.py train` has saved ~/ml/model/:
    python export_et.py            # fp32 only
    python export_et.py --int8     # fp32 + int8 (post-training quantization)

Writes ~/ml/model/dinov2_fp32.pte (and dinov2_int8.pte), et_results.json, and points
config.json at the fp32 program so the live demo uses ExecuTorch.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from depth_ml import DATA, MODEL_DIR, Embedder, load_labels, predict

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime


class Features(torch.nn.Module):
    """Same feature as depth_ml.Embedder.embed, with the position encoding precomputed for a
    fixed input size (avoids exporting the bicubic interpolation)."""

    def __init__(self, m, H, W):
        super().__init__()
        self.m = m
        with torch.no_grad():
            dummy = torch.zeros(1, 3, H, W)
            t = m.patch_embed(dummy)
            t = torch.cat((m.cls_token.expand(1, -1, -1), t), dim=1)
            _, _, w, h = dummy.shape                 # DINOv2's own (w, h) naming convention
            self.register_buffer("pos", m.interpolate_pos_encoding(t, w, h).clone())
        self.n_reg = getattr(m, "num_register_tokens", 0) or 0
        gh, gw = H // 14, W // 14
        ch, cw = max(2, gh // 3), max(2, gw // 3)
        self.gh, self.gw, self.ch, self.cw = gh, gw, ch, cw
        self.r0, self.c0 = (gh - ch) // 2, (gw - cw) // 2

    def forward(self, x):
        b = x.shape[0]
        t = self.m.patch_embed(x)
        t = torch.cat((self.m.cls_token.expand(b, -1, -1), t), dim=1) + self.pos
        if self.n_reg:
            t = torch.cat((t[:, :1], self.m.register_tokens.expand(b, -1, -1), t[:, 1:]), dim=1)
        for blk in self.m.blocks:
            t = blk(t)
        t = self.m.norm(t)
        cls = t[:, 0]
        p = t[:, 1 + self.n_reg:].reshape(b, self.gh, self.gw, -1)
        centre = p[:, self.r0:self.r0 + self.ch, self.c0:self.c0 + self.cw].mean(dim=(1, 2))
        return torch.cat([cls, centre], dim=1)


def lower(module, example, path):
    ep = torch.export.export(module, (example,))
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(path, "wb") as f:
        f.write(prog.buffer)
    return os.path.getsize(path) / 1e6


def bench(method, x, n=30):
    for _ in range(5):
        method.execute([x])
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        method.execute([x])
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[int(len(ts) * 0.95)]


def evaluate(method, emb, head, sessions):
    out = {}
    for s in sessions:
        rows = load_labels(s)
        feats = []
        for f, _ in rows:
            x = emb.preprocess(Image.open(os.path.join(DATA, s, "frames", f))).unsqueeze(0)
            feats.append(method.execute([x])[0].numpy())
        y = np.array([d for _, d in rows])
        err = np.abs(predict(head, np.concatenate(feats)) - y) / 10
        out[s] = {"mae_cm": float(err.mean()), "median_cm": float(np.median(err))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--n-calib", type=int, default=32)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    d = np.load(os.path.join(MODEL_DIR, "head.npz"))
    head = {k: d[k] for k in d.files if k != "alpha"}
    test_sessions = list(cfg["results"].keys())

    emb = Embedder(cfg["variant"], cfg["height"])
    H, W = cfg["height"], cfg["width"]
    feat = Features(emb.model, H, W).eval()

    # 1. the wrapper must reproduce the training features exactly
    s0 = cfg["train_sessions"][0]
    files = [f for f, _ in load_labels(s0)]
    imgs = [Image.open(os.path.join(DATA, s0, "frames", f)) for f in files[:4]]
    with torch.no_grad():
        ref = emb.embed(imgs)
        mine = feat(torch.stack([emb.preprocess(im) for im in imgs])).numpy()
    print(f"wrapper vs training features, max diff: {np.abs(ref - mine).max():.2e}")

    x = emb.preprocess(imgs[0]).unsqueeze(0)
    results = {}

    # 2. fp32
    p32 = os.path.join(MODEL_DIR, "dinov2_fp32.pte")
    size = lower(feat, x, p32)
    m32 = Runtime.get().load_program(p32).load_method("forward")
    diff = np.abs(m32.execute([x])[0].numpy() - mine[:1]).max()
    p50, p95 = bench(m32, x)
    print(f"fp32: {size:.1f} MB, p50 {p50:.1f} ms, p95 {p95:.1f} ms, max diff vs PyTorch {diff:.2e}")
    acc = evaluate(m32, emb, head, test_sessions)
    print("fp32 accuracy:", {s: round(v["mae_cm"], 1) for s, v in acc.items()}, "cm MAE")
    results["executorch_fp32"] = {"size_mb": size, "p50_ms": p50, "p95_ms": p95, "accuracy": acc}

    # 3. int8 post-training quantization (optional)
    if args.int8:
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config)
            try:
                from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
            except ImportError:
                from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e

            q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
            gm = prepare_pt2e(torch.export.export(feat, (x,)).module(), q)
            calib = files[::max(1, len(files) // args.n_calib)][:args.n_calib]
            with torch.no_grad():
                for f in calib:
                    gm(emb.preprocess(Image.open(os.path.join(DATA, s0, "frames", f))).unsqueeze(0))
            gm = convert_pt2e(gm)
            p8 = os.path.join(MODEL_DIR, "dinov2_int8.pte")
            size8 = lower(gm, x, p8)
            m8 = Runtime.get().load_program(p8).load_method("forward")
            p50, p95 = bench(m8, x)
            print(f"int8: {size8:.1f} MB, p50 {p50:.1f} ms, p95 {p95:.1f} ms")
            acc8 = evaluate(m8, emb, head, test_sessions)
            print("int8 accuracy:", {s: round(v["mae_cm"], 1) for s, v in acc8.items()}, "cm MAE")
            results["executorch_int8"] = {"size_mb": size8, "p50_ms": p50, "p95_ms": p95, "accuracy": acc8}
        except Exception as e:
            print("int8 export failed (fp32 is still saved):", repr(e))

    json.dump(results, open(os.path.join(MODEL_DIR, "et_results.json"), "w"), indent=2)
    cfg["pte"] = "dinov2_fp32.pte"
    json.dump(cfg, open(os.path.join(MODEL_DIR, "config.json"), "w"), indent=2)
    print("\nconfig.json now points the live demo at dinov2_fp32.pte")


if __name__ == "__main__":
    main()
