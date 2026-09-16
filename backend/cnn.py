# -*- coding: utf-8 -*-
"""
手写数字识别 CNN —— 训练 / 推理 / 算子化分析的数据模型
纯 numpy 实现(卷积+ReLU+MaxPool+全连接, Adam)。
MNIST 从公开镜像下载一次并缓存。

网络(LeNet 风格, 28×28 输入):
  Conv1: 8 个 5×5(单通道输入, 每个输出滤波器即一个 2D 算子)
  ReLU → MaxPool 2×2 → 14×14
  Conv2: 16 个 5×5(8 通道输入, 每个输出 = 8 个 2D 分量算子的叠加)
  ReLU → MaxPool 2×2 → 7×7
  FC 784→120 (ReLU) → 10 (Softmax)
"""
from __future__ import annotations

import gzip
import json
import math
import os
import struct
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "cnn_mnist.json")
MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

# ---------------- numpy 惰性导入(允许纯标准库降级) ----------------
try:
    import numpy as np
    NP_OK = True
except Exception:  # noqa: BLE001
    np = None
    NP_OK = False

if not NP_OK:
    # 尝试从常见本地环境目录补位(可用 CNN_PYTHONPATH 指定 numpy 所在目录)
    _cands = []
    _env = os.environ.get("CNN_PYTHONPATH")
    if _env:
        _cands.append(_env)
    _cands.append(os.path.join(DATA_DIR, "..", ".venv", "lib", "python3", "site-packages"))
    for _c in _cands:
        if not os.path.isdir(_c):
            continue
        try:
            import sys
            if _c not in sys.path:
                sys.path.insert(0, _c)
            import numpy  # type: ignore  # noqa: F811
            np = numpy
            NP_OK = True
            break
        except Exception:  # noqa: BLE001
            continue

ARCH = {
    "name": "LeNet-ish MNIST CNN",
    "input": [28, 28, 1],
    "layers": [
        {"kind": "conv", "filters": 8, "k": 5, "pad": 2, "note": "单通道: 每个输出滤波器=1 个 2D 卷积算子"},
        {"kind": "relu"},
        {"kind": "pool", "size": 2},
        {"kind": "conv", "filters": 16, "k": 5, "pad": 2, "note": "8 通道: 每个输出=8 个 2D 分量算子叠加"},
        {"kind": "relu"},
        {"kind": "pool", "size": 2},
        {"kind": "fc", "out": 120, "note": "7×7×16=784 → 120"},
        {"kind": "relu"},
        {"kind": "fc", "out": 10, "note": "→ 10 类 Softmax"},
    ],
}


# ---------------------------------------------------------------- 数据
def _download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    print(f"[cnn] 下载 {os.path.basename(dst)} ...")
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        raw = f.read()
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, rows, cols)
    return a


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8).copy()


def ensure_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in MNIST_URLS.items():
        _download(url, os.path.join(DATA_DIR, name))
    if not (os.path.exists(os.path.join(DATA_DIR, "Xtr.npy"))
            and os.path.exists(os.path.join(DATA_DIR, "Ytr.npy"))):
        print("[cnn] 解析 MNIST 缓存 ...")
        Xtr = _read_idx_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
        Ytr = _read_idx_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
        Xte = _read_idx_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
        Yte = _read_idx_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
        np.save(os.path.join(DATA_DIR, "Xtr.npy"), Xtr)
        np.save(os.path.join(DATA_DIR, "Ytr.npy"), Ytr)
        np.save(os.path.join(DATA_DIR, "Xte.npy"), Xte)
        np.save(os.path.join(DATA_DIR, "Yte.npy"), Yte)
    Xtr = np.load(os.path.join(DATA_DIR, "Xtr.npy"))
    Ytr = np.load(os.path.join(DATA_DIR, "Ytr.npy"))
    Xte = np.load(os.path.join(DATA_DIR, "Xte.npy"))
    Yte = np.load(os.path.join(DATA_DIR, "Yte.npy"))
    return (Xtr.astype(np.float32) / 255.0, Ytr,
            Xte.astype(np.float32) / 255.0, Yte)


# ---------------------------------------------------------------- 网络
def init_weights(seed=2026):
    rng = np.random.RandomState(seed)
    W = {}
    for key, fin, fout, k, scale in (
        ("c1", 1, 8, 5, 1.0), ("c2", 8, 16, 5, 1.0)):
        W[key + "w"] = rng.normal(0, scale / math.sqrt(fin * k * k),
                                  (fout, fin, k, k)).astype(np.float32)
        W[key + "b"] = np.zeros(fout, dtype=np.float32)
    W["f1w"] = rng.normal(0, 0.08, (120, 784)).astype(np.float32)
    W["f1b"] = np.zeros(120, dtype=np.float32)
    W["f2w"] = rng.normal(0, 0.08, (10, 120)).astype(np.float32)
    W["f2b"] = np.zeros(10, dtype=np.float32)
    return W


def im2col(x, pad):
    """x: (B,C,H,W) → (B, C*5*5, H*W), stride 1, 核 5×5"""
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    B2, _, H2, W2 = xp.shape
    oh, ow = H2 - 4, W2 - 4
    cols = np.empty((B, C, 5, 5, oh, ow), dtype=x.dtype)
    for i in range(5):
        for j in range(5):
            cols[:, :, i, j] = xp[:, :, i:i + oh, j:j + ow]
    return cols.reshape(B, C * 25, oh * ow)


def conv_fwd(x, w, b, pad):
    """x (B,C,H,W); w (F,C,5,5) → (B,F,H,W) stride1"""
    B = x.shape[0]
    cols = im2col(x, pad)                     # (B, C*25, P)
    out = np.einsum("bfp,of->bop", cols, w.reshape(w.shape[0], -1))  # (B,F,P)
    out += b.reshape(1, -1, 1)
    H = x.shape[2]
    return out.reshape(B, w.shape[0], H, H)


def conv_bwd(g, x, w, pad):
    """返回 (dx, dw, db); g (B,F,H,H)"""
    B = x.shape[0]
    cols = im2col(x, pad)                     # (B,C*25,P)
    g2 = g.reshape(g.shape[0], g.shape[1], -1)
    wm = w.reshape(w.shape[0], -1)            # (F,C25)
    dw = np.einsum("bop,bfp->of", g2, cols)   # (F,C25)
    db = g2.sum(axis=(0, 2))
    dcols = np.einsum("bop,of->bfp", g2, wm)  # (B,C25,P)
    dx = np.zeros_like(np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))))
    dcols = dcols.reshape(B, x.shape[1], 5, 5, x.shape[2], x.shape[3])
    for i in range(5):
        for j in range(5):
            dx[:, :, i:i + x.shape[2], j:j + x.shape[3]] += dcols[:, :, i, j]
    if pad:
        dx = dx[:, :, pad:-pad, pad:-pad]
    return dx, dw.reshape(w.shape), db


def pool_fwd(x):
    B, C, H, W = x.shape
    xr = x.reshape(B, C, H // 2, 2, W // 2, 2)
    m = xr.max(axis=(3, 5))
    return m


def pool_mask(x):
    B, C, H, W = x.shape
    xr = x.reshape(B, C, H // 2, 2, W // 2, 2)
    m = xr.max(axis=(3, 5), keepdims=True)
    mask = (xr == m)
    # 平局时取第一个
    first = np.zeros_like(mask)
    first[:, :, :, 0, :, 0] = True
    mask &= first | ~mask.any(axis=(3, 5), keepdims=True)
    return mask


def pool_bwd(g, x):
    B, C, H, W = x.shape
    xr = x.reshape(B, C, H // 2, 2, W // 2, 2)
    mask = xr == xr.max(axis=(3, 5), keepdims=True)
    gr = g.reshape(B, C, H // 2, 1, W // 2, 1)
    dxr = np.where(mask, gr, 0.0)
    return dxr.reshape(B, C, H, W)


def relu(x): return np.maximum(x, 0)


def forward(x, W):
    """返回 (logits, cache)"""
    x1 = x
    z1 = conv_fwd(x1, W["c1w"], W["c1b"], 2); a1 = relu(z1)
    p1 = pool_fwd(a1)
    z2 = conv_fwd(p1, W["c2w"], W["c2b"], 2); a2 = relu(z2)
    p2 = pool_fwd(a2)
    flat = p2.reshape(p2.shape[0], -1)
    z3 = flat.dot(W["f1w"].T) + W["f1b"]; a3 = relu(z3)
    z4 = a3.dot(W["f2w"].T) + W["f2b"]
    return z4, (x1, z1, a1, p1, z2, a2, p2, flat, z3, a3, z4)


def softmax(z):
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)


def loss_grad(z, y):
    B = z.shape[0]
    p = softmax(z)
    onehot = np.zeros_like(p)
    onehot[np.arange(B), y] = 1.0
    dz = (p - onehot) / B
    ce = -np.mean(np.log(p[np.arange(B), y] + 1e-12))
    return ce, dz


def backward(dz, cache, W):
    x1, z1, a1, p1, z2, a2, p2, flat, z3, a3, _z4 = cache
    g = {}
    d4 = dz
    g["f2w"] = d4.T.dot(a3)
    g["f2b"] = d4.sum(axis=0)
    d3 = d4.dot(W["f2w"]) * (a3 > 0)
    g["f1w"] = d3.T.dot(flat)
    g["f1b"] = d3.sum(axis=0)
    dflat = d3.dot(W["f1w"]).reshape(p2.shape)
    dp2 = dflat
    da2 = pool_bwd(dp2, a2)
    dz2 = da2 * (a2 > 0)
    dx2, g["c2w"], g["c2b"] = conv_bwd(dz2, p1, W["c2w"], 2)
    da1 = pool_bwd(dx2, a1)
    dz1 = da1 * (a1 > 0)
    _dx1, g["c1w"], g["c1b"] = conv_bwd(dz1, x1, W["c1w"], 2)
    return g


def accuracy(X, y, W, batch=256):
    corr = tot = 0
    for s in range(0, len(X), batch):
        z, _ = forward(X[s:s + batch], W)
        corr += int((z.argmax(axis=1) == y[s:s + batch]).sum())
        tot += len(X[s:s + batch])
    return corr / tot


# ---------------------------------------------------------------- 训练
def _project_sym(W, mode):
    """把卷积滤波器按 180° 对称性投影(实验用):
    mode='even' 取偶部 k←(k+flip)/2 (自伴 ⇒ 拉氏量可导出类)
    mode='odd'  取奇部 k←(k-flip)/2 (反自伴/陀螺类)
    就地更新 W 的 c1w/c2w。"""
    for key in ("c1w", "c2w"):
        w = W[key]
        f = np.flip(np.flip(w, axis=2), axis=3)
        if mode == "even":
            np.copyto(w, (w + f) * 0.5)
        else:
            np.copyto(w, (w - f) * 0.5)


def train(subset=12000, epochs=4, batch=64, lr=1e-3, seed=2026,
          log_every=5, max_minutes=40.0, sym="", sym_every_step=True):
    """在 MNIST 子集上训练; 返回 (weights, report)。
    sym ∈ {"", "even", "odd"}: 每步后把卷积滤波器投影为偶核/奇核(实验)。"""
    assert NP_OK, "需要 numpy"
    Xtr, Ytr, Xte, Yte = ensure_mnist()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=subset, replace=False)
    Xs, Ys = Xtr[idx], Ytr[idx]
    # 留 1000 做验证
    nval = min(1000, subset // 10)
    Xv, Yv = Xs[:nval], Ys[:nval]
    Xt, Yt = Xs[nval:], Ys[nval:]
    W = init_weights(seed)
    m = {k: np.zeros_like(v) for k, v in W.items()}
    v = {k: np.zeros_like(val) for k, val in W.items()}
    steps = 0
    t0 = time.time()
    history = []
    best = None
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(Xt))
        Xt, Yt = Xt[perm], Yt[perm]
        for s in range(0, len(Xt), batch):
            xb = Xt[s:s + batch][:, None, :, :]
            yb = Yt[s:s + batch]
            if len(xb) < 2:
                continue
            z, cache = forward(xb, W)
            ce, dz = loss_grad(z, yb)
            g = backward(dz, cache, W)
            steps += 1
            lr_t = lr * (0.99 ** (steps / 400))
            for k2 in W:
                m[k2] = 0.9 * m[k2] + 0.1 * g[k2]
                v[k2] = 0.999 * v[k2] + 0.001 * (g[k2] ** 2)
                mh = m[k2] / (1 - 0.9 ** steps)
                vh = v[k2] / (1 - 0.999 ** steps)
                W[k2] -= lr_t * mh / (np.sqrt(vh) + 1e-8)
            if sym:
                _project_sym(W, sym)
            if steps % log_every == 0:
                accv = accuracy(Xv[:, None, :, :], Yv, W, 500)
                history.append({"step": steps, "epoch": ep, "loss": round(float(ce), 5),
                                "val_acc": round(float(accv), 4)})
            if time.time() - t0 > max_minutes * 60:
                break
        accv = accuracy(Xv[:, None, :, :], Yv, W, 500)
        print(f"[cnn] epoch {ep}: val_acc={accv:.4f} ({time.time()-t0:.0f}s)")
        history.append({"step": steps, "epoch": ep, "val_acc": round(float(accv), 4)})
        if best is None or accv > best[0]:
            best = (float(accv), {k2: v2.copy() for k2, v2 in W.items()})
    if best:
        W = best[1]
    acc_test = accuracy(Xte[:, None, :, :], Yte, W, 512)
    acc_val = best[0] if best else 0.0
    return W, {
        "subset": subset, "epochs": epochs, "seed": seed,
        "val_acc": round(acc_val, 4), "test_acc": round(float(acc_test), 4),
        "secs": round(time.time() - t0, 1), "history": history[-40:],
    }


def save_model(W, report):
    obj = {"arch": ARCH, "weights": {k: v.tolist() for k, v in W.items()},
           "report": report}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def load_model():
    return load_model_file(MODEL_PATH)


def load_model_file(path):
    """读取任意模型 JSON(与 save_model/predict 同格式), 供多模型/演示页使用。"""
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    W = {k: np.asarray(v, dtype=np.float32) for k, v in obj["weights"].items()}
    return W, obj.get("report")


def predict_pixels(pixels, W=None):
    """pixels: 28×28 数值(0..255 或 0..1) → (probs list10, pred)"""
    if W is None:
        W, _ = load_model()
        assert W is not None, "模型未训练"
    a = np.asarray(pixels, dtype=np.float32)
    if a.shape != (28, 28):
        raise ValueError("需要 28×28")
    if a.max() > 1.5:
        a = a / 255.0
    z, _ = forward(a[None, None, :, :], W)
    p = softmax(z)[0]
    return [round(float(x), 6) for x in p], int(p.argmax())


def model_status():
    W, rep = load_model()
    if W is None:
        return {"trained": False, "numpy": NP_OK}
    return {"trained": True, "numpy": NP_OK, "report": rep}


# ---------------------------------------------------------------- 算子化分析
def layer_analyses(W=None):
    """把训练好的卷积滤波器放进算子设计空间逐层分析(用 kernelmath.quick_analyze)。
    conv1(单通道): 每个输出滤波器 = 1 个 2D 算子 → 完整逐卡分析。
    conv2(8 通道): 每个输出 = 8 个 2D 分量算子 → 逐分量统计 + 每输出主分量。
    """
    import kernelmath as km
    if W is None:
        W, _ = load_model()
        if W is None:
            raise ValueError("模型未训练")
    c1w = W["c1w"]  # (8,1,5,5)
    c2w = W["c2w"]  # (16,8,5,5)

    def kround(k):
        return [[round(float(v), 5) for v in row] for row in k]

    def agg_stats(qas):
        n = len(qas)
        if n == 0:
            return {}
        def cnt(f):
            return sum(1 for q in qas if f(q))
        even_n = cnt(lambda q: q["type"] == "even")
        odd_n = cnt(lambda q: q["type"] == "odd")
        general_n = cnt(lambda q: q["type"] == "general")
        psd_n = cnt(lambda q: q["psd"])
        nsd_n = cnt(lambda q: q["nsd"])
        indef_n = cnt(lambda q: q["definite"] == "indefinite")
        compl_n = cnt(lambda q: q["definite"] == "complex")
        mean_even = sum(q["even_frac"] for q in qas) / n
        mean_odd = sum(q["odd_frac"] for q in qas) / n
        kinds = {}
        for q in qas:
            kinds[q["filter_kind"]] = kinds.get(q["filter_kind"], 0) + 1
        tpl = {}
        for q in qas:
            tpl[q["template"]["best"]] = tpl.get(q["template"]["best"], 0) + 1
        orient = [0] * 12
        for q in qas:
            if q["orient_deg"]:
                b = min(11, int(q["orient_deg"]["deg"] // 15))
                orient[b] += 1
        return {
            "n": n,
            "even_n": even_n, "odd_n": odd_n, "general_n": general_n,
            "psd_n": psd_n, "nsd_n": nsd_n, "indef_n": indef_n, "complex_n": compl_n,
            "mean_even_frac": round(mean_even, 4), "mean_odd_frac": round(mean_odd, 4),
            "kinds": kinds, "templates": tpl, "orient_hist": orient,
        }

    def agg_cplx(cps):
        """复化(虚数场)聚合: 动能占比均值/范围、Hermitian PSD 计数、复化实谱范围。"""
        n = len(cps)
        if n == 0:
            return {}
        ks = [c["kinetic_share"] for c in cps]
        psd = sum(1 for c in cps if c["herm_psd"])
        psd_alt = sum(1 for c in cps if c["herm_psd_alt"])
        lo = min(c["hc_min"] for c in cps)
        hi = max(c["hc_max"] for c in cps)
        lo_a = min(c["alt_min"] for c in cps)
        hi_a = max(c["alt_max"] for c in cps)
        return {
            "n": n,
            "kinetic_mean": round(sum(ks) / n, 4),
            "kinetic_lo": round(min(ks), 4), "kinetic_hi": round(max(ks), 4),
            "herm_psd_n": psd, "herm_psd_alt_n": psd_alt,
            "hc_min": round(lo, 4), "hc_max": round(hi, 4),
            "alt_min": round(lo_a, 4), "alt_max": round(hi_a, 4),
        }

    # ---- conv1: 单通道算子 ----
    conv1 = []
    for f in range(c1w.shape[0]):
        k = c1w[f, 0].tolist()
        qa = km.quick_analyze(k)
        conv1.append({"idx": f, "k": kround(k), "qa": qa,
                      "cplx": km.complex_quick(k)})
    # ---- conv2: 多通道 → 逐分量 ----
    conv2 = []
    for o in range(c2w.shape[0]):
        comps = []
        for ci in range(c2w.shape[1]):
            k = c2w[o, ci].tolist()
            qa = km.quick_analyze(k)
            cp = km.complex_quick(k)
            comps.append({
                "cin": ci,
                "type": qa["type"], "definite": qa["definite"],
                "even_frac": qa["even_frac"], "odd_frac": qa["odd_frac"],
                "psd": qa["psd"],
                "norm": round(float(np.linalg.norm(c2w[o, ci])), 5),
                "template": qa["template"]["best"],
                "filter_kind": qa["filter_kind"],
                "kin": cp["kinetic_share"], "hpsd": bool(cp["herm_psd"]),
            })
        dom = max(range(len(comps)), key=lambda i: comps[i]["norm"])
        domk = c2w[o, dom].tolist()
        qad = km.quick_analyze(domk)
        conv2.append({
            "out": o,
            "dominant_cin": int(dom),
            "dominant_k": kround(domk),
            "dominant_qa": qad,
            "cplx": km.complex_quick(domk),
            "components": comps,
        })
    agg1 = agg_stats([f["qa"] for f in conv1])
    agg2 = agg_stats([o["dominant_qa"] for o in conv2])
    # conv2 分量级汇总(全部 输出×通道 个 2D 分量)
    comp_qas = []
    comp_cps = []
    for o in range(c2w.shape[0]):
        for ci in range(c2w.shape[1]):
            qa = km.quick_analyze(c2w[o, ci].tolist())
            comp_qas.append(qa)
            comp_cps.append(km.complex_quick(c2w[o, ci].tolist()))
    return {
        "arch": ARCH,
        "conv1": {"filters": conv1, "agg": agg1,
                  "cplx_agg": agg_cplx([f["cplx"] for f in conv1]),
                  "note": "第一层: 单通道输入, 每个输出滤波器即一个 2D 平移等变卷积算子"},
        "conv2": {"outputs": conv2,
                  "agg_output": agg2,
                  "comp_agg": agg_stats(comp_qas),
                  "cplx_comp_agg": agg_cplx(comp_cps),
                  "note": "第二层: 8 通道输入 → 每个输出是 8 个 2D 分量算子(逐输入通道)的叠加"},
    }
