# -*- coding: utf-8 -*-
"""端到端分类机制 —— 定量解剖(只推理, 不训练)

对 data/cnn_mnist.json 的 98% 模型做机制级量化:
  M1 各级(像素→pool2 卷积特征→FC1 120维)的线性可分度(岭回归/质心, 分半估)
  M2 通道×数字 平均激活与判别力(F 比): 哪些算子通道在"实现"哪个数字类
  M3 末层决策几何: 10 个决策方向(W2 行)的范数/余弦矩阵、类中心几何
  M4 边界与混淆: margin 分布、混淆矩阵、最易混对
  M5 稀疏门控: 各层 ReLU 激活率(机制: 特征"门"选择)

结果写 data/mech.json, 供 /api/cnn/mech 与页面 ⑤ 使用。
python3 backend/mech.py   (需要 numpy)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(DATA_DIR, "mech.json")

if not cnn.NP_OK:
    sys.exit("需要 numpy 环境(CNN_PYTHONPATH=...)")
np = cnn.np

K = 10


def collect(limit=8000):
    """前向收集各阶段特征与统计量。"""
    X, Y, Xte, Yte = cnn.ensure_mnist()
    W, _ = cnn.load_model()
    n = min(limit, len(Yte))
    Xte, Yte = Xte[:n], Yte[:n]
    feats = {"pix": [], "pool2": [], "h": []}
    logits = []
    y = Yte
    # 通道平均激活(正确分类样本内)
    acc_c1 = np.zeros((8, K))
    acc_c2 = np.zeros((16, K))
    cnt_c = np.zeros(K, dtype=int)
    marg = []
    preds = []
    B = 256
    for s in range(0, n, B):
        xb = Xte[s:s + B][:, None, :, :]
        z, cache = cnn.forward(xb, W)
        x1, z1, a1, p1, z2, a2, p2, flat, z3, a3, _ = cache
        yb = Yte[s:s + B]
        # conv1 通道平均激活 (B,8) — 28×28 平均
        m1 = a1.mean(axis=(2, 3))                      # (B,8)
        m2 = a2.mean(axis=(2, 3))                      # (B,16)
        logits.append(z)
        feats["pix"].append(xb.reshape(len(xb), -1))
        feats["pool2"].append(p2.reshape(len(xb), -1))
        feats["h"].append(a3)
        p = z.argmax(axis=1)
        preds.append(p)
        top = z.max(axis=1)
        zs = np.sort(z, axis=1)
        marg.extend((top - zs[:, -2]).tolist())
        for t in range(len(yb)):
            if p[t] == yb[t]:
                cls = int(yb[t])
                acc_c1[:, cls] += m1[t]
                acc_c2[:, cls] += m2[t]
                cnt_c[cls] += 1
    for k2 in feats:
        feats[k2] = np.concatenate(feats[k2], axis=0).astype(np.float64)
    logits = np.concatenate(logits, axis=0)
    preds = np.concatenate(preds)
    marg = np.array(marg)
    acc_c1 = acc_c1 / np.maximum(cnt_c, 1)[None, :]
    acc_c2 = acc_c2 / np.maximum(cnt_c, 1)[None, :]
    cm = np.zeros((K, K), dtype=int)
    for t in range(n):
        cm[int(y[t]), int(preds[t])] += 1
    return dict(Xte=Xte, Yte=Yte, W=W, feats=feats, logits=logits,
                preds=preds, marg=marg, y=y, cm=cm,
                c1mean=acc_c1, c2mean=acc_c2, cnt=cnt_c)


def ridge_split_acc(F, yy, frac=0.6, lam=1e-3):
    """岭回归读头分半准确率(在给定特征上的线性可分度)。"""
    m = int(len(yy) * frac)
    A = F[:m]
    Y = np.zeros((m, K), dtype=np.float64)
    Y[np.arange(m), yy[:m]] = 1.0
    Xt = F[m:]
    AtA = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    B = np.linalg.solve(AtA, A.T @ Y)
    p = (Xt @ B).argmax(axis=1)
    return float((p == yy[m:]).mean())


def centroid_split_acc(F, yy, frac=0.6):
    m = int(len(yy) * frac)
    mus = []
    for c in range(K):
        mus.append(F[:m][yy[:m] == c].mean(axis=0))
    mus = np.array(mus)
    Xt = F[m:]
    d = Xt @ mus.T
    p = d.argmax(axis=1)
    return float((p == yy[m:]).mean())


def anova_f(F1d, yy):
    """单通道判别力: 组间/组内方差比(F 统计量主体)。"""
    gb = F1d.shape[0] / K
    overall = F1d.mean()
    vb = sum(((F1d[yy == c].mean() - overall) ** 2) for c in range(K)) / (K - 1)
    vw = sum(((F1d[yy == c] - F1d[yy == c].mean()) ** 2).sum() for c in range(K))
    vw = vw / max(1, len(F1d) - K)
    return float(vb / (vw + 1e-9))


def run(limit=8000):
    R = collect(limit)
    y, F = R["Yte"], R["feats"]
    # ---- M1 线性可分度 ----
    stages = {}
    for name, Fx in F.items():
        stages[name] = {
            "ridge": round(ridge_split_acc(Fx, y), 4),
            "centroid": round(centroid_split_acc(Fx, y), 4),
        }
    # ---- M4 混淆/margin ----
    cm = R["cm"].tolist()
    acc = float((R["preds"] == y).mean())
    marg = R["marg"]
    pairs = []
    for i in range(K):
        for j in range(K):
            if i != j:
                pairs.append((int(cm[i][j]), int(i), int(j)))
    pairs.sort(reverse=True)
    conf_pairs = [[t[2], t[1], t[0]] for t in pairs[:6]]
    # ---- M5 激活稀疏度(各阶段一次全量估计) ----
    W = R["W"]
    act = {}
    ztot = tot1 = 0
    for s in range(0, len(y), 256):
        z, cache = cnn.forward(R["Xte"][s:s + 256][:, None, :, :], W)
        x1, z1, a1, p1, z2, a2, p2, flat, z3, a3, _ = cache
        act1 = (a1 > 0).mean()
        act2 = (a2 > 0).mean()
        act3 = (a3 > 0).mean()
        # 汇总
        ztot += act3
        tot1 += act1
    act["relu_rate"] = {"conv1": round(float(tot1 / 40), 4),
                        "conv2": round(float(ztot / 40), 4)}
    # ---- M3 决策方向几何 ----
    W2 = W["f2w"]                                  # (10,120)
    norm2 = np.linalg.norm(W2, axis=1).tolist()
    cos2 = []
    for i in range(K):
        row = []
        for j in range(K):
            ci = float(W2[i] @ W2[j] / (np.linalg.norm(W2[i]) * np.linalg.norm(W2[j]) + 1e-9))
            row.append(round(ci, 3))
        cos2.append(row)
    # 类中心(120 维)余弦, 用一半样本
    Fh = F["h"]
    mus = np.array([Fh[y == c].mean(axis=0) for c in range(K)])
    cosmus = []
    for i in range(K):
        row = []
        for j in range(K):
            ci = float(mus[i] @ mus[j] / (np.linalg.norm(mus[i]) * np.linalg.norm(mus[j]) + 1e-9))
            row.append(round(ci, 3))
        cosmus.append(row)
    near = []
    for i in range(K):
        for j in range(i + 1, K):
            near.append([cos2[i][j], i, j])
    near.sort(reverse=True)
    dir_close = [[b, a, v] for v, a, b in near[:4]]
    # ---- M2 通道判别力(F) ----
    ch = {}
    for layer, Fx in (("conv1", R["c1mean"]), ("conv2", R["c2mean"])):
        # F 用每图通道激活(需要重算: 此处近似用图像间 c1mean? 重算更严谨—从 collect 返回按图激活)
        ch[layer] = {"argmax": [], "score": []}
    # 直接从激活流重算 per-image 通道激活 F 与 top 数字
    m1all, m2all = [], []
    Wm = R["W"]
    for s in range(0, len(y), 256):
        z, cache = cnn.forward(R["Xte"][s:s + 256][:, None, :, :], Wm)
        _, _, a1, _, _, a2, *_ = cache
        m1all.append(a1.mean(axis=(2, 3)))
        m2all.append(a2.mean(axis=(2, 3)))
    m1all = np.concatenate(m1all)
    m2all = np.concatenate(m2all)
    res = {}
    for lname, Mat, meanT in (("conv1", m1all, R["c1mean"]), ("conv2", m2all, R["c2mean"])):
        scores = []
        for c in range(Mat.shape[1]):
            f = anova_f(Mat[:, c], y)
            scores.append(round(f, 2))
        top = meanT.argmax(axis=1).tolist()
        val = meanT.max(axis=1).tolist()
        res[lname] = {
            "scores": scores,
            "argmax_digit": top,
            "best_mean": [round(v, 4) for v in val],
            "mean_matrix": [[round(float(x), 4) for x in row] for row in meanT],
        }
    # margin 统计
    marr = R["marg"]
    out = {
        "n_test": len(y),
        "accuracy": round(acc, 4),
        "stages": stages,
        "margin": {
            "mean": round(float(marr.mean()), 4),
            "p10": round(float(np.percentile(marr, 10)), 4),
            "p50": round(float(np.percentile(marr, 50)), 4),
            "neg_frac": round(float((marr < 0).mean()), 4),
        },
        "confusion": cm,
        "conf_pairs": conf_pairs,
        "dir_norm": [round(v, 3) for v in norm2],
        "dir_cos": cos2,
        "class_cos": cosmus,
        "dir_close": dir_close,
        "channels": res,
        "relu_rate": {"conv1": round(float((m1all > 0).mean()), 4),
                      "conv2": round(float((m2all > 0).mean()), 4),
                      "fc1": act["relu_rate"]["conv2"]},
        "class_count": R["cnt"].tolist(),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("acc", out["accuracy"], "stages", out["stages"], "saved", OUT)
    return out


if __name__ == "__main__":
    run()
