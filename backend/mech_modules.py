#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模块作用机理 —— 系统消融实验(module ablation)

目标: 对这个最简单的 MNIST CNN, 定量回答“每个模块到底起什么作用”。
方法: 固定数据/轮数/优化器(单变量对照), 分别替换或关闭一个模块, 比较测试准确率;
      再在训练好的基线上做推理期干预(关 ReLU / 换池化 / 对称投影 / 打乱核 /
      置零通道 / 遮挡输入 / 梯度显著性), 观察精度变化与信息流向。

训练期消融(每个变体独立训练):
  base              原架构: Conv8·5²→P2→Conv16·5²→P2→FC120→FC10 (dropout 0.25)
  no_dropout        去掉 Dropout
  no_hidden         去掉隐藏层 FC120 (784→10)
  mlp_no_conv       去掉全部卷积(784→120→10 全连接)
  conv1_only        只保留一层卷积(8 个 5×5)
  no_relu           去掉所有 ReLU(整网线性)
  avg_pool          2×2 最大池化 → 平均池化
  stride_downsample 最大池化 → 步长 2 卷积(学习式降采样)
  even_kernels      每步把卷积核投影为偶核(自伴 ⇒ 可导出二次拉氏量)
  frozen_random_conv 卷积核冻结为随机值, 只训练分类头(检验“卷积特征是否需要学习”)

推理期消融(冻结 base 权重):
  relu_off@conv1/conv2/fc1, avg_pool_infer, even_project_infer,
  shuffle_kernels(核内位置打乱, 对照), zero_ch(逐通道置零), occlusion(中心/边缘遮挡),
  saliency(真类 logit 对输入的梯度, 每类平均)

输出: data/modules.json   (页面 ⑥ 与文档使用)
用法: python3 backend/mech_modules.py [--subset 30000] [--epochs 3] [--test 2000] [--seeds 2026]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnn  # noqa: E402

OUT = os.path.join(cnn.DATA_DIR, "modules.json")
KS, PAD = 5, 2


# --------------------------------------------------------------------- 网络
class FlexNet(nn.Module):
    """可配置 CNN: 深度/隐藏层/非线性/池化方式/Dropout 自由组合。"""

    def __init__(self, depth=2, hidden=120, dropout=0.25, pool="max", nonlinear=True):
        super().__init__()
        self.depth, self.pool, self.nonlinear = depth, pool, nonlinear
        if depth >= 1:
            self.c1 = nn.Conv2d(1, 8, KS, padding=PAD)
        if depth >= 2:
            self.c2 = nn.Conv2d(8, 16, KS, padding=PAD)
        if pool == "stride":  # 学习式降采样(步长 2 卷积)替代池化
            if depth >= 1:
                self.s1 = nn.Conv2d(8, 8, 3, stride=2, padding=1)
            if depth >= 2:
                self.s2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        if pool == "none":
            fin = {2: 28 * 28 * 16, 1: 28 * 28 * 8, 0: 784}[depth]
        else:
            fin = {2: 7 * 7 * 16, 1: 14 * 14 * 8, 0: 784}[depth]
        self.fc1 = nn.Linear(fin, hidden) if hidden > 0 else None
        self.fc2 = nn.Linear(hidden if hidden > 0 else fin, 10)
        self.drop = nn.Dropout(dropout)

    def down(self, h, which):
        if self.pool == "max":
            return F.max_pool2d(h, 2)
        if self.pool == "avg":
            return F.avg_pool2d(h, 2)
        if self.pool == "none":
            return h
        return getattr(self, "s%d" % which)(h)  # stride 卷积

    def forward(self, x):
        h = x
        if self.depth >= 1:
            h = self.down(self.act(self.c1(h)), 1)
        if self.depth >= 2:
            h = self.down(self.act(self.c2(h)), 2)
        v = h.flatten(1)
        v = self.drop(v)
        if self.fc1 is not None:
            v = self.act(self.fc1(v))
            v = self.drop(v)
            return self.fc2(v)
        return self.fc2(v)

    def act(self, h):
        return F.relu(h) if self.nonlinear else h


def even_project(model, mode="even"):
    """把卷积核投影为偶核 k←(k±k̄)/2 (自伴/反自伴), 就地。"""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name and p.dim() == 4:
                f = torch.flip(p, dims=[2, 3])
                p.copy_((p + f) * 0.5 if mode == "even" else (p - f) * 0.5)


# --------------------------------------------------------------------- 数据
def load(subset, test_n, seed):
    Xtr, Ytr, Xte, Yte = cnn.ensure_mnist()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(subset, len(Xtr)), replace=False)
    Xs, Ys = Xtr[idx], Ytr[idx]
    nval = 1000
    Xv, Yv = Xs[:nval], Ys[:nval]
    Xt, Yt = Xs[nval:], Ys[nval:]
    T = lambda a: torch.tensor(a.astype(np.float32))
    return (T(Xt)[:, None], torch.tensor(Yt.astype(np.int64)),
            T(Xv)[:, None], torch.tensor(Yv.astype(np.int64)),
            T(Xte[:test_n])[:, None], torch.tensor(Yte[:test_n].astype(np.int64)))


def test_acc(model, X, Y, bs=512):
    model.eval()
    ok = 0
    with torch.no_grad():
        for s in range(0, len(X), bs):
            ok += int((model(X[s:s + bs]).argmax(1) == Y[s:s + bs]).sum())
    return ok / len(X)


# --------------------------------------------------------------------- 训练
def train_variant(name, cfg, data, args, note=""):
    Xt, Yt, Xv, Yv, Xte, Yte = data
    torch.manual_seed(args.seed)
    net_kw = {k: v for k, v in cfg.items() if k not in ('even', 'frozen_conv')}
    model = FlexNet(**net_kw)
    if cfg.get("frozen_conv"):
        for n, p in model.named_parameters():
            if p.dim() == 4:
                p.requires_grad_(False)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 2), gamma=0.5)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    bs = args.batch
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt) - bs + 1, bs):
            i = perm[s:s + bs]
            opt.zero_grad()
            loss = crit(model(Xt[i]), Yt[i])
            loss.backward()
            opt.step()
            if cfg.get("even"):
                even_project(model, "even")
        sched.step()
    acc = test_acc(model, Xte, Yte)
    vox = test_acc(model, Xv, Yv)
    print(f"  [{name:18s}] test={acc:.4f} val={vox:.4f} params={params:6d} {time.time()-t0:5.1f}s {note}",
          flush=True)
    return model, {"name": name, "note": note, "test_acc": round(acc, 4), "val_acc": round(vox, 4),
                   "params": params, "secs": round(time.time() - t0, 1), "config": cfg}


# --------------------------------------------------------------------- 推理期工具
def fwd_ablated(state, x, opt=None):
    """带干预的函数式前向; state: 权重 dict(torch 张量)"""
    o = opt or {}
    def act(h, key):
        return F.relu(h) if o.get("relu_" + key, True) else h
    def pool(h):
        return F.avg_pool2d(h, 2) if o.get("avg_pool") else F.max_pool2d(h, 2)
    w = state
    h1 = act(F.conv2d(x, w["c1w"], w["c1b"], padding=PAD), "conv1")
    p1 = pool(h1)
    if o.get("zero_ch1") is not None:
        p1 = p1.clone(); p1[:, o["zero_ch1"]] = 0
    h2 = act(F.conv2d(p1, w["c2w"], w["c2b"], padding=PAD), "conv2")
    p2 = pool(h2)
    if o.get("zero_ch2") is not None:
        p2 = p2.clone(); p2[:, o["zero_ch2"]] = 0
    v = p2.flatten(1)
    h = act(F.linear(v, w["f1w"], w["f1b"]), "fc1")
    z = F.linear(h, w["f2w"], w["f2b"])
    return z, p2, v, h


def acc_ablated(state, X, Y, opt=None, bs=512):
    ok = 0
    with torch.no_grad():
        for s in range(0, len(X), bs):
            z, *_ = fwd_ablated(state, X[s:s + bs], opt)
            ok += int((z.argmax(1) == Y[s:s + bs]).sum())
    return ok / len(X)


def state_even(state):
    out = {}
    for k, v in state.items():
        if v.dim() == 4:
            out[k] = (v + torch.flip(v, dims=[2, 3])) * 0.5
        else:
            out[k] = v
    return out


def state_shuffle(state, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in state.items():
        if v.dim() == 4:
            f = v.shape[0]
            perm = torch.randperm(KS * KS, generator=g)
            out[k] = v.reshape(f, -1, KS * KS)[:, :, perm].reshape(v.shape)
        else:
            out[k] = v
    return out


def occlusion_test(state, X, Y, box=None, band=None):
    """遮挡输入区域后测精度: box=(y0,y1,x0,x1) 或 band='rows'/'cols'"""
    Xm = X.clone()
    if box:
        Xm[:, :, box[0]:box[1], box[2]:box[3]] = 0
    if band == "rows":
        Xm[:, :, 8:20, :] = 0
    if band == "cols":
        Xm[:, :, :, 8:20] = 0
    return acc_ablated(state, Xm, Y)


def channel_importance(state, X, Y, layer, base_acc):
    n = 8 if layer == 1 else 16
    res = []
    for c in range(n):
        opt = {"zero_ch1" if layer == 1 else "zero_ch2": c}
        a = acc_ablated(state, X, Y, opt)
        res.append({"ch": c, "acc": round(a, 4), "drop_pp": round((base_acc - a) * 100, 2)})
    return res


def saliency(state, X, Y, per_class=200):
    """真类 logit 对输入的梯度, 按类平均 → 14×14 下采样热图 + 中心/边缘占比"""
    maps = np.zeros((10, 14, 14), dtype=np.float64)
    cnt = np.zeros(10, dtype=int)
    for c in range(10):
        sel = (Y == c).nonzero(as_tuple=True)[0][:per_class]
        if len(sel) == 0:
            continue
        x = X[sel].clone().requires_grad_(True)
        z, *_ = fwd_ablated(state, x)
        score = z.gather(1, Y[sel].view(-1, 1)).sum()
        g = torch.autograd.grad(score, x)[0].abs()
        gm = g.mean(dim=1).detach().numpy()                      # (n,28,28)
        down = gm.reshape(len(sel), 14, 2, 14, 2).mean(axis=(2, 4))
        maps[c] += down.sum(axis=0)
        cnt[c] += len(sel)
    for c in range(10):
        if cnt[c]:
            maps[c] /= cnt[c]
    tot = maps.sum() + 1e-9
    center = maps[:, 5:9, 5:9].sum() / tot
    border = (maps[:, :3, :].sum() + maps[:, -3:, :].sum() +
              maps[:, :, :3].sum() + maps[:, :, -3:].sum()) / tot
    return {
        "per_class": [[[round(float(v), 5) for v in row] for row in maps[c]] for c in range(10)],
        "center_ratio": round(float(center), 4),
        "border_ratio": round(float(border), 4),
    }


# --------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reuse", action="store_true", help="复用 modules.json 中的训练结果, 只重训 base 以补跑推理消融")
    args = ap.parse_args()

    data = load(args.subset, args.test, args.seed)
    Xt, Yt, Xv, Yv, Xte, Yte = data
    print(f"数据: 训练 {len(Xt)} / 验证 {len(Xv)} / 测试 {len(Xte)}; epochs={args.epochs}")

    VARIANTS = [
        ("base", dict(depth=2, hidden=120, dropout=0.25, pool="max", nonlinear=True), "原架构基线"),
        ("no_dropout", dict(depth=2, hidden=120, dropout=0.0, pool="max", nonlinear=True), "去掉 Dropout"),
        ("no_hidden", dict(depth=2, hidden=0, dropout=0.25, pool="max", nonlinear=True), "去掉隐藏层 FC120"),
        ("mlp_no_conv", dict(depth=0, hidden=120, dropout=0.25, pool="max", nonlinear=True), "去掉全部卷积(纯 MLP)"),
        ("conv1_only", dict(depth=1, hidden=120, dropout=0.25, pool="max", nonlinear=True), "只保留一层卷积"),
        ("no_relu", dict(depth=2, hidden=120, dropout=0.25, pool="max", nonlinear=False), "去掉所有 ReLU(整网线性)"),
        ("avg_pool", dict(depth=2, hidden=120, dropout=0.25, pool="avg", nonlinear=True), "最大池化→平均池化"),
        ("stride_downsample", dict(depth=2, hidden=120, dropout=0.25, pool="stride", nonlinear=True), "池化→步长2卷积"),
        ("even_kernels", dict(depth=2, hidden=120, dropout=0.25, pool="max", nonlinear=True, even=True), "卷积核投影为偶核(自伴)"),
        ("linear_all", dict(depth=2, hidden=120, dropout=0.25, pool="avg", nonlinear=False), "去 ReLU + 平均池化(近似线性系统)"),
        ("no_pool", dict(depth=2, hidden=120, dropout=0.25, pool="none", nonlinear=True), "完全不做池化/降采样"),
        ("frozen_random_conv", dict(depth=2, hidden=120, dropout=0.25, pool="max", nonlinear=True, frozen_conv=True), "卷积核冻结为随机值"),
    ]
    training = []
    base_state = None
    models = {}
    prev = None
    if args.reuse and os.path.isfile(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            print("复用已有训练结果(--reuse):", len(prev.get("training", [])), "个变体")
        except Exception:
            prev = None
    for name, cfg, note in VARIANTS:
        if prev and name != "base":
            continue
        model, rep = train_variant(name, cfg, data, args, note)
        training.append(rep)
        if name == "base":
            sd = model.state_dict()
            base_state = {"c1w": sd["c1.weight"].clone(), "c1b": sd["c1.bias"].clone(),
                          "c2w": sd["c2.weight"].clone(), "c2b": sd["c2.bias"].clone(),
                          "f1w": sd["fc1.weight"].clone(), "f1b": sd["fc1.bias"].clone(),
                          "f2w": sd["fc2.weight"].clone(), "f2b": sd["fc2.bias"].clone()}
        models[name] = model
    if prev:
        training = prev.get("training", training)
        args.epochs = (prev.get("config") or {}).get("epochs", args.epochs)

    print("推理期消融(base 权重冻结):")
    base_acc = acc_ablated(base_state, Xte, Yte)
    inference = [{"name": "baseline", "acc": round(base_acc, 4), "delta_pp": 0.0, "note": "基线"}]
    for key, label in (("conv1", "关闭 Conv1 后的 ReLU"), ("conv2", "关闭 Conv2 后的 ReLU"),
                       ("fc1", "关闭 FC1 后的 ReLU")):
        a = acc_ablated(base_state, Xte, Yte, {"relu_" + key: False})
        inference.append({"name": "relu_off@" + key, "acc": round(a, 4),
                          "delta_pp": round((a - base_acc) * 100, 2), "note": label})
    a = acc_ablated(base_state, Xte, Yte, {"avg_pool": True})
    inference.append({"name": "avg_pool_infer", "acc": round(a, 4),
                      "delta_pp": round((a - base_acc) * 100, 2), "note": "推理时改平均池化"})
    a = acc_ablated(state_even(base_state), Xte, Yte)
    inference.append({"name": "even_project_infer", "acc": round(a, 4),
                      "delta_pp": round((a - base_acc) * 100, 2), "note": "推理时把卷积核投影为偶核"})
    a = acc_ablated(state_shuffle(base_state), Xte, Yte)
    inference.append({"name": "shuffle_kernels", "acc": round(a, 4),
                      "delta_pp": round((a - base_acc) * 100, 2), "note": "核内像素位置打乱(对照)"})
    for name, label in (("center", "遮挡中心 8×8"), ("border", "遮挡四周 3px")):
        box = (10, 18, 10, 18) if name == "center" else None
        Xm = Xte.clone()
        if box:
            Xm[:, :, box[0]:box[1], box[2]:box[3]] = 0
        else:
            Xm[:, :, :3, :] = 0; Xm[:, :, -3:, :] = 0; Xm[:, :, :, :3] = 0; Xm[:, :, :, -3:] = 0
        a = acc_ablated(base_state, Xm, Yte)
        inference.append({"name": "occlusion_" + name, "acc": round(a, 4),
                          "delta_pp": round((a - base_acc) * 100, 2), "note": label})
    # 用“填充墨迹”而非置零: 边缘本来就是背景, 置零等于没遮挡
    for name, note in (("center_ones", "中心 8×8 填满墨迹(破坏笔画)"),
                       ("border_ones", "四周 3px 填满墨迹(边界伪证据)")):
        Xm = Xte.clone()
        if name == "center_ones":
            Xm[:, :, 10:18, 10:18] = 1.0
        else:
            Xm[:, :, :3, :] = 1.0; Xm[:, :, -3:, :] = 1.0; Xm[:, :, :, :3] = 1.0; Xm[:, :, :, -3:] = 1.0
        a = acc_ablated(base_state, Xm, Yte)
        inference.append({"name": "occlusion_" + name, "acc": round(a, 4),
                          "delta_pp": round((a - base_acc) * 100, 2), "note": note})
    for name, note in (("rows", "遮挡中间 12 行(8–20)"), ("cols", "遮挡中间 12 列(8–20)")):
        a = occlusion_test(base_state, Xte, Yte, band=name)
        inference.append({"name": "occlusion_" + name, "acc": round(a, 4),
                          "delta_pp": round((a - base_acc) * 100, 2), "note": note})
    for it in inference:
        print(f"  [{it['name']:20s}] acc={it['acc']:.4f} Δ={it['delta_pp']:+.2f}pp  {it['note']}")

    print("逐通道重要性(置零单通道后的精度):")
    ch1 = channel_importance(base_state, Xte, Yte, 1, base_acc)
    ch2 = channel_importance(base_state, Xte, Yte, 2, base_acc)
    print("  conv1:", [(c['ch'], c['drop_pp']) for c in ch1])
    print("  conv2 top5:", sorted([(c['ch'], c['drop_pp']) for c in ch2], key=lambda t: -t[1])[:5])

    print("显著性(真类 logit 对输入梯度, 按类平均):")
    sal = saliency(base_state, Xte, Yte)
    print(f"  中心区能量占比 {sal['center_ratio']:.3f} · 边缘占比 {sal['border_ratio']:.3f}")

    out = {
        "config": (prev or {}).get("config") or
                  {"subset": len(Xt), "epochs": args.epochs, "batch": args.batch,
                   "lr": args.lr, "test_n": len(Xte), "seed": args.seed,
                   "arch": "Conv8·5²→P2→Conv16·5²→P2→FC120→FC10"},
        "baseline": {"test_acc": round(base_acc, 4)},
        "training": training,
        "inference": inference,
        "channel_importance": {"conv1": ch1, "conv2": ch2},
        "saliency": sal,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
