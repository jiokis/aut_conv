# -*- coding: utf-8 -*-
"""定量实验: 从「算子×拉氏量」架构提炼 CNN 设计规则

实验 A —— 约束代价: 同一数据/轮数下, 把卷积滤波器每步投影为
    even(偶核=自伴=拉氏量可导出类) / odd(奇核=反自伴=陀螺类) vs 无约束
实验 B —— 复现性: 3 个随机种子, 统计卷积滤波器算子性质(偶/奇占比、
    正定数、方向差分占比、朝向均匀性)的均值±范围

结果写 data/exp_rules.json, 供 /api/cnn/rules 与页面引用。
python3 backend/exp_rules.py   (需要 numpy 环境, 见 cnn 模块说明)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cnn  # noqa: E402
import kernelmath as km  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(DATA_DIR, "exp_rules.json")


def op_stats(W):
    """单个模型的算子量化统计: conv1(8 个 2D 算子) + conv2 全部分量(128)。"""
    c1 = [km.quick_analyze(W["c1w"][f, 0].tolist()) for f in range(8)]
    comp = [km.quick_analyze(W["c2w"][o, ci].tolist())
            for o in range(16) for ci in range(8)]

    def agg(qs):
        n = len(qs)
        odd = [q["odd_frac"] for q in qs]
        even = [q["even_frac"] for q in qs]
        psd = sum(1 for q in qs if q["psd"])
        dirn = sum(1 for q in qs if q["filter_kind"] == "directional")
        oddheavy = sum(1 for q in qs if q["odd_frac"] > 0.5)
        # 朝向均匀性: 方向性滤波器的双倍角环形均值长度 R∈[0,1] (1=单向集中)
        th = [q["orient_deg"]["deg"] for q in qs
              if q["orient_deg"] and q["orient_deg"]["strength"] > 0.1]
        R = None
        if len(th) >= 3:
            x = sum(math.cos(2 * math.radians(a)) for a in th) / len(th)
            y = sum(math.sin(2 * math.radians(a)) for a in th) / len(th)
            R = round(math.hypot(x, y), 4)
        return {
            "n": n, "mean_even": round(sum(even) / n, 4),
            "mean_odd": round(sum(odd) / n, 4),
            "psd": psd, "directional": dirn,
            "odd_heavy": oddheavy, "orient_R": R,
            "odd_min": round(min(odd), 3), "odd_max": round(max(odd), 3),
        }

    return {"conv1": agg(c1), "conv2": agg(comp)}


def run():
    cnn.ensure_mnist()
    results = {"config": {}, "modes": [], "seeds": []}
    # ---- 实验 A: 约束代价(固定 seed) ----
    for mode in ("none", "even", "odd"):
        t0 = time.time()
        W, rep = cnn.train(subset=15000, epochs=2, batch=64, lr=2e-3,
                           seed=7, sym="" if mode == "none" else mode)
        st = op_stats(W)
        entry = {"mode": mode, "test_acc": rep["test_acc"],
                 "secs": round(time.time() - t0, 1), "stats": st}
        results["modes"].append(entry)
        print(f"[A] {mode:5s} test_acc={rep['test_acc']:.4f} "
              f"conv1 odd%={st['conv1']['mean_odd']:.3f} psd={st['conv1']['psd']} "
              f"dir={st['conv1']['directional']} | conv2 odd%={st['conv2']['mean_odd']:.3f} "
              f"psd={st['conv2']['psd']} ({entry['secs']}s)", flush=True)
    # ---- 实验 B: 多种子复现性(无约束) ----
    for sd in (101, 202, 303):
        t0 = time.time()
        W, rep = cnn.train(subset=15000, epochs=2, batch=64, lr=2e-3, seed=sd)
        st = op_stats(W)
        entry = {"seed": sd, "test_acc": rep["test_acc"], "stats": st,
                 "secs": round(time.time() - t0, 1)}
        results["seeds"].append(entry)
        print(f"[B] seed={sd} test_acc={rep['test_acc']:.4f} "
              f"conv1 odd%={st['conv1']['mean_odd']:.3f} "
              f"conv1 psd={st['conv1']['psd']} dir={st['conv1']['directional']} "
              f"orient_R={st['conv1']['orient_R']} | "
              f"conv2 odd%={st['conv2']['mean_odd']:.3f} psd={st['conv2']['psd']}", flush=True)

    def agg_runs(runs, key="conv1"):
        vals = [r["stats"][key] for r in runs]
        out = {}
        for k in ("mean_odd", "mean_even", "psd", "directional", "odd_heavy", "n"):
            vs = [v[k] for v in vals]
            out[k] = {"mean": round(sum(vs) / len(vs), 4),
                      "lo": min(vs), "hi": max(vs)} if vs else None
        Rs = [v["orient_R"] for v in vals if v["orient_R"] is not None]
        out["orient_R"] = {"mean": round(sum(Rs) / len(Rs), 4),
                           "lo": min(Rs), "hi": max(Rs)} if Rs else None
        return out

    results["agg_seeds_conv1"] = agg_runs(results["seeds"], "conv1")
    results["agg_seeds_conv2"] = agg_runs(results["seeds"], "conv2")
    results["config"] = {"subset": 15000, "epochs": 2, "lr": 2e-3,
                         "acc_base_seed7": results["modes"][0]["test_acc"]}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    run()
