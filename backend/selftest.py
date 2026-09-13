# -*- coding: utf-8 -*-
"""数值自检: python3 backend/selftest.py
验证 FFT 卷积 / 符号采样 / 分解与分类结论与解析值一致。"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernelmath as km

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAILS.append(name)


CROSS = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
LAP4 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
GAUSS = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
SOBELX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
DELTA = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

# --- 符号解析一致性 ---
for name, k, expect in [
    ("cross DC=4", CROSS, (0.0, 0.0, 4.0)),
    ("cross H(π,0)=0", CROSS, (math.pi, 0.0, 0.0)),
    ("cross H(π,π)=-4", CROSS, (math.pi, math.pi, -4.0)),
    ("lap4 H(0,0)=0", LAP4, (0.0, 0.0, 0.0)),
    ("lap4 H(π,π)=-8", LAP4, (math.pi, math.pi, -8.0)),
]:
    wx, wy, want = expect
    got = km.symbol_value(k, wx, wy).real
    check(f"analytic symbol {name}", abs(got - want) < 1e-9, f"got {got}")

r = km.classify(CROSS)
check("cross even180", r["parity"]["even180"])
check("cross d4", r["parity"]["d4"])
check("cross not psd", not r["psd"])
check("cross self_adjoint/lagrangian", r["lagrangian_derivable"])
sp = r["spectrum"]
check("cross re_min=-4", abs(sp["re_min"] + 4.0) < 1e-3, str(sp["re_min"]))
check("cross re_max=4", abs(sp["re_max"] - 4.0) < 1e-3, str(sp["re_max"]))
check("cross psd min-at-corner",
      abs(abs(sp["re_min_at"][0]) - math.pi) < 1e-2 and abs(abs(sp["re_min_at"][1]) - math.pi) < 1e-2,
      str(sp["re_min_at"]))
lf = r["laplacian"]
check("cross = 1·Δ4 + 4·δ", lf["exact"] and abs(lf["a"] - 1) < 1e-6 and abs(lf["b"] - 4) < 1e-6,
      f"a={lf['a']} b={lf['b']} res={lf['rel_residual']}")

rL = km.classify(LAP4)
check("lap4 even", rL["parity"]["even180"])
check("lap4 nsd", rL["nsd"], str(rL["spectrum"]["re_max"]))

rG = km.classify(GAUSS)
check("gauss psd", rG["psd"], f"min={rG['spectrum']['re_min']}")
check("gauss dc=16", abs(rG["spectrum"]["dc"] - 16) < 1e-9)

rS = km.classify(SOBELX)
check("sobelx not even", not rS["parity"]["even180"])
check("sobelx odd fraction > 0", rS["parity"]["odd_frac"] > 0.5, str(rS["parity"]["odd_frac"]))
check("sobelx not lagrangian", not rS["lagrangian_derivable"])

# --- FFT 卷积 vs 直接 O(N²M²) 卷积 ---
def direct_conv(k, u, pad=km.PAD):
    N = km.MESH_N
    P = km.PROBE_SIZE
    M = len(k)
    c = M // 2
    out = [[0.0] * P for _ in range(P)]
    for x in range(P):
        for y in range(P):
            s = 0.0
            for a in range(M):
                for b in range(M):
                    ux = pad + x - (a - c)
                    uy = pad + y - (b - c)
                    if 0 <= ux < N and 0 <= uy < N:
                        s += k[a][b] * u[ux][uy]
            out[x][y] = s
    return out

k5 = [[0, 0, 1, 0, 0], [0, 2, -1, 2, 0], [1, -1, 3, -1, 1], [0, 2, -1, 2, 0], [0, 0, 1, 0, 0]]
import random
rng = random.Random(7)
u64 = [[rng.uniform(-1, 1) for _ in range(km.PROBE_SIZE)] for _ in range(km.PROBE_SIZE)]
canvas = [[0.0 + 0.0j] * km.MESH_N for _ in range(km.MESH_N)]
for x in range(km.PROBE_SIZE):
    for y in range(km.PROBE_SIZE):
        canvas[km.PAD + x][km.PAD + y] = complex(u64[x][y])
Fk = km.fft2(km.embed_kernel(k5, km.MESH_N), inverse=False)
FX = km.fft2(canvas, inverse=False)
Yf = [[FX[i][j] * Fk[i][j] for j in range(km.MESH_N)] for i in range(km.MESH_N)]
yv = km.fft2(Yf, inverse=True)
fft_conv = [[yv[km.PAD + x][km.PAD + yy].real for yy in range(km.PROBE_SIZE)] for x in range(km.PROBE_SIZE)]
direct = direct_conv(k5, [[v.real for v in row] for row in canvas])
mxerr = max(abs(fft_conv[x][y] - direct[x][y])
            for x in range(km.PROBE_SIZE) for y in range(km.PROBE_SIZE))
check("FFT 卷积 vs 直接卷积 maxerr", mxerr < 1e-9, f"err={mxerr:.3e}")

# --- δ 核的脉冲响应 = 核本身(中心对齐) ---
probes = km.run_probes(DELTA)
imp = next(p for p in probes if p["id"] == "impulse")
def ds48(src):
    return [[src[int(round(i * 63 / 47))][int(round(j * 63 / 47))] for j in range(48)] for i in range(48)]
delta64 = [[0.0] * 64 for _ in range(64)]
delta64[32][32] = 1.0
kd = ds48(delta64)
err = max(abs(imp["output"][i][j] - kd[i][j]) for i in range(48) for j in range(48))
check("δ 核: K*δ = K", err < 1e-9, f"err={err}")

# --- 参数化族生成 ---
for fam, params in [("gaussian", {"sigma": 1.0}), ("exponential", {"lambda": 1.5}),
                    ("box", {"radius": 1.0}), ("heat", {"t": 0.6}), ("rational", {"a": 1.5, "p": 1})]:
    kk, note = km.parametric_kernel(fam, 9, params)
    chk = km.classify(kk)
    if fam in ("gaussian", "exponential", "heat", "rational"):
        check(f"parametric {fam}: even", chk["parity"]["even180"])
        check(f"parametric {fam}: psd", chk["psd"], f"min={chk['spectrum']['re_min']}")
    check(f"parametric {fam} dims", len(kk) == 9)

# --- 随机生成类别 ---
for kind in ("general", "even", "odd", "psd"):
    kk = km.random_kernel(5, kind, seed=3)
    chk = km.classify(kk)
    if kind == "even":
        check("random even even180", chk["parity"]["even180"])
    if kind == "odd":
        check("random odd odd180", chk["parity"]["odd180"])
    if kind == "psd":
        check("random psd psd", chk["psd"], f"min={chk['spectrum']['re_min']}")

# --- 复化(虚数场)快速分析 ---
cp_cross = km.complex_quick(CROSS)
check("复化: 偶核动能占比=0", cp_cross["kinetic_share"] == 0.0)
cp_sobel = km.complex_quick(SOBELX)
check("复化: 纯奇核动能占比=1", abs(cp_sobel["kinetic_share"] - 1.0) < 1e-9, str(cp_sobel["kinetic_share"]))
check("复化: 奇核实谱存在(动能化)", abs(cp_sobel["hc_max"]) > 1e-9 and abs(cp_sobel["hc_min"]) > 1e-9,
      f"hc=[{cp_sobel['hc_min']},{cp_sobel['hc_max']}]")
qsp = km.quick_spectrum(CROSS)
check("复化: 偶核 hc 谱=Re 谱", abs(cp_cross["hc_max"] - qsp["re_max"]) < 1e-6)
_rng = random.Random(11)
gen = [[_rng.uniform(-2, 2) for _ in range(5)] for _ in range(5)]
qa = km.quick_analyze(gen)
cp = km.complex_quick(gen)
check("复化: 动能占比≈奇部占比", abs(cp["kinetic_share"] - qa["odd_frac"]) < 5e-5,
      f"{cp['kinetic_share']} vs {qa['odd_frac']}")

print("\n结果:", "全部通过 ✔" if not FAILS else f"{len(FAILS)} 项失败 ✘")
sys.exit(1 if FAILS else 0)
