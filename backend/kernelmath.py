# -*- coding: utf-8 -*-
"""
算子(卷积核)生成与分析 —— 数学引擎
纯 Python 3, 零第三方依赖。复现 DeepSeek 分享对话《卷积算子的等变性与拉氏函数》中的
完整分析管线:

  平移等变(L1) -> 自伴/偶核(L2, 可导出二次拉格朗日泛函) -> 正定(L3, 凸能量)
  -> 微分算子格林核(L4) -> 群等变核(L5)

以及: 偶/奇分解、谱分解(势能/耗散/陀螺)、Rayleigh 耗散函数、频率响应、滤波行为探针。
按对话偏好, 全程实数框架(不引入复数化, 奇部 -> 陀螺项; 负定偶部 -> 耗散项)。
"""
from __future__ import annotations

import cmath
import math
import random

EPS = 1e-9
TOL = 1e-7          # 判定用相对容差
MESH_N = 128        # 谱分析网格(2 的幂)
PROBE_SIZE = 64     # 探针域边长
PAD = (MESH_N - PROBE_SIZE) // 2   # 探针在画布中的起始偏移


# --------------------------------------------------------------------------
# 1. FFT(基 2 迭代)与二维 FFT
# --------------------------------------------------------------------------
def fft1(x, inverse: bool = False):
    """一维 FFT, 长度须为 2 的幂。inverse=True 时归一化 1/N。"""
    n = len(x)
    a = [complex(v) for v in x]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = 2.0 * math.pi / length
        wlen = complex(math.cos(ang), -math.sin(ang) if not inverse else math.sin(ang))
        half = length >> 1
        for start in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * w
                a[start + k] = u + v
                a[start + k + half] = u - v
                w *= wlen
        length <<= 1
    if inverse:
        for i in range(n):
            a[i] /= n
    return a


def fft2(A, inverse: bool = False):
    """二维 FFT, A 为 list[list[complex]], 行列数均须为 2 的幂。"""
    R = len(A)
    C = len(A[0])
    a = [row[:] for row in A]
    for r in range(R):
        a[r] = fft1(a[r], inverse)
    for c in range(C):
        col = fft1([a[r][c] for r in range(R)], inverse)
        for r in range(R):
            a[r][c] = col[r]
    return a


# --------------------------------------------------------------------------
# 2. 频域符号(频率响应)
# --------------------------------------------------------------------------
def kernel_size(k):
    M = len(k)
    assert M % 2 == 1, "内核边长必须为奇数"
    for row in k:
        assert len(row) == M, "内核必须是正方形"
    return M


def embed_kernel(k, N: int):
    """把边长 M 的核嵌入 N×N 周期格点, 核中心放在格点索引 0 处。
    如此做 2D-FFT 得到的就是符号 H(ω)=Σ k[a,b] exp(-i ω·(a,b)),
    且 FFT 卷积可直接使用同一嵌入。"""
    M = kernel_size(k)
    c = M // 2
    B = [[0.0 + 0.0j for _ in range(N)] for _ in range(N)]
    for a in range(M):
        for b in range(M):
            v = k[a][b]
            if v != 0.0:
                B[(a - c) % N][(b - c) % N] = complex(v)
    return B


def omega_of_index(i: int, N: int) -> float:
    """显示用索引 -> 角频率(弧度)。中心(N/2) 对应 ω=0, 范围 [-π, π)。"""
    return 2.0 * math.pi * (i - N // 2) / N


def index_of_omega(w: float, N: int) -> int:
    """把 [-π,π) 内的角频率映射回显示索引(用于在谱网格上找极值点)。"""
    i = int(round(w * N / (2.0 * math.pi))) + N // 2
    return i % N


def symbol_value(k, wx: float, wy: float) -> complex:
    """解析地求符号 H(wx, wy) = Σ k[a,b] exp(-i (wx(a-c)+wy(b-c)))。"""
    M = len(k)
    c = M // 2
    s = 0.0 + 0.0j
    for a in range(M):
        xa = (a - c) * wx
        row = k[a]
        for b in range(M):
            if row[b]:
                s += row[b] * cmath.exp(-1j * (xa + (b - c) * wy))
    return s


def spectral_mesh(k):
    """返回 (N,N) 复矩阵: 显示索引中心为 ω=0 的符号采样网格。"""
    return spectral_mesh_n(k, MESH_N)


def analyze_spectrum(k):
    """基于 MESH_N 网格的符号统计(对带限核, 网格采样即精确)。"""
    N = MESH_N
    D = spectral_mesh(k)
    half = N // 2
    re_min = re_max = None
    im_abs_max = 0.0
    mag_max = 0.0
    min_wi = min_wj = max_wi = max_wj = 0
    # 顺便抽取 64×64 降采样网格(偶数索引)
    ds = N // 64
    mesh_re = []
    mesh_mag = []
    for di in range(0, N, ds):
        row_re = []
        row_mag = []
        for dj in range(0, N, ds):
            v = D[di][dj]
            row_re.append(round(v.real, 5))
            row_mag.append(round(abs(v), 5))
            r = v.real
            if re_min is None or r < re_min:
                re_min = r
                min_wi, min_wj = di, dj
            if re_max is None or r > re_max:
                re_max = r
                max_wi, max_wj = di, dj
            im = abs(v.imag)
            if im > im_abs_max:
                im_abs_max = im
            m = abs(v)
            if m > mag_max:
                mag_max = m
        mesh_re.append(row_re)
        mesh_mag.append(row_mag)
    if re_min is None:
        re_min = re_max = 0.0
    w0 = omega_of_index(0, N)
    return {
        "N": N,
        "re_min": round(re_min, 6),
        "re_max": round(re_max, 6),
        "re_min_at": [round(omega_of_index(min_wi, N), 4), round(omega_of_index(min_wj, N), 4)],
        "re_max_at": [round(omega_of_index(max_wi, N), 4), round(omega_of_index(max_wj, N), 4)],
        "im_abs_max": round(im_abs_max, 6),
        "mag_max": round(mag_max, 6),
        "dc": round(D[half][half].real, 6),
        "h_pi_pi": round(symbol_value(k, math.pi, math.pi).real, 6),
        "h_pi_0": round(symbol_value(k, math.pi, 0.0).real, 6),
        "h_0_pi": round(symbol_value(k, 0.0, math.pi).real, 6),
        "h_pi2_pi2": round(symbol_value(k, math.pi / 2, math.pi / 2).real, 6),
        "mesh_re": mesh_re,
        "mesh_mag": mesh_mag,
        # 解析切片(高分辨率曲线)
        "slice_x": _slice(k, 0),
        "slice_diag": _slice(k, math.pi / 4),
        "omega_axis": [round(math.pi * t, 4) for t in _linspace(-1.0, 1.0, 129)],
    }


def _linspace(a, b, n):
    if n == 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _slice(k, angle):
    """沿方向 (cosθ, sinθ) 过原点的符号实部切片。"""
    ca = math.cos(angle)
    sa = math.sin(angle)
    out = []
    for t in _linspace(-1.0, 1.0, 129):
        v = symbol_value(k, math.pi * t * ca, math.pi * t * sa)
        out.append(round(v.real, 5))
    return out


# --------------------------------------------------------------------------
# 3. 对称性 / 奇偶分解
# --------------------------------------------------------------------------
def decompose_parity(k):
    M = len(k)
    c = M // 2
    ks = [[0.0] * M for _ in range(M)]
    ka = [[0.0] * M for _ in range(M)]
    sum_e2 = sum_o2 = 0.0
    even180 = True
    odd180 = True
    for i in range(M):
        for j in range(M):
            v = k[i][j]
            w = k[2 * c - i][2 * c - j]
            if abs(v - w) > TOL:
                even180 = False
            if abs(v + w) > TOL:
                odd180 = False
            s = 0.5 * (v + w)
            a = 0.5 * (v - w)
            ks[i][j] = round(s, 9)
            ka[i][j] = round(a, 9)
            sum_e2 += s * s
            sum_o2 += a * a
    # 对称性: 反射轴 / 90° 旋转(D4)
    def all_sym(pred):
        for i in range(M):
            for j in range(M):
                if not pred(i, j):
                    return False
        return True
    ref_v = all_sym(lambda i, j: abs(k[i][j] - k[i][2 * c - j]) <= TOL)   # 左右镜像
    ref_h = all_sym(lambda i, j: abs(k[i][j] - k[2 * c - i][j]) <= TOL)   # 上下镜像
    ref_d = all_sym(lambda i, j: abs(k[i][j] - k[j][i]) <= TOL)           # 主对角线
    ref_ad = all_sym(lambda i, j: abs(k[i][j] - k[2 * c - j][2 * c - i]) <= TOL)  # 副对角线
    d4 = all_sym(lambda i, j: abs(k[i][j] - k[2 * c - j][i]) <= TOL)      # 90° 旋转
    total = sum_e2 + sum_o2
    if total <= 0:
        return {
            "m": M, "type": "zero", "even180": True, "odd180": True,
            "d4": True, "ref_v": True, "ref_h": True, "ref_d": True, "ref_ad": True,
            "even_frac": 0.0, "odd_frac": 0.0, "ks": ks, "ka": ka,
            "zero": True,
        }
    return {
        "m": M,
        "type": "even" if even180 else ("odd" if odd180 else "general"),
        "even180": bool(even180), "odd180": bool(odd180),
        "d4": bool(d4), "ref_v": bool(ref_v), "ref_h": bool(ref_h),
        "ref_d": bool(ref_d), "ref_ad": bool(ref_ad),
        "even_frac": round(sum_e2 / total, 6),
        "odd_frac": round(sum_o2 / total, 6),
        "ks": ks, "ka": ka,
        "zero": False,
    }


def decompose_spectral(k):
    """对自伴部分(偶核)做谱分解: H = H+ + H-, 其中 H+≥0 (势能), -H-≥0 (耗散 D)。
    用 IDFT 把钳位谱投影回支撑域 M×M (最小二乘意义, 会有振铃, 标注为近似)。"""
    M = len(k)
    ks = decompose_parity(k)["ks"]
    N = MESH_N
    c = M // 2
    half = N // 2
    F = fft2(embed_kernel(ks, N), inverse=False)
    # 对 FFT 索引系重建偶符号(直接是实值)
    Hp = [[0.0 + 0.0j for _ in range(N)] for _ in range(N)]
    Hn = [[0.0 + 0.0j for _ in range(N)] for _ in range(N)]
    pot_sum = diss_sum = 0.0
    for i in range(N):
        for j in range(N):
            r = F[i][j].real
            if r >= 0:
                Hp[i][j] = complex(r)
                pot_sum += r
            else:
                Hn[i][j] = complex(r)
                diss_sum += -r
    kp = _idft_crop(Hp, N, M)
    kn = _idft_crop(Hn, N, M)   # 负定自伴部分(符号 ≤ 0)
    kd = [[-v for v in row] for row in kn]  # D = -K_neg, 正定耗散核
    return {
        "pot_energy_sum": round(pot_sum, 5),
        "diss_energy_sum": round(diss_sum, 5),
        "k_pos_approx": kp,
        "k_neg_approx": kn,
        "k_diss_approx": kd,
    }


def _idft_crop(H, N: int, M: int):
    """把钳位谱逆变换回空间, 裁出中心 M×M 窗口作为近似核。
    fft2(inverse=True) 已在两维各归一 1/N, 合计 1/N², 即标准 IDFT。"""
    c = M // 2
    g = fft2(H, inverse=True)
    # g 的行列索引系与 embed_kernel 相同: 中心在索引 0。
    out = [[0.0] * M for _ in range(M)]
    for a in range(M):
        for b in range(M):
            out[a][b] = round(g[(a - c) % N][(b - c) % N].real, 6)
    return out


# --------------------------------------------------------------------------
# 4. 拉普拉斯拟合 与 滤波刻画
# --------------------------------------------------------------------------
def laplacian_fit(k):
    """最小二乘: K ≈ a·L4_m + b·δ (L4_m 为同尺寸 4-邻域离散拉普拉斯, 中心 -4)。"""
    M = len(k)
    c = M // 2
    L = [[0.0] * M for _ in range(M)]
    L[c][c] = -4.0
    if c >= 1:
        L[c - 1][c] = L[c + 1][c] = L[c][c - 1] = L[c][c + 1] = 1.0
    # 正规方程 (2×2)
    a11 = a12 = a22 = b1 = b2 = 0.0
    for i in range(M):
        for j in range(M):
            lv = L[i][j]
            kv = k[i][j]
            dv = 1.0 if (i == c and j == c) else 0.0
            a11 += lv * lv
            a12 += lv * dv
            a22 += dv * dv
            b1 += lv * kv
            b2 += dv * kv
    det = a11 * a22 - a12 * a12
    res = 1.0
    if abs(det) > EPS:
        ca = (b1 * a22 - b2 * a12) / det
        cb = (a11 * b2 - a12 * b1) / det
        sse = tot = 0.0
        for i in range(M):
            for j in range(M):
                e = k[i][j] - (ca * L[i][j] + (cb if i == c and j == c else 0.0))
                sse += e * e
                tot += k[i][j] * k[i][j]
        res = math.sqrt(sse / tot) if tot > 0 else 0.0
        return {"a": round(ca, 6), "b": round(cb, 6),
                "rel_residual": round(res, 6),
                "exact": res < 1e-6 and abs(cb - (k[c][c] + 4.0 * ca)) < 1e-6}
    return {"a": 0.0, "b": 0.0, "rel_residual": 1.0, "exact": False}


def filter_profile(k, sp):
    """粗略滤波类型刻画(基于符号的径向均值 + 解析方向采样)。"""
    N = MESH_N
    half = N // 2
    D = spectral_mesh(k)
    nb = 40
    bins = [0.0] * nb
    cnt = [0] * nb
    for i in range(N):
        for j in range(N):
            dx = (i - half) / half
            dy = (j - half) / half
            r = math.hypot(dx, dy)
            if r <= 1.0:
                b = int(r * (nb - 1))
                bins[b] += abs(D[i][j])
                cnt[b] += 1
    prof = [round(bins[b] / cnt[b], 5) if cnt[b] else 0.0 for b in range(nb)]
    low = sum(prof[:10]) / 10.0
    mid = sum(prof[15:26]) / 11.0
    high = sum(prof[32:]) / (len(prof) - 32)
    dc = sp["dc"]
    kind = "other"
    if abs(dc) <= 1e-6 and high > low * 2:
        kind = "highpass"      # 高通/差分类
    elif high <= low * 0.35 and low > 0:
        kind = "lowpass"
    elif high > low * 2.5 and low > 0:
        kind = "highpass"
    elif mid > low * 1.5 and mid > high * 1.5:
        kind = "bandpass"
    elif low > 0 and abs(low - high) <= max(low, high) * 0.15:
        kind = "flat"
    # 各向异性(解析角点采样)
    ax = abs(symbol_value(k, math.pi, 0.0)) + abs(symbol_value(k, 0.0, math.pi))
    dg = abs(symbol_value(k, math.pi, math.pi)) + abs(symbol_value(k, math.pi, -math.pi))
    anis = abs(ax - dg) / (ax + dg + 1e-12)
    return {"kind": kind, "radial": prof,
            "mean_low": round(low, 5), "mean_high": round(high, 5),
            "anisotropy": round(anis, 5)}


# --------------------------------------------------------------------------
# 5. 探针(滤波行为 / 能量语义)
# --------------------------------------------------------------------------
def make_probes():
    P = PROBE_SIZE
    cx = cy = P / 2.0   # 整数格点 P//2 = 32
    probes = [
        ("constant", "常数场 u≡1", lambda x, y: 1.0),
        ("ramp_x", "低频斜坡 u=x/16", lambda x, y: (x - (P - 1) / 2.0) / 16.0),
        ("sine_low", "低频正弦 sin(2πx/16)", lambda x, y: math.sin(2.0 * math.pi * x / 16.0)),
        ("checker", "棋盘模式 (-1)^(x+y)", lambda x, y: 1.0 if int(x + y) % 2 == 0 else -1.0),
        ("impulse", "中心脉冲 δ", lambda x, y: 1.0 if abs(x - cx) < 0.5 and abs(y - cy) < 0.5 else 0.0),
    ]
    return probes


def _canvas_probe(f, P):
    N = MESH_N
    X = [[0.0 + 0.0j for _ in range(N)] for _ in range(N)]
    for x in range(P):
        for y in range(P):
            X[PAD + x][PAD + y] = complex(f(x, y))
    return X


def _downsample48(src):
    """64×64 -> 48×48 (最近邻按比例重采样)。"""
    out = []
    for i in range(48):
        row = []
        si = int(round(i * 63 / 47))
        for j in range(48):
            sj = int(round(j * 63 / 47))
            row.append(round(src[si][sj], 4))
        out.append(row)
    return out


def run_probes(k):
    N = MESH_N
    P = PROBE_SIZE
    Fk = fft2(embed_kernel(k, N), inverse=False)
    results = []
    for pid, name, f in make_probes():
        X = _canvas_probe(f, P)
        FX = fft2(X, inverse=False)
        Yf = [[FX[i][j] * Fk[i][j] for j in range(N)] for i in range(N)]
        conv = fft2(Yf, inverse=True)
        # 统计与作用量(取探针盒内区域; 该区域卷积无周期卷绕)
        s = 0.0        # S[u]=½Σ u·(K*u)
        w = 0.0        # ⟨u, K u⟩
        e = 0.0
        out_min = out_max = None
        out_box = []
        in_box = []
        for x in range(P):
            orow = []
            irow = []
            for yy in range(P):
                u = X[PAD + x][PAD + yy].real
                v = conv[PAD + x][PAD + yy].real
                orow.append(round(v, 4))
                irow.append(round(u, 4))
                s += 0.5 * u * v
                w += u * v
                e += 0.5 * u * u
                if out_min is None or v < out_min:
                    out_min = v
                if out_max is None or v > out_max:
                    out_max = v
            out_box.append(orow)
            in_box.append(irow)
        results.append({
            "id": pid, "name": name,
            "S": round(s, 5),
            "work": round(w, 5),           # dE/dt = -work (梯度流 ∂u/∂t=-Ku, E=½‖u‖²)
            "energyE": round(e, 5),
            "out_min": round(out_min, 5) if out_min is not None else 0.0,
            "out_max": round(out_max, 5) if out_max is not None else 0.0,
            "input": _downsample48(in_box),
            "output": _downsample48(out_box),
        })
    return results


# --------------------------------------------------------------------------
# 6. 分类 / 结论
# --------------------------------------------------------------------------
def classify(k):
    """主分析入口。返回完整 JSON 安全的分析报告。"""
    M = kernel_size(k)
    sp = analyze_spectrum(k)
    par = decompose_parity(k)
    psd = False
    nsd = False
    # 正定性容差: 谱截断/大格点 IDFT 采样会带来 ~1e-4·max|H| 的数值噪声
    psd_tol = 1e-4 * max(abs(sp["re_min"]), abs(sp["re_max"]), 1e-6)
    if par["even180"]:
        psd = sp["re_min"] >= -psd_tol
        nsd = sp["re_max"] <= psd_tol
    indefinite = par["even180"] and not psd and not nsd
    definite = "psd" if psd else ("nsd" if nsd else
                                  ("indefinite" if par["even180"] else "complex_symbol"))
    spec = decompose_spectral(k)
    lf = laplacian_fit(k)
    fp = filter_profile(k, sp)
    probes = run_probes(k)

    # --- 设计空间层级定位 ---
    levels = {"L1_all": True, "L2_sym": bool(par["even180"]), "L3_pos": psd,
              "L5_group": bool(par["d4"])}
    if par["d4"]:
        levels["L5_group_note"] = "D4(含 90° 旋转 + 镜像) 等变"
    elif par["ref_v"] or par["ref_h"]:
        levels["L5_group_note"] = "反射(镜像)等变, 非 90° 旋转"

    # 汇总表(镜像对话最终表格)
    summary = [
        ("核尺寸", f"{M}×{M}"),
        ("中心值 k[0,0]", f"{k[M//2][M//2]:g}"),
        ("DC 增益 Σk", f"{sp['dc']:g}"),
        ("L² 范数", f"{round(math.sqrt(sum(v*v for row in k for v in row)), 5)}"),
        ("奇偶类型", {"even": "偶核(中心对称)", "odd": "奇核(反中心对称)",
                     "general": "一般核(含偶+奇)"}[par["type"]]),
        ("180° 旋转对称", "是" if par["even180"] else "否"),
        ("90° 旋转(D4)", "是" if par["d4"] else "否"),
        ("自伴(偶函数)", "是 → 可导出二次拉格朗日泛函" if par["even180"] else
         "否 → 不能由(实)拉格朗日量直接导出"),
        ("符号类型", "实值(自伴)" if par["even180"] else "复值(存在反自伴部分)"),
        ("谱范围 Re ĥ", f"[{sp['re_min']:g}, {sp['re_max']:g}]"),
        ("正定性", "正定(PSD)" if psd else ("负定(NSD)" if nsd else
                     ("不定(indefinite)" if indefinite else "含复谱(非自伴, 无实正定性)"))),
        ("能量泛函凸性", "凸(良定)" if psd else ("凹" if nsd else
                          ("非凸(存在负能量模式)" if indefinite else "不适用(非自伴)"))),
        ("频率响应 H(ω₁,ω₂)", f"{_fresp(sp)}"),
        ("与离散拉普拉斯关系", _lap_text(lf)),
        ("设计空间定位", _level_text(levels)),
    ]

    conclusions = []
    conclusions.append({"t": "info", "text": "卷积结构 ⇒ 平移等变(L1)自动满足: "
                        "任何 k 都给出平移等变算子, 等变性不限制核形状。"})
    if par["even180"]:
        conclusions.append({"t": "ok", "text": "偶核 k(x)=k(-x) ⇒ 卷积算子自伴, "
                            "可由二次拉格朗日泛函 S[u]=½∫u(k*u) 导出(δS/δu = k*u)。"})
    else:
        if par["type"] == "odd":
            conclusions.append({"t": "warn", "text": "纯奇核(反自伴) ⇒ 陀螺/相位项: "
                                "能量守恒但破坏时间反演, 不能由实拉格朗日量导出, "
                                "需辛结构/辅助变量容纳。"})
        else:
            conclusions.append({"t": "warn", "text": "一般核 = 偶部(自伴, 可导拉氏量)"
                                " + 奇部(反自伴, 陀螺项)。线性组合拉格朗日量最多只能覆盖偶部。"})
    if psd:
        conclusions.append({"t": "ok", "text": "符号 ĥ(ω) ≥ 0 ⇒ 落在正定锥 K_pos(L3): "
                            "能量凸、谱非负, 对应格林核 / 协方差函数 / RKHS 正定核。"})
    elif par["even180"]:
        conclusions.append({"t": "warn", "text": "自伴但非正定 ⇒ 能量泛函非凸, "
                            "存在负能量模式(谱 {:.4g} 处最负)。正谱部分=势能核, "
                            "负谱部分经 Rayleigh 耗散函数解释为耗散 D=-A_neg。" .format(sp["re_min"])})
    else:
        conclusions.append({"t": "info", "text": "符号为复值 ⇒ 实数框架下无正定性概念; "
                            "反自伴部分对应陀螺项(不耗能), 真正的耗散来自负定自伴部分。"})
    if par["odd_frac"] > 1e-4:
        conclusions.append({"t": "info", "text": "奇部能量占比 {:.1%} ⇒ 陀螺/方向性分量; "
                            "其平方 ⟨u,A_anti u⟩=0, 在梯度流中不改变 L² 能量。".format(par["odd_frac"])})
    if par["even180"] and sp["re_max"] <= 1e-9 and sp["re_min"] < -1e-9:
        conclusions.append({"t": "ok", "text": "负定自伴核 ⇒ -K 正定; 作为动力学 "
                            "∂u/∂t=-A u 的 A 部分使 L² 能量单调耗散, 即 Rayleigh 耗散 D 的来源。"})
    if lf["rel_residual"] < 1e-5:
        conclusions.append({"t": "ok", "text": "精确分解: K = {a:g}·Δ₄ + {b:g}·δ "
                            "(离散拉普拉斯 + 中心冲激)。".format(a=lf["a"], b=lf["b"])})
    elif lf["rel_residual"] < 0.35:
        conclusions.append({"t": "info", "text": "近似 K ≈ {a:g}·Δ₄ + {b:g}·δ, "
                            "相对残差 {r:.0%} —— 与微分算子的拉普拉斯核相近。"
                            .format(a=lf["a"], b=lf["b"], r=lf["rel_residual"])})
    # 探针结论
    chk = next((p for p in probes if p["id"] == "checker"), None)
    if chk is not None and chk["work"] < 0:
        conclusions.append({"t": "warn", "text": "棋盘探针作用量 S<0(⟨u,Ku⟩={:.4g})"
                            " ⇒ 存在负能量高频模式, 印证不定性。".format(chk["work"])})

    meta = {
        "dc": sp["dc"], "center": k[M // 2][M // 2],
        "sum_abs": round(sum(abs(v) for row in k for v in row), 5),
        "l2": round(math.sqrt(sum(v * v for row in k for v in row)), 5),
        "mass_center": _mass_center(k),
    }

    return {
        "meta": meta,
        "parity": par,
        "spectrum": sp,
        "definite": definite,
        "psd": psd, "nsd": nsd, "indefinite": indefinite,
        "self_adjoint": bool(par["even180"]),
        "lagrangian_derivable": bool(par["even180"]),
        "spec_split": spec,
        "laplacian": lf,
        "filter": fp,
        "probes": probes,
        "levels": levels,
        "summary": summary,
        "conclusions": conclusions,
    }


def _mass_center(k):
    M = len(k)
    c = M // 2
    sx = sy = wsum = 0.0
    for i in range(M):
        for j in range(M):
            v = abs(k[i][j])
            sx += (i - c) * v
            sy += (j - c) * v
            wsum += v
    if wsum <= 0:
        return [0.0, 0.0]
    return [round(sx / wsum, 4), round(sy / wsum, 4)]


def _fresp(sp):
    return f"2cos型 H(π,π)={sp['h_pi_pi']:g}; H(π,0)={sp['h_pi_0']:g}; DC={sp['dc']:g}" \
        if False else f"H(0,0)={sp['dc']:g}, H(π,0)={sp['h_pi_0']:g}, H(π,π)={sp['h_pi_pi']:g}"


def _lap_text(lf):
    if lf["rel_residual"] < 1e-5:
        return f"K = {lf['a']:g}·Δ₄ + {lf['b']:g}·δ (精确)"
    if lf["rel_residual"] < 0.35:
        return f"K ≈ {lf['a']:g}·Δ₄ + {lf['b']:g}·δ (残差 {lf['rel_residual']:.0%})"
    return "与离散拉普拉斯无显著线性关系"


def _level_text(levels):
    parts = []
    if levels.get("L3_pos"):
        parts.append("L3 正定锥(凸能量)")
    elif levels.get("L2_sym"):
        parts.append("L2 自伴子空间(可导出拉氏量)")
    if levels.get("L5_group"):
        parts.append("L5 群等变: " + levels.get("L5_group_note", "90°旋转"))
    if parts:
        return "; ".join(parts)
    return "L1 平移等变(一般卷积核)"


# --------------------------------------------------------------------------
# 7. 生成器
# --------------------------------------------------------------------------
def random_kernel(m: int, kind: str, intensity: float = 1.0, seed: int | None = None):
    """按类别生成随机核: general/even/odd/psd。数值取小整数 ±1 与强度噪声混合。"""
    rng = random.Random(seed)
    c = m // 2
    vals = [0.0] * (m * m)
    for t in range(m * m):
        r = rng.random()
        if r < 0.45:
            vals[t] = 0.0
        elif r < 0.70:
            vals[t] = 1.0
        elif r < 0.88:
            vals[t] = -1.0
        else:
            vals[t] = round(rng.uniform(-2, 2), 3)

    def cell(i, j):
        return vals[i * m + j]

    k = [[0.0] * m for _ in range(m)]
    if kind == "general":
        for i in range(m):
            for j in range(m):
                k[i][j] = cell(i, j)
    elif kind == "even":
        for i in range(m):
            for j in range(m):
                v = cell(i, j)
                w = cell(2 * c - i, 2 * c - j)
                k[i][j] = round(0.5 * (v + w), 3)
    elif kind == "odd":
        for i in range(m):
            for j in range(m):
                v = cell(i, j)
                w = cell(2 * c - i, 2 * c - j)
                k[i][j] = round(0.5 * (v - w), 3)
    elif kind == "psd":
        # 随机偶核 + 单次谱分析后加中心抬升 δ, 使符号 ≥ 0
        base = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                v = cell(i, j)
                w = cell(2 * c - i, 2 * c - j)
                base[i][j] = 0.5 * (v + w)
        sp = analyze_spectrum(base)
        if sp["re_min"] < -1e-9:
            base[c][c] += (-sp["re_min"]) * 1.05
        k = [[round(v, 3) for v in row] for row in base]
    else:
        raise ValueError("unknown kind: " + str(kind))
    # 强度缩放
    if abs(intensity - 1.0) > 1e-9:
        mx = max(abs(v) for row in k for v in row) or 1.0
        sc = intensity / mx
        k = [[round(v * sc, 4) for v in row] for row in k]
    return k


def parametric_kernel(family: str, m: int, params: dict):
    """参数化核族:
      gaussian      exp(-r²/2σ²)                      偶+正定(L3)
      exponential  exp(-r/λ) (Yukawa, (1-λ²Δ) 格林核) 偶+正定(L3/L4)
      box          圆盘/方窗 支撑 1, 0 外             偶
      heat         exp(-t|ξ|²) 热核符号 -> 空间核       偶+正定(L4)
      rational     1/(1+a|ξ|²)^p 符号 -> 空间核        偶+正定(L4, 微分算子格林核)
    """
    c = m // 2
    if family == "gaussian":
        sigma = float(params.get("sigma", 0.8))
        k = [[math.exp(-((i - c) ** 2 + (j - c) ** 2) / (2.0 * sigma * sigma))
              for j in range(m)] for i in range(m)]
        return _round_grid(k), _family_note(family, params)
    if family == "exponential":
        lam = float(params.get("lambda", 1.0))
        k = [[math.exp(-math.hypot(i - c, j - c) / lam) for j in range(m)] for i in range(m)]
        return _round_grid(k), _family_note(family, params)
    if family == "box":
        r = float(params.get("radius", 1.0))
        k = [[1.0 if math.hypot(i - c, j - c) <= r + 1e-9 else 0.0 for j in range(m)] for i in range(m)]
        return _round_grid(k), _family_note(family, params)
    if family in ("heat", "rational"):
        return _symbol_family_kernel(family, m, params), _family_note(family, params)
    raise ValueError("unknown family: " + str(family))


def _symbol_family_kernel(family, m, params):
    """符号族 -> 空间核:
    ω 在 [-π,π] 对称网格采样(中心在 N/2), IDFT 后用 (-1)^(a+b) 校正
    采样格点偏移, 得到连续逆变换在整数格点上的无相移采样, 再中心裁 m×m。
    带宽 [-π,π] 内的硬截断会在核上留下轻微振铃(参数取宽时趋于可忽略)。"""
    N = 256
    half = N // 2
    t = float(params.get("t", 0.6))
    a = float(params.get("a", 1.5))
    p = float(params.get("p", 1.0))
    H = [[0.0 + 0.0j for _ in range(N)] for _ in range(N)]
    for i in range(N):
        wi = (i - half) * (2.0 * math.pi) / N
        for j in range(N):
            wj = (j - half) * (2.0 * math.pi) / N
            r2 = wi * wi + wj * wj
            if family == "heat":
                H[i][j] = complex(math.exp(-t * r2))
            else:
                H[i][j] = complex(1.0 / (1.0 + a * r2) ** p)
    h = fft2(H, inverse=True)
    c = m // 2
    k = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            x = i - c
            y = j - c
            v = h[x % N][y % N].real
            if (x + y) & 1:
                v = -v
            k[i][j] = round(v, 6)
    return _round_grid(k)


def _round_grid(g, nd=6):
    return [[round(float(v), nd) for v in row] for row in g]


def _family_note(family, params):
    notes = {
        "gaussian": "高斯核 exp(-r²/2σ²): 偶核; 符号恒正 ⇒ 在窗口截断可忽略时属 L3 正定锥; "
                    "也是热算子 e^{tΔ} 的作用核(与 L4 格林核族相连)。正定性以右侧分析面板为准。",
        "exponential": "指数/Yukawa 核 exp(-r/λ): 偶 + 全空间正定; 是 (1-λ²Δ) 在 2D 的格林核, "
                       "对应 L4 微分算子格林核(高阶拉格朗日泛函 S=½∫(|u|²+λ²|∇u|²))。"
                       "窗口截断可能引入小负谱。",
        "box": "支撑核(窗口): 平移等变(L1); 偶对称但符号在角落取负 ⇒ 通常落在 L2 而非 L3。",
        "heat": "热核/高斯过程协方差 e^{-t|ω|²}(符号族 → 大格点 IDFT → 窗口采样): "
                "全空间正定且为热算子格林核(L3+L4); 有限窗口截断可能引入 ~小负谱, 以分析面板判定为准。",
        "rational": "有理符号 (1+a|ω|²)^{-p}(Helmholtz 型): 大格点 IDFT → 窗口采样; "
                    "全空间正定, 对应 (1-aΔ)^p 型微分算子的格林核(L3+L4); 窗口截断可引入小负谱。",
    }
    return notes[family]


# --------------------------------------------------------------------------
# 便利: 小工具
# --------------------------------------------------------------------------
def to_json_safe(obj):
    """把 numpy 无关的普通结构转 JSON 安全结构(此处即普通 list/dict)。"""
    return obj


# --------------------------------------------------------------------------
# 8. 轻量快速分析(用于 CNN 逐滤波器扫描, 网格 32, 不做探针/大谱图)
# --------------------------------------------------------------------------
def spectral_mesh_n(k, N: int):
    """谱网格, 显式网格尺寸(默认 128 的轻量版)。"""
    F = fft2(embed_kernel(k, N), inverse=False)
    half = N // 2
    D = [[0.0 + 0.0j for _ in range(N)] for _ in range(N)]
    for di in range(N):
        src_i = (di + half) % N
        for dj in range(N):
            src_j = (dj + half) % N
            D[di][dj] = F[src_i][src_j]
    return D


def quick_spectrum(k, N: int = 32):
    """小型网格符号统计。"""
    D = spectral_mesh_n(k, N)
    half = N // 2
    re_min = re_max = None
    im_max = 0.0
    for row in D:
        for v in row:
            r = v.real
            if re_min is None or r < re_min:
                re_min = r
            if re_max is None or r > re_max:
                re_max = r
            im = abs(v.imag)
            if im > im_max:
                im_max = im
    if re_min is None:
        re_min = re_max = 0.0
    return {
        "re_min": round(re_min, 5), "re_max": round(re_max, 5),
        "im_abs_max": round(im_max, 5),
        "dc": round(D[half][half].real, 5),
        "h_pi_pi": round(symbol_value(k, math.pi, math.pi).real, 5),
        "h_pi_0": round(symbol_value(k, math.pi, 0.0).real, 5),
        "h_0_pi": round(symbol_value(k, 0.0, math.pi).real, 5),
        "h_half_0": round(symbol_value(k, math.pi / 2, 0.0).real, 5),
        "h_0_half": round(symbol_value(k, 0.0, math.pi / 2).real, 5),
    }


def quick_filter_type(k, sp):
    """极简频带标签(供 CNN 扫描): 低通/高通/方向差分/混合。"""
    dc = sp["dc"]
    lo = (dc + sp["h_half_0"] + sp["h_0_half"]) / 3.0
    hi = (sp["h_pi_0"] + sp["h_0_pi"] + sp["h_pi_pi"]) / 3.0
    a = abs(sp["h_pi_0"] - sp["h_0_pi"])  # 方向不对称(轴间)
    if abs(dc) <= 1e-6 and abs(hi) > abs(lo) * 2:
        return "highpass"           # 差分/高通类
    if abs(hi) <= abs(lo) * 0.3 and abs(lo) > 0:
        return "lowpass"
    if a > max(abs(lo), abs(hi), 1e-9) * 0.5:
        return "directional"
    if dc > 0 and abs(dc - hi) <= dc * 0.5:
        return "lowpass"
    return "mixed"


def quick_analyze(k, N: int = 32):
    """逐滤波器快速分析(无探针/无大图), 输出 JSON 安全小结构。"""
    par = decompose_parity(k)
    sp = quick_spectrum(k, N)
    psd_tol = 1e-4 * max(abs(sp["re_min"]), abs(sp["re_max"]), 1e-6)
    psd = bool(par["even180"] and sp["re_min"] >= -psd_tol)
    nsd = bool(par["even180"] and sp["re_max"] <= psd_tol)
    if par["even180"]:
        definite = "psd" if psd else ("nsd" if nsd else "indefinite")
    else:
        definite = "complex"   # 复符号(非自伴)
    lf = laplacian_fit(k)
    fk = quick_filter_type(k, sp)
    if par["type"] == "odd":
        fk = "directional"   # 奇核 = 一阶反自伴 → 必为方向差分类(符号纯虚相位)
    return {
        "type": par["type"],
        "even180": bool(par["even180"]),
        "d4": bool(par["d4"]),
        "even_frac": par["even_frac"],
        "odd_frac": par["odd_frac"],
        "definite": definite,
        "psd": psd, "nsd": nsd,
        "spectrum": sp,
        "filter_kind": fk,
        "lap": {"a": lf["a"], "b": lf["b"], "rel": lf["rel_residual"], "exact": lf["exact"]},
        "orient_deg": _kernel_orientation(k, par),
        "template": _template_match(k),
    }


def complex_quick(k, N: int = 32):
    """虚数场(复化)视角的快速分析。

    对任意实核 k, 卷积算子 A 在复 L² 场上做自伴/反自伴分解:
        A = A_s + A_a,   A_s 自伴(偶核 k_s),  A_a 反自伴(奇核 k_a)。
    乘 i 后 iA_a 为自伴(Hermitian): 复化 Hermitian 算子(核 c = k_s − i k_a)的符号
        ĥ_c(ω) = Re ĥ(ω) + Im ĥ(ω) ∈ ℝ   (另一符号取法为 Re − Im)
    恒为实值 ⇒ 任何实核在复场上都对应实值 Hermitian 二次拉氏量 S=½∫ĥ_c|û|²,
    奇部不再是"陀螺/不可导出", 而成为谱 ±|…| 的『动能/电流』项(类似 i∂→动量)。
    返回: 复化实谱 min/max、Hermitian 正定性、动能(imag)占比、Re/Im 功率。
    """
    D = spectral_mesh_n(k, N)
    re2 = im2 = 0.0
    hmin = hmax = None
    a_min = a_max = None
    for row in D:
        for v in row:
            re = v.real
            im = v.imag
            re2 += re * re
            im2 += im * im
            hc = re + im          # ĉ = k_s − i k_a 的实符号
            ac = re - im          # ĉ = k_s + i k_a 的实符号
            if hmin is None or hc < hmin:
                hmin = hc
            if hmax is None or hc > hmax:
                hmax = hc
            if a_min is None or ac < a_min:
                a_min = ac
            if a_max is None or ac > a_max:
                a_max = ac
    if hmin is None:
        hmin = hmax = a_min = a_max = 0.0
    tot = re2 + im2
    scale = max(abs(hmin), abs(hmax), 1e-6)
    herm_psd = hmin >= -1e-4 * scale
    scale2 = max(abs(a_min), abs(a_max), 1e-6)
    herm_psd_alt = a_min >= -1e-4 * scale2
    return {
        "kinetic_share": round(im2 / tot, 5) if tot > 0 else 0.0,   # 奇部→动能功率占比
        "hc_min": round(hmin, 5), "hc_max": round(hmax, 5),          # 复化实谱范围(Re+Im)
        "alt_min": round(a_min, 5), "alt_max": round(a_max, 5),      # 另一符号取法(Re−Im)
        "herm_psd": bool(herm_psd), "herm_psd_alt": bool(herm_psd_alt),
    }


def _kernel_orientation(k, par):
    """方向性: 对(奇/一般)核的结构张量主轴角(0~180°); 偶核返回 None。"""
    if par["odd_frac"] < 0.05:
        return None
    M = len(k)
    c = M // 2
    # 用奇部 ka 的有限差分估计梯度场方向
    gx = [[0.0] * M for _ in range(M)]
    gy = [[0.0] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            ka = 0.5 * (k[i][j] - k[2 * c - i][2 * c - j])
            il = k[i][j - 1] if j > 0 else k[i][j]
            ir = k[i][j + 1] if j < M - 1 else k[i][j]
            iu = k[i - 1][j] if i > 0 else k[i][j]
            idw = k[i + 1][j] if i < M - 1 else k[i][j]
            # 对奇部分响应做梯度(差分算子作用于 ka 视为场)
            gx[i][j] = 0.5 * (ir - il)
            gy[i][j] = 0.5 * (idw - iu)
            _ = ka
    sxx = syy = sxy = 0.0
    for i in range(M):
        for j in range(M):
            w = abs(k[i][j])
            sxx += gx[i][j] ** 2 * w
            syy += gy[i][j] ** 2 * w
            sxy += gx[i][j] * gy[i][j] * w
    tr = sxx + syy
    if tr <= 1e-12:
        return None
    theta = 0.5 * math.degrees(math.atan2(2 * sxy, sxx - syy))
    theta = theta % 180.0
    # 一致性
    strength = math.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (tr + 1e-12)
    return {"deg": round(theta, 1), "strength": round(strength, 4)}


def _template_match(k):
    """与经典算子的归一化相关系数(去均值), 返回 top 候选。"""
    M = len(k)
    templates = {
        "sobel_x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        "sobel_y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        "prewitt_x": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        "prewitt_y": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        "deriv_x": [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        "deriv_y": [[0, -1, 0], [0, 0, 0], [0, 1, 0]],
        "lap4": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        "lap8": [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
        "cross": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        "box3": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "gauss5": [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
    }

    def center(t):
        tm = len(t)
        if tm > M:
            return None
        out = [[0.0] * M for _ in range(M)]
        off = (M - tm) // 2
        for i in range(tm):
            for j in range(tm):
                out[off + i][off + j] = t[i][j]
        return out

    kf = [float(v) for row in k for v in row]
    kmean = sum(kf) / len(kf)
    kv = [v - kmean for v in kf]
    kn = math.sqrt(sum(v * v for v in kv)) or 1e-12
    scores = {}
    for name, t in templates.items():
        tc = center(t)
        if tc is None:
            continue
        tf = [float(v) for row in tc for v in row]
        tm = sum(tf) / len(tf)
        tv = [v - tm for v in tf]
        tn = math.sqrt(sum(v * v for v in tv)) or 1e-12
        s = sum(a * b for a, b in zip(kv, tv)) / (kn * tn)
        scores[name] = round(s, 4)
    best = max(scores, key=scores.get)
    return {"best": best, "score": scores[best], "top3": sorted(
        scores.items(), key=lambda kv: -kv[1])[:3]}
