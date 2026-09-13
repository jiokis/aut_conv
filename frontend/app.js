/* ================================================================
 * 算子生成与分析工作台 — 前端应用
 * 渲染: 核编辑器 / 生成器 / 分析报告(频率响应·谱分解·能量语义·探针)
 * ================================================================ */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------------- 工具 ---------------- */
function fmt(x, digits) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a === 0) return "0";
  if (digits) return x.toFixed(digits);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(2);
  if (a >= 100) return x.toFixed(1);
  if (a >= 1) return (+x.toFixed(3)).toString();
  return (+x.toFixed(4)).toString();
}
function fmtv(x) { return (x > 0 ? "+" : "") + fmt(x); }
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

function toast(msg, isErr) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show" + (isErr ? " err" : "");
  clearTimeout(t._tm);
  t._tm = setTimeout(() => (t.className = "toast"), 3200);
}

function debounce(fn, ms) {
  let tm = null;
  return (...a) => { clearTimeout(tm); tm = setTimeout(() => fn(...a), ms); };
}

/* ---------------- 配色 ---------------- */
function lerp(a, b, t) { return a + (b - a) * t; }
const STOPS_DIV = [
  [-1, [62, 88, 255]],      // 负: 蓝
  [-0.25, [90, 150, 255]],
  [0, [9, 14, 30]],
  [0.25, [255, 180, 96]],
  [1, [255, 108, 60]],      // 正: 橙红
];
const STOPS_SEQ = [
  [0, [12, 10, 42]],
  [0.3, [72, 40, 150]],
  [0.55, [50, 130, 190]],
  [0.78, [80, 210, 170]],
  [1, [255, 220, 120]],
];
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
      const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
      return [lerp(c0[0], c1[0], f), lerp(c0[1], c1[1], f), lerp(c0[2], c1[2], f)];
    }
  }
  return stops[stops.length - 1][1];
}

/* 画 2D 热图 */
function drawHeat(canvas, data, opts = {}) {
  const { mode = "div", symmetric = null, range = null } = opts;
  const R = data.length, C = data[0].length;
  canvas.width = C; canvas.height = R;
  canvas.style.imageRendering = "pixelated";
  const ctx = canvas.getContext("2d");
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const v = data[i][j];
    if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const sym = symmetric === null ? mode === "div" : symmetric;
  const span = sym ? Math.max(Math.abs(mn), Math.abs(mx), 1e-12)
                   : (mx - mn || 1e-12);
  const img = ctx.createImageData(C, R);
  for (let i = 0; i < R; i++) {
    for (let j = 0; j < C; j++) {
      const v = data[i][j];
      let t = sym ? v / span : (v - mn) / (mx - mn);
      const c = mode === "div" ? ramp(STOPS_DIV, t) : ramp(STOPS_SEQ, t);
      const p = (i * C + j) * 4;
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2];
      img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return { mn, mx, span };
}

/* 折线图 */
function drawChart(canvas, series, opts = {}) {
  const W = 640, H = 260;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const { xmin = -Math.PI, xmax = Math.PI, title } = opts;
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const v of s.y) {
    if (v < lo) lo = v; if (v > hi) hi = v;
  }
  if (lo === hi) { lo -= 1; hi += 1; }
  const padL = 34, padR = 12, padT = 14, padB = 26;
  const pw = W - padL - padR, ph = H - padT - padB;
  const X = (x) => padL + ((x - xmin) / (xmax - xmin)) * pw;
  const Y = (y) => padT + (1 - (y - lo) / (hi - lo)) * ph;
  ctx.strokeStyle = "rgba(120,150,210,.14)";
  ctx.fillStyle = "#7f92b5";
  ctx.font = "10px Consolas, monospace";
  ctx.lineWidth = 1;
  // 网格
  for (let g = 0; g <= 4; g++) {
    const gy = padT + (ph * g) / 4, v = lo + ((hi - lo) * (4 - g)) / 4;
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(W - padR, gy); ctx.stroke();
    ctx.fillText(fmt(v, 2), 2, gy + 3);
  }
  for (const lab of [-Math.PI, -Math.PI / 2, 0, Math.PI / 2, Math.PI]) {
    ctx.beginPath(); ctx.moveTo(X(lab), padT); ctx.lineTo(X(lab), H - padB); ctx.stroke();
    ctx.fillText(lab === Math.PI ? "π" : lab === -Math.PI ? "−π"
      : lab === 0 ? "0" : lab > 0 ? "π/2" : "−π/2", X(lab) - 8, H - 10);
  }
  // 0 基线
  if (lo < 0 && hi > 0) {
    ctx.strokeStyle = "rgba(255,255,255,.35)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.x.length; i++) {
      if (!Number.isFinite(s.y[i])) continue;
      const px = X(s.x[i]), py = Y(s.y[i]);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  if (title) { ctx.fillStyle = "#d9e4ff"; ctx.font = "700 11px inherit"; ctx.fillText(title, padL, padT - 4); }
}

/* ---------------- 预置库 ---------------- */
const PRESETS = [
  { id: "cross", name: "十字核(对话主角)", lv: "L2", prov: "cross",
    desc: "偶核·非正定·可导出拉氏量 · K = Δ₄ + 4δ · 棋盘模式能量为负",
    k: [[0, 1, 0], [1, 0, 1], [0, 1, 0]] },
  { id: "lap4", name: "离散拉普拉斯 Δ₄", lv: "L2", prov: "lap",
    desc: "中心 −4: 负定自伴(NSD) ⇒ −K 正定, 即耗散/正则项模板",
    k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] },
  { id: "lap8", name: "8 邻域拉普拉斯", lv: "L2", prov: "lap",
    desc: "含对角邻域, NSD; 符号更各向同性(半径 8 邻接)",
    k: [[1, 1, 1], [1, -8, 1], [1, 1, 1]] },
  { id: "gauss3", name: "二项式高斯 3×3", lv: "L3", prov: "gaussian",
    desc: "正定(PSD) ⇒ 凸能量 / 协方差 / 高斯过程核",
    k: [[1, 2, 1], [2, 4, 2], [1, 2, 1]] },
  { id: "gauss5", name: "二项式高斯 5×5", lv: "L3", prov: "gaussian",
    desc: "更平滑的 PSD 低通核(尺度 2)",
    k: [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]] },
  { id: "delta", name: "中心冲激 δ", lv: "L3", prov: "",
    desc: "恒等算子: 符号 ≡1(PSD), 平凡拉氏量 S=½‖u‖²",
    k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
  { id: "box3", name: "3×3 全 1 窗口", lv: "L2", prov: "",
    desc: "偶核但符号取负(角落谱 <0) ⇒ 非正定, 作对比",
    k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]] },
  { id: "sharpen", name: "锐化 5δ−Δ₄", lv: "L2", prov: "lap",
    desc: "自伴·不定; 高频增强(unsharp), K = −Δ₄ + 5δ",
    k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  { id: "sobelx", name: "Sobel X(梯度)", lv: "L1", prov: "sobel",
    desc: "180° 奇核(反自伴): 陀螺/方向语义, 不能由拉氏量导出",
    k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
  { id: "sobely", name: "Sobel Y(梯度)", lv: "L1", prov: "sobel",
    desc: "同上, 沿 y 方向",
    k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
  { id: "derivx", name: "一阶差分 ∂x", lv: "L1", prov: "sobel",
    desc: "纯 x 奇部: ⟨u,Ku⟩=0 ⇒ 能量守恒的相位旋转(陀螺)",
    k: [[0, 0, 0], [-1, 0, 1], [0, 0, 0]] },
  { id: "prewitt", name: "Prewitt X", lv: "L1", prov: "sobel",
    desc: "差分 + 邻域平均混合的一般核",
    k: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
];

/* 参数化核族参数定义 */
const FAMILIES = {
  gaussian: { params: [{ k: "sigma", label: "σ (宽度)", min: 0.3, max: 5, step: 0.1, def: 1.2 }],
    note: "高斯核 exp(−r²/2σ²): 偶核, 符号恒正 ⇒ L3 正定(窗口截断忽略时); 也是热算子 e^{tΔ} 作用核。" },
  exponential: { params: [{ k: "lambda", label: "λ (衰减长)", min: 0.3, max: 4, step: 0.1, def: 1.5 }],
    note: "指数/Yukawa 核 exp(−r/λ): 偶 + 正定; 是 (1−λ²Δ) 在 2D 的格林核 ⇒ L4 微分算子格林核(拉氏量 S=½∫(|u|²+λ²|∇u|²))。" },
  box: { params: [{ k: "radius", label: "半径 R", min: 0.5, max: 7, step: 0.5, def: 2 }],
    note: "支撑窗核: 平移等变(L1), 偶对称; 符号通常取负(角落) ⇒ 落在 L2 而非 L3。" },
  heat: { params: [{ k: "t", label: "t (扩散时间)", min: 0.1, max: 4, step: 0.05, def: 0.6 }],
    note: "热核(符号 e^{−t|ω|²}): 大格点 IDFT → 窗口采样; 全空间正定+热算子格林核(L3/L4)。窗口截断或引入小负谱, 以分析面板判定为准。" },
  rational: { params: [{ k: "a", label: "a (算子尺度)", min: 0.2, max: 6, step: 0.1, def: 1.5 },
    { k: "p", label: "p (幂次)", min: 1, max: 4, step: 1, def: 1 }],
    note: "有理符号 (1+a|ω|²)^{−p}(Helmholtz 型): 大格点 IDFT → 窗口采样; 对应 (1−aΔ)^p 格林核 ⇒ L3/L4。" },
};

/* ---------------- 应用状态 ---------------- */
const state = {
  kernel: null,       // 2D number[]
  paint: 0,
  provenance: "",     // 来源提示
  note: "",           // 生成器说明
  result: null,
  busy: false,
  queued: false,
  auto: true,
  currentProbe: "constant",
  lastM: 3,
};

/* ---------------- API ---------------- */
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "API 错误");
  return j;
}

/* ================================================================
 * 核网格编辑器
 * ================================================================ */
function rebuildGrid() {
  const m = state.kernel.length;
  const tbl = $("kgrid");
  tbl.innerHTML = "";
  const c = (m - 1) / 2;
  for (let i = 0; i < m; i++) {
    const tr = document.createElement("tr");
    for (let j = 0; j < m; j++) {
      const td = document.createElement("td");
      if (i === c && j === c) td.className = "cell-ctr";
      const inp = document.createElement("input");
      inp.type = "text"; inp.inputMode = "decimal";
      inp.value = fmt(state.kernel[i][j], 3);
      inp.dataset.i = i; inp.dataset.j = j;
      inp.addEventListener("focus", (e) => e.target.select());
      inp.addEventListener("input", () => {
        const v = parseFloat(inp.value);
        state.kernel[i][j] = Number.isFinite(v) ? v : 0;
        paintCellClass(inp, state.kernel[i][j]);
        scheduleAnalyze();
      });
      inp.addEventListener("blur", () => { inp.value = fmt(state.kernel[i][j], 3); });
      td.appendChild(inp); tr.appendChild(td);
    }
    tbl.appendChild(tr);
  }
}
function paintCellClass(inp, v) {
  inp.classList.remove("n-pos", "n-neg");
  if (v > 1e-9) inp.classList.add("n-pos");
  else if (v < -1e-9) inp.classList.add("n-neg");
}
function paintAll() {
  const m = state.kernel.length;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const inp = document.querySelector(`#kgrid input[data-i="${i}"][data-j="${j}"]`);
      if (inp) { inp.value = fmt(state.kernel[i][j], 3); paintCellClass(inp, state.kernel[i][j]); }
    }
  }
}
function currentSize() { return state.kernel ? state.kernel.length : 3; }

function scheduleAnalyze() {
  if (state.auto) debouncedAnalyze();
  renderMini();
  $("curSizeTag").textContent = `${currentSize()}×${currentSize()}`;
}
const debouncedAnalyze = debounce(() => analyze(), 420);

/* ================================================================
 * 分析 & 渲染
 * ================================================================ */
async function analyze() {
  if (!state.kernel || state.busy) { state.queued = true; return; }
  state.busy = true;
  const dot = $("busyDot");
  dot.classList.add("on");
  try {
    const t0 = performance.now();
    const j = await api("/api/analyze", { kernel: state.kernel });
    const ms = Math.round(performance.now() - t0);
    state.result = j.result;
    $("engineMeta").textContent = `分析 ${ms} ms · Python 纯算引擎`;
    renderReport();
  } catch (e) {
    toast("分析失败: " + e.message, true);
  } finally {
    state.busy = false;
    dot.classList.remove("on");
    if (state.queued) { state.queued = false; analyze(); }
  }
}

/* -------- 报告渲染 -------- */
function renderReport() {
  const r = state.result;
  if (!r) return;
  const body = $("reportBody");
  const parts = [];
  parts.push(verdictHTML(r));
  parts.push(metricsHTML(r));
  parts.push('<div class="grid-2">');
  parts.push(spectrumCard(r));
  parts.push(parityCard(r));
  parts.push(energyCard(r));
  parts.push(probeCard(r));
  parts.push(designSpaceCard(r));
  parts.push(summaryCard(r));
  parts.push("</div>");
  parts.push(conclCard(r));
  body.innerHTML = parts.join("");
  // 绘制 canvas
  drawHeat($("hmSpectrum"), r.spectrum.mesh_re, { mode: "div" });
  const meshMag = $("hmMag");
  if (meshMag) drawHeat(meshMag, r.spectrum.mesh_mag, { mode: "seq", symmetric: false });
  const hmHp = $("hmHp"), hmHn = $("hmHn");
  const re = r.spectrum.mesh_re;
  if (hmHp && hmHn) {
    const hp = re.map(row => row.map(v => Math.max(0, v)));
    const hn = re.map(row => row.map(v => Math.max(0, -v)));
    drawHeat(hmHp, hp, { mode: "seq", symmetric: false });
    drawHeat(hmHn, hn, { mode: "seq", symmetric: false });
  }
  // 对称性分解小图
  const smalls = [
    ["hmK", r.parity.zero ? r.parity.ks : state.kernel],
    ["hmKs", r.parity.ks], ["hmKa", r.parity.ka],
  ];
  if (r.self_adjoint) {
    smalls.push(["hmPos", r.spec_split.k_pos_approx]);
    smalls.push(["hmNeg", r.spec_split.k_neg_approx]);
    smalls.push(["hmD", r.spec_split.k_diss_approx]);
  }
  for (const [id, dat] of smalls) { const el = $(id); if (el && dat) drawHeat(el, dat, { mode: "div" }); }
  // 切片曲线
  const om = r.spectrum.omega_axis;
  drawChart($("chSlices"), [
    { x: om, y: r.spectrum.slice_x, color: "#ffb64d", label: "Re Ĥ(ω₁, 0)" },
    { x: om, y: r.spectrum.slice_diag, color: "#3fd7ff", label: "Re Ĥ(t, t)" },
  ], {});
  // 探针切换显示
  renderProbePanel();
  renderMini();
}

/* ---------- 各卡片 HTML ---------- */
function verdictHTML(r) {
  const par = r.parity;
  const L = (n, on, txt, extra) =>
    `<span class="lv-chip L${n} ${on ? "on" : ""}" title="${extra || txt}">L${n} <span class="lv-txt">${txt}</span></span>`;
  const lv4 = isL4();
  const chips =
    L(1, true, "平移等变", "卷积结构天然满足: 任何核 k 都给平移等变算子, 不限制形状") +
    L(2, r.self_adjoint, "自伴·可导拉氏量", "偶核 k(x)=k(−x) ⇔ 自伴 ⇔ δS/δu=k*u") +
    L(3, r.psd, "正定·凸能量", "ĥ(ω)≥0: RKHS/协方差/格林核层") +
    L(4, lv4, "格林/微分核", lv4 ? "来源为微分算子格林核族或 K 为 Δ 的组合" : "微分算子格林核族") +
    L(5, r.parity.d4, "群等变(旋转)", "90° 旋转/反射对称 D4");
  const type = par.type === "even" ? "偶核" : par.type === "odd" ? "奇核" : "一般核";
  const defTxt = r.psd ? "正定(凸能量)" : r.nsd ? "负定(取负号后为耗散 D)" : r.indefinite ? "不定(有正负谱)" : "复符号(非自伴)";
  let hl;
  if (r.psd) hl = `<b class="ok">正定</b>: 二次拉氏量 S[u]=½∫u(K*u) 凸, 谱非负, 落 L3 正定锥`;
  else if (r.self_adjoint) hl = `<b class="warn">自伴但非正定</b>: 可导出拉氏量但能量不凸(谱 ${fmt(r.spectrum.re_min)} … ${fmt(r.spectrum.re_max)}), 正谱=势能、负谱=耗散 D`;
  else if (par.type === "odd") hl = `<b class="bad">奇核/反自伴</b>: 陀螺项(能量守恒), 不能由实拉氏量导出`;
  else hl = `<b class="info">一般核</b>: 偶部(自伴)可导拉氏量 + 奇部(陀螺)反自伴, 线性组合拉氏量最多覆盖偶部`;
  return `<div class="verdict">
    <div class="lv-chips">${chips}</div>
    <div class="headline" style="margin-top:12px">当前核: <b class="info">${type} ${par.m}×${par.m}</b> — ${hl}</div>
    <div class="small-txt">奇偶类型: <b>${type}</b> · 正定性: <b>${defTxt}</b> · 奇部(陀螺)能量占比 <b>${(par.odd_frac * 100).toFixed(1)}%</b>${state.provenance ? " · 来源: " + esc(state.provenance) : ""}</div>
  </div>`;
}
function isL4() {
  const r = state.result;
  const fam = state.provenance;
  const famSet = new Set(["gaussian", "exponential", "heat", "rational", "cross", "lap"]);
  if (famSet.has(fam)) return true;
  if (r && r.laplacian && r.laplacian.exact) return true;
  return false;
}

function metricsHTML(r) {
  const sp = r.spectrum;
  const m = (k, v, cls, title) =>
    `<div class="metric" title="${esc(title || "")}"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`;
  const clsDef = r.psd ? "ok" : r.nsd ? "info" : r.indefinite ? "warn" : "bad";
  return `<div class="metrics">
    ${m("DC 增益 Σk", fmt(r.meta.dc), "info", "常数场的响应 H(0,0)")}
    ${m("L² 范数", fmt(r.meta.l2), "", "核能量的平方根")}
    ${m("谱范围 Re Ĥ", fmt(sp.re_min) + " … " + fmt(sp.re_max), clsDef, "最小值/最大值在 " + sp.re_min_at + " / " + sp.re_max_at)}
    ${m("偶部占比", (r.parity.even_frac * 100).toFixed(0) + "%", r.self_adjoint ? "ok" : "info", "Σk_s²/Σk²")}
    ${m("正定性", r.psd ? "PSD" : r.nsd ? "NSD" : r.indefinite ? "不定" : "复符号", clsDef, "判定容差 1e-4·max|H|")}
    ${m("拉格朗日可导出", r.lagrangian_derivable ? "✓ 是" : "✗ 否", r.lagrangian_derivable ? "ok" : "bad", "⇔ 偶核(自伴)")}
    ${m("滤波器型", filterKindCN(r.filter.kind), "info", "低频/高频均值: " + fmt(r.filter.mean_low) + " / " + fmt(r.filter.mean_high))}
    ${m("各向异性", fmt(r.filter.anisotropy), r.filter.anisotropy > 0.2 ? "warn" : "", "奈奎斯特轴 vs 对角 |H| 差异")}
  </div>`;
}
function filterKindCN(k) {
  return { lowpass: "低通/平滑", highpass: "高通/差分", bandpass: "带通", flat: "平坦/恒等", other: "混合" }[k] || k;
}

/* 频率响应卡 */
function spectrumCard(r) {
  const sp = r.spectrum;
  const selfadj = r.self_adjoint;
  const div = (selfadj && r.indefinite) || r.nsd;
  const magExtra = selfadj ? "" :
    `<div class="small-txt" style="margin-top:4px">非自伴核: 符号复值 — 实部(下方)是 ⟨u,Ku⟩ 的能量部分, 虚部 |Im|max=${fmt(sp.im_abs_max)} 来自反自伴(陀螺)部分。</div>`;
  let splitFigs = "";
  if (selfadj) {
    splitFigs = `<div class="mini-cmp">
      <div><h3>谱分解 · 势能核符号 H⁺</h3><canvas class="hm" id="hmHp"></canvas>
        <div class="cap">H⁺=max(Re Ĥ,0) · 谱和(128² 采样)≈${fmt(r.spec_split.pot_energy_sum)}</div></div>
      <div><h3>谱分解 · 耗散核符号 −H⁻(=D)</h3><canvas class="hm" id="hmHn"></canvas>
        <div class="cap">D 符号=−min(Re Ĥ,0)≥0 · 谱和≈${fmt(r.spec_split.diss_energy_sum)}<br>Rayleigh 函数 R[v]=½∫v(D*v)</div></div>
    </div>`;
  }
  return `<div class="card fig">
    <h3>频率响应(符号) Re Ĥ(ω₁,ω₂)<span class="note">DC 在中心 · 范围 [−π,π]²</span></h3>
    <canvas class="hm" id="hmSpectrum" style="height:230px"></canvas>
    <div class="cap">Re Ĥ ∈ [${fmt(sp.re_min)}, ${fmt(sp.re_max)}] · 极值位置(ω₁,ω₂) = (${sp.re_min_at}) 最小, (${sp.re_max_at}) 最大</div>
    ${magExtra}
    <div style="margin-top:6px"><h3>幅值 |Ĥ(ω)|${selfadj ? "" : " · 非自伴: 存在虚部(陀螺/旋转成分), 实部才是能量部分"}</h3>
    <canvas class="hm" id="hmMag" style="height:120px"></canvas>
    <div class="cap">|Ĥ| max = ${fmt(sp.mag_max)}</div></div>
    <h3 style="margin-top:12px">一维切片: 沿 ω₁ 轴(橙) 与 对角线 ω₁=ω₂(青)</h3>
    <canvas class="chart" id="chSlices" style="height:200px"></canvas>
    <div class="cap">实部切片; 灰虚线为 0。曲线穿过零点 ⇒ 符号变号 ⇒ 非正定。</div>
    ${splitFigs}
  </div>`;
}

/* 对称性/分解卡 */
function parityCard(r) {
  const par = r.parity;
  const div = (id, label) => `<div><h3>${label}</h3><canvas class="hm" id="${id}" style="max-width:180px;margin:0 auto"></canvas></div>`;
  const symDesc = [];
  if (par.even180) symDesc.push("<span class='badge cyan'>中心对称 180°</span>");
  if (par.odd180) symDesc.push("<span class='badge violet'>反中心对称(奇)</span>");
  if (par.d4) symDesc.push("<span class='badge green'>90° 旋转 D4</span>");
  if (par.ref_v) symDesc.push("<span class='badge gray'>左右镜像</span>");
  if (par.ref_h) symDesc.push("<span class='badge gray'>上下镜像</span>");
  if (par.ref_d) symDesc.push("<span class='badge gray'>主对角</span>");
  if (par.ref_ad) symDesc.push("<span class='badge gray'>副对角</span>");
  const eq = `<div class="formula">k = k<sub>s</sub> + k<sub>a</sub>,　k<sub>s</sub>(x)=½[k(x)+k(−x)] 偶/自伴,　k<sub>a</sub>(x)=½[k(x)−k(−x)] 奇/反自伴</div>`;
  return `<div class="card fig">
    <h3>对称性与奇偶分解<span class="note">自伴 ⇔ 偶核</span></h3>
    <div class="legend-line" style="margin-bottom:8px">${symDesc.join("") || "无显著对称性(L1 一般核)"}</div>
    <div class="mini-cmp">
      ${div("hmK", "核 K")}
      ${div("hmKs", "偶部 k<sub>s</sub> (自伴/势能+耗散源)")}
      ${div("hmKa", "奇部 k<sub>a</sub> (反自伴/陀螺)")}
    </div>
    ${eq}
    <div class="stat-row"><span class="st">偶部占比 <b class="pos">${(par.even_frac * 100).toFixed(1)}%</b></span>
    <span class="st">奇部占比 <b class="neg">${(par.odd_frac * 100).toFixed(1)}%</b></span>
    <span class="st">奇偶类型 <b>${par.type === "even" ? "even(偶)" : par.type === "odd" ? "odd(奇)" : "general(偶+奇)"}</b></span></div>
    <div class="small-txt">180° 旋转: 奇偶类型决定算子能否成为二次拉格朗日泛函的变分导数。偶部可, 奇部对应 ⟨u,A<sub>a</sub>u⟩=0 的能量守恒旋转。</div>
  </div>`;
}

/* 能量语义卡 */
function energyCard(r) {
  const par = r.parity;
  let rows = "";
  if (r.self_adjoint) {
    rows += `<div class="formula">S[u] = ½ ∫ u (K*u) dx　(= ½∫ ĥ(ω)|û(ω)|² dω)　→　δS/δu = K*u</div>`;
    rows += `<div class="small-txt">${r.psd ? "谱 ĥ≥0 ⇒ S 凸(良定), 无负能量模式 — 核方法/协方差/正则项素材。"
      : r.nsd ? "谱 ĥ≤0 ⇒ S 凹; −K 正定: 作为 −∂u/∂t 一侧是过阻尼耗散(Rayleigh), 用于稳定/耗散层。"
      : "谱变号 ⇒ S 非凸; 负谱区间对应负能量模式(棋盘等高频模式)。谱分解: S = S<sub>+</sub> + S<sub>−</sub>, 负部经 Rayleigh 函数纳入变分框架。"}</div>`;
  } else {
    rows += `<div class="small-txt">${par.type === "odd" ? "奇核: 反自伴 ⇒ 陀螺项。"
      : "一般核: 由偶部与奇部叠加。"} 在实数框架下, 偶部给出势能/耗散, 奇部为陀螺(不耗能)。</div>`;
  }
  const A = `<div class="formula">一般实卷积算子　A = A<sub>pot</sub> + A<sub>gyro</sub> − D
  A<sub>pot</sub>: 正谱自伴(势能, 来自拉氏量)　A<sub>gyro</sub>: 奇部反自伴(能量守恒)
  D: 正定偶核(耗散, Rayleigh 函数 R[∂u]=½∫∂u(D*∂u))</div>`;
  let flowTxt = "";
  if (r.probes && r.probes.length) {
    const rowsP = r.probes.map(p => {
      const w = p.work;
      const cls = w > 1e-9 ? "pos" : w < -1e-9 ? "neg" : "";
      const sym = w > 1e-9 ? "耗散(衰减)" : w < -1e-9 ? "增长(不稳定)" : "守恒/旋转";
      return `<tr><td>${esc(p.name)}</td><td class="mono">${fmt(p.S)}</td><td class="mono ${cls}">${w > 0 ? "+" : ""}${fmt(w)}</td><td>${sym}</td></tr>`;
    }).join("");
    flowTxt = `<div style="margin-top:8px"><b>梯度流 ∂u/∂t = −K u 下的能量速率</b> dE/dt = −⟨u,Ku⟩ (E=½‖u‖²), 探针逐个给出:</div>
    <table class="sumt" style="margin-top:6px"><tr><th>模式 u</th><th>S[u]=½⟨u,Ku⟩</th><th>⟨u,Ku⟩</th><th>dE/dt 语义</th></tr>${rowsP}</table>
    <div class="small-txt" style="margin-top:5px">正 ⟨u,Ku⟩(绿色) ⇒ 正定方向(自伴正谱)使 L² 能量衰减; 负值(红色) ⇒ 负能量模式(自伴负谱/不定核), 非凸来源; 近乎 0 ⇒ 反自伴旋转。</div>`;
  }
  return `<div class="card fig">
    <h3>拉格朗日泛函 & 能量语义<span class="note">自伴⇔可导出 · 谱三分为势能/陀螺/耗散</span></h3>
    ${rows}${A}
    <div class="legend-line">${r.lagrangian_derivable
      ? "<span><span class='dot' style='background:var(--green)'></span>可由二次拉氏量导出(自伴)</span>"
      : "<span><span class='dot' style='background:var(--red)'></span>不能由实拉氏量直接导出(需扩展变分)</span>"}
    </div>
    ${flowTxt}
  </div>`;
}

/* 探针卡 */
function probeCard(r) {
  const ids = ["constant", "ramp_x", "sine_low", "checker", "impulse"];
  const chips = ids.map(id => {
    const p = r.probes.find(q => q.id === id);
    const sel = state.currentProbe === id ? "sel" : "";
    return `<button class="btn chip ${sel}" data-probe="${id}">${esc(p ? p.name.split(" ")[0] : id)}</button>`;
  }).join("");
  return `<div class="card fig">
    <h3>滤波行为探针(卷积响应)<span class="note">点击切换模式</span></h3>
    <div class="probe-bar">${chips}</div>
    <div id="probeDetail" style="margin-top:10px"></div>
  </div>`;
}
function renderProbePanel() {
  const r = state.result;
  const wrap = $("probeDetail");
  if (!r || !wrap) return;
  const p = r.probes.find(q => q.id === state.currentProbe) || r.probes[0];
  state.currentProbe = p.id;
  // 更新 chip
  document.querySelectorAll("[data-probe]").forEach(b => b.classList.toggle("sel", b.dataset.probe === p.id));
  const wcls = p.work > 1e-9 ? "pos" : p.work < -1e-9 ? "neg" : "";
  const modeDesc = p.work > 1e-9 ? "自伴正谱方向 → 梯度流能量衰减(耗散性)" :
    p.work < -1e-9 ? "负能量模式 → 梯度流能量增长(负谱 / 非凸证据)" : "反自伴(陀螺)或零谱 → 能量守恒";
  wrap.innerHTML = `<div class="stat-row">
    <span class="st">⟨u,Ku⟩ <b class="${wcls}">${p.work > 0 ? "+" : ""}${fmt(p.work)}</b></span>
    <span class="st">S[u]=½⟨u,Ku⟩ <b>${fmt(p.S)}</b></span>
    <span class="st">输出范围 <b>${fmt(p.out_min)} … ${fmt(p.out_max)}</b></span>
    <span class="st">能量语义 <b class="${wcls}">${modeDesc}</b></span></div>
    <div class="mini-cmp">
      <div><canvas class="hm" id="hmPin"></canvas><div class="cap">输入模式 u: ${esc(p.name)}</div></div>
      <div><canvas class="hm" id="hmPout"></canvas><div class="cap">K*u 响应</div></div>
    </div>`;
  const a = $("hmPin"), b = $("hmPout");
  if (a) drawHeat(a, p.input, { mode: "div" });
  if (b) drawHeat(b, p.output, { mode: "div" });
  document.querySelectorAll("[data-probe]").forEach(btn =>
    btn.onclick = () => { state.currentProbe = btn.dataset.probe; renderProbePanel(); });
}

/* 设计空间卡 */
function designSpaceCard(r) {
  const par = r.parity;
  const provL4 = isL4();
  const levels = [
    ["L1", "所有卷积核", "平移等变(卷积结构)", true, "violet"],
    ["L2", "自伴偶核", "可导出二次拉氏量", r.self_adjoint, "blue"],
    ["L3", "正定锥 K_pos", "凸能量 · RKHS/协方差", r.psd, "green"],
    ["L4", "微分算子格林核", "有理/指数符号 · 高阶拉氏量", provL4, "orange"],
    ["L5", "群等变核", "90° 旋转 D4 等", par.d4, "pink"],
  ];
  const items = levels.map(([lv, nm, ds, on, c]) => `<div class="preset" style="cursor:default;opacity:${on ? 1 : .5}">
    <div class="nm lv-${c}">${lv} · ${nm}</div><div class="ds">${ds} ${on ? "" : "(不满足)"}</div></div>`).join("");
  const extra = [];
  if (r.laplacian && r.laplacian.rel_residual < 0.35) {
    extra.push(r.laplacian.exact
      ? `<div class="small-txt" style="margin-top:6px">🧮 <b>精确分解</b>: K = ${fmt(r.laplacian.a)}·Δ₄ + ${fmt(r.laplacian.b)}·δ — 与离散拉普拉斯算子直接相关(微分结构, 高阶拉氏量素材)。</div>`
      : `<div class="small-txt" style="margin-top:6px">🧮 <b>近似</b> K ≈ ${fmt(r.laplacian.a)}·Δ₄ + ${fmt(r.laplacian.b)}·δ (相对残差 ${(r.laplacian.rel_residual * 100).toFixed(0)}%)</div>`);
  }
  if (state.note) extra.push(`<div class="fam-note" style="margin-top:8px">${esc(state.note)}</div>`);
  return `<div class="card fig">
    <h3>设计空间定位(对话层级体系)<span class="note">每层 ⊆ 上层</span></h3>
    <div class="preset-grid" style="grid-template-columns:repeat(auto-fill,minmax(130px,1fr))">${items}</div>
    <div class="formula" style="margin-top:10px">所有卷积核(L1 平移等变)
&nbsp;→&nbsp;+ 自伴/偶核 = L2(可导拉氏量)
&nbsp;→&nbsp;+ 正定 ĥ≥0 = L3(凸能量)
&nbsp;→&nbsp;+ 微分结构 = L4(格林核)
&nbsp;→&nbsp;+ 群作用 = L5(旋转等变)</div>
    ${extra.join("")}
  </div>`;
}

/* 汇总表卡 */
function summaryCard(r) {
  const rows = r.summary.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="mono">${v}</td></tr>`).join("");
  return `<div class="card fig">
    <h3>属性总表<span class="note">镜像对话分析的总结表</span></h3>
    <table class="sumt">${rows}</table>
  </div>`;
}

/* 结论卡 */
function conclCard(r) {
  const ic = { ok: "✔", warn: "⚠", info: "✱", bad: "✘" };
  const items = r.conclusions.map(c => `<li class="${c.t}"><span class="ic">${ic[c.t] || "·"}</span><span>${c.text}</span></li>`).join("");
  return `<div class="card fig" style="grid-column:1/-1">
    <h3>自动结论</h3>
    <ul class="concl">${items}</ul>
  </div>`;
}

/* 迷你当前算子 */
function renderMini() {
  const box = $("curMini");
  if (!state.kernel) { box.innerHTML = '<div class="empty">尚未载入内核</div>'; return; }
  const m = state.kernel.length;
  const dc = state.kernel.flat().reduce((a, b) => a + b, 0);
  box.innerHTML = `<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap">
    <canvas class="hm" id="hmMini" style="width:150px;height:150px;border:1px solid var(--line);background:#060a13"></canvas>
    <div class="small-txt">
      <div style="margin-bottom:4px"><b>${m}×${m} 核</b>${state.provenance ? " · " + esc(state.provenance) : ""}</div>
      <div>DC 增益: <code class="p">${fmt(dc)}</code></div>
      <div>中心值: <code class="p">${fmt(state.kernel[(m - 1) / 2][(m - 1) / 2])}</code></div>
      ${state.result ? `<div>谱: <code class="p">[${fmt(state.result.spectrum.re_min)}, ${fmt(state.result.spectrum.re_max)}]</code></div>
      <div>正定性: <code class="p">${state.result.psd ? "PSD" : state.result.nsd ? "NSD" : state.result.indefinite ? "不定" : "复符号"}</code></div>` : ""}
    </div></div>`;
  const cv = $("hmMini");
  if (cv) drawHeat(cv, state.kernel, { mode: "div" });
}

/* ================================================================
 * 操作: 尺寸 / 预置 / 参数化 / 随机 / 编辑动作
 * ================================================================ */
function setKernel(k, provenance, note, run = true) {
  // 深拷贝并规整
  state.kernel = k.map(row => row.map(v => (Number.isFinite(+v) ? +v : 0)));
  state.provenance = provenance || "";
  state.note = note || "";
  $("szSlider").value = state.kernel.length;
  $("szOut").textContent = state.kernel.length + "×" + state.kernel.length;
  $("curSizeTag").textContent = state.kernel.length + "×" + state.kernel.length;
  rebuildGrid();
  paintAll();
  renderMini();
  if (run) analyze();
}

function resizeTo(m) {
  const old = state.kernel || [[0]];
  const nk = [];
  const cOld = (old.length - 1) / 2, cNew = (m - 1) / 2;
  for (let i = 0; i < m; i++) {
    const row = [];
    for (let j = 0; j < m; j++) {
      const oi = i - cNew + cOld, oj = j - cNew + cOld;
      row.push(oi >= 0 && oi < old.length && oj >= 0 && oj < old.length ? old[oi][oj] : 0);
    }
    nk.push(row);
  }
  setKernel(nk, state.provenance, state.note);
}

/* 预置 UI */
function renderPresets() {
  const grid = $("presetGrid");
  grid.innerHTML = PRESETS.map(p => `<button class="preset" data-id="${p.id}">
    <div class="nm">${esc(p.name)}</div>
    <div><span class="badge lv-${p.lv.charAt(1) === "1" ? "1" : p.lv.charAt(1) === "2" ? "2" : p.lv.charAt(1) === "3" ? "3" : "4"}" style="color:var(--muted);border-color:var(--line)">${p.lv}</span>
    <span class="lv-${p.lv.charAt(1)}">${p.k.length}×${p.k.length}</span></div>
    <div class="ds">${esc(p.desc)}</div></button>`).join("");
  grid.querySelectorAll(".preset").forEach(btn => {
    btn.onclick = () => {
      const p = PRESETS.find(q => q.id === btn.dataset.id);
      setKernel(p.k, p.prov, "", true);
      toast("已载入预置: " + p.name);
    };
  });
}

/* 参数化族 UI */
function buildFamParams() {
  const fam = $("famSel").value;
  const def = FAMILIES[fam];
  const box = $("famParams");
  box.innerHTML = "";
  for (const p of def.params) {
    const wrap = document.createElement("div");
    wrap.className = "row";
    wrap.innerHTML = `<label>${esc(p.label)}</label>
      <input type="range" data-pk="${p.k}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.def}">
      <span class="val-out" data-pout="${p.k}">${p.def}</span>`;
    box.appendChild(wrap);
    const rng = wrap.querySelector("input");
    const out = wrap.querySelector("span");
    rng.oninput = () => { out.textContent = (+rng.value).toFixed(p.step < 1 ? 2 : 0); };
    wrap.querySelectorAll("input,select").forEach(x => x.addEventListener("change", () => $("famNote").textContent = FAMILIES[$("famSel").value].note));
  }
  $("famNote").textContent = def.note;
}

function readFamParams() {
  const fam = $("famSel").value;
  const o = {};
  document.querySelectorAll("#famParams input[type=range]").forEach(r => {
    o[r.dataset.pk] = parseFloat(r.value);
  });
  return { fam, params: o, m: parseInt($("famSize").value, 10) };
}

async function famGenerate() {
  const { fam, params, m } = readFamParams();
  // box 半径不能超过 (m-1)/2
  if (fam === "box") params.radius = Math.min(params.radius, (m - 1) / 2);
  const btn = $("btnFamGen");
  btn.disabled = true;
  try {
    const j = await api("/api/parametric", { family: fam, m, params });
    setKernel(j.kernel, fam, j.note, false);
    state.result = null;
    await analyze();
    toast(`已生成 ${FAMILIES[fam] ? FAMILIES[fam].note.split(":")[0] : fam} 核 ${m}×${m}`);
  } catch (e) { toast("生成失败: " + e.message, true); }
  finally { btn.disabled = false; }
}

/* 随机生成 */
async function randGenerate() {
  const m = parseInt($("randSize").value, 10);
  const kind = $("randKind").value;
  const seedTxt = $("randSeed").value.trim();
  const seed = seedTxt === "" ? undefined : seedTxt;
  const btn = $("btnRand");
  btn.disabled = true;
  try {
    const j = await api("/api/random", { m, kind, seed });
    const names = { general: "一般核", even: "偶核", odd: "奇核", psd: "正定核" };
    setKernel(j.kernel, "random:" + kind, "", false);
    state.result = null;
    await analyze();
    toast(`随机 ${names[kind]} ${m}×${m} 已生成并分析`);
  } catch (e) { toast("生成失败: " + e.message, true); }
  finally { btn.disabled = false; }
}

/* 导出 / 导入 */
function exportJSON() {
  if (!state.kernel) return;
  const obj = {
    app: "operator-lagrangian-lab",
    generated_at: new Date().toISOString(),
    kernel: state.kernel,
    provenance: state.provenance,
    result: state.result,
  };
  const blob = new Blob([JSON.stringify(obj, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `kernel-${currentSize()}x${currentSize()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}
function importFromText() {
  const ta = $("kjson");
  ta.classList.remove("err");
  const txt = ta.value.trim();
  let k = null;
  if (txt.startsWith("[")) {
    try { k = JSON.parse(txt); } catch { k = null; }
  } else {
    try {
      k = txt.split(/[;\n]/).filter(s => s.trim()).map(line =>
        line.split(",").map(s => parseFloat(s.trim())));
    } catch { k = null; }
  }
  const m = k ? k.length : 0;
  if (!k || !Array.isArray(k) || m % 2 === 0 || m > 15 || !k.every(row => Array.isArray(row) && row.length === m && row.every(v => Number.isFinite(+v)))) {
    ta.classList.add("err");
    toast("导入失败: 需要 N×N(N 为奇数 ≤15) 二维数值数组, 或 'a,b;c,d' 文本", true);
    return;
  }
  setKernel(k.map(r => r.map(v => +v)), state.provenance, state.note);
  toast(`已导入 ${m}×${m} 核`);
}
function copyKernel() {
  navigator.clipboard.writeText(JSON.stringify(state.kernel)).then(
    () => toast("已复制核 JSON"), () => toast("复制失败(需 https/localhost)", true));
}

/* ================================================================
 * 启动
 * ================================================================ */
function bindUI() {
  // tabs
  document.querySelectorAll("#genTabs button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll("#genTabs button").forEach(x => x.classList.toggle("active", x === b));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("active", p.id === b.dataset.tab));
    };
  });
  // 尺寸
  $("szSlider").addEventListener("input", () => {
    const m = parseInt($("szSlider").value, 10);
    $("szOut").textContent = m + "×" + m;
  });
  $("szSlider").addEventListener("change", () => resizeTo(parseInt($("szSlider").value, 10)));
  // 笔值
  document.querySelectorAll(".paint-row .btn[data-v]").forEach(b => {
    b.onclick = () => {
      state.paint = parseFloat(b.dataset.v);
      document.querySelectorAll(".paint-row .btn[data-v]").forEach(x => x.classList.toggle("sel", x === b));
    };
  });
  // 单元格点击填值
  $("kgrid").addEventListener("mousedown", (e) => {
    const inp = e.target.closest("input");
    if (!inp) return;
    if (e.button !== 0) return;
    // 仅在未选中文本时用笔值覆盖
    const sel = window.getSelection().toString();
    if (!sel) {
      const i = +inp.dataset.i, j = +inp.dataset.j;
      if (inp === document.activeElement) return;
      state.kernel[i][j] = state.paint;
      inp.value = fmt(state.kernel[i][j], 3);
      paintCellClass(inp, state.kernel[i][j]);
      scheduleAnalyze();
      e.preventDefault();
    }
  });
  // 双击进入手动输入
  $("kgrid").addEventListener("dblclick", (e) => {
    const inp = e.target.closest("input");
    if (inp) inp.focus();
  });
  $("btnRun").onclick = () => analyze();
  $("btnClear").onclick = () => {
    const m = currentSize();
    setKernel(Array.from({ length: m }, () => Array(m).fill(0)), "", "");
  };
  $("btnSym").onclick = () => {
    const ks = state.result ? state.result.parity.ks : null;
    if (ks) setKernel(ks, state.provenance + "·偶部", "");
  };
  $("btnAnti").onclick = () => {
    const ka = state.result ? state.result.parity.ka : null;
    if (ka) setKernel(ka, state.provenance + "·奇部", "");
  };
  $("btnImp").onclick = importFromText;
  $("btnCpy").onclick = copyKernel;
  $("btnExport").onclick = exportJSON;
  // 参数化
  $("famSel").addEventListener("change", buildFamParams);
  $("famSize").addEventListener("input", () => $("famSizeOut").textContent = $("famSize").value);
  $("famSize").addEventListener("change", () => $("famSizeOut").textContent = $("famSize").value);
  $("btnFamGen").onclick = famGenerate;
  buildFamParams();
  // 随机
  $("randSize").addEventListener("input", () => $("randSizeOut").textContent = $("randSize").value);
  $("btnRand").onclick = randGenerate;
  renderPresets();
}

async function boot() {
  bindUI();
  try {
    const h = await fetch("/api/health").then(r => r.json());
    $("engineMeta").textContent = `引擎已连接 · ${h.time}`;
  } catch {
    $("engineMeta").textContent = "引擎未连接(需 python3 backend/main.py)";
    toast("后端未连接: 请运行 python3 backend/main.py 后刷新", true);
    return;
  }
  // 默认载入对话主角: 十字核
  const cross = PRESETS.find(p => p.id === "cross");
  setKernel(cross.k, cross.prov, "", false);
  await analyze();
}

document.addEventListener("DOMContentLoaded", boot);
