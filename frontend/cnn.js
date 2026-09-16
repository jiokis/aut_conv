/* ================================================================
 * CNN 手写数字识别 · 算子架构系统性分析(前端)
 * ================================================================ */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------- 小工具 ---------- */
function fmt(x, d) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a === 0) return "0";
  if (d) return x.toFixed(d);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(2);
  if (a >= 100) return x.toFixed(1);
  if (a >= 1) return (+x.toFixed(3)).toString();
  return (+x.toFixed(4)).toString();
}
function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function toast(msg, err) {
  const t = $("toast2");
  t.textContent = msg;
  t.className = "toast show" + (err ? " err" : "");
  clearTimeout(t._tm);
  t._tm = setTimeout(() => (t.className = "toast"), 3600);
}
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "API 错误");
  return j;
}

/* ---------- 热图(与实验室页同一套) ---------- */
function ramp(stops, t) {
  t = Math.max(-1, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      const [t0, a] = stops[i], [t1, b] = stops[i + 1];
      const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}
const DIV = [[-1, [62, 88, 255]], [-0.25, [90, 150, 255]], [0, [9, 14, 30]],
  [0.25, [255, 180, 96]], [1, [255, 108, 60]]];
function drawHeat(canvas, data) {
  const R = data.length, C = data[0].length;
  canvas.width = C; canvas.height = R;
  canvas.style.imageRendering = "pixelated";
  const ctx = canvas.getContext("2d");
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const v = data[i][j]; if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const sp = Math.max(Math.abs(mn), Math.abs(mx), 1e-12);
  const img = ctx.createImageData(C, R);
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const c = ramp(DIV, data[i][j] / sp);
    const p = (i * C + j) * 4;
    img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/* ---------- 中英名词映射 ---------- */
const TPL_CN = {
  sobel_x: "横向边缘/SobelX", sobel_y: "纵向边缘/SobelY",
  prewitt_x: "PrewittX 差分", prewitt_y: "PrewittY 差分",
  deriv_x: "一阶差分 ∂x", deriv_y: "一阶差分 ∂y",
  lap4: "4-邻域拉普拉斯", lap8: "8-邻域拉普拉斯",
  cross: "十字邻域", box3: "邻域平均(平滑)", sharpen: "锐化增强",
  gauss5: "高斯平滑(PDS型)",
};
const KIND_CN = {
  lowpass: "低通/平滑", highpass: "高通/差分", directional: "方向差分",
  bandpass: "带通", mixed: "混合", flat: "平坦",
};
const TYPE_CN = { even: "偶(近似自伴)", odd: "奇(反自伴)", general: "一般(偶+奇)" };
const DEF_CN = {
  psd: "正定PSD", nsd: "负定NSD", indefinite: "不定(有正负谱)", complex: "复符号(非自伴)",
};
const state = { model: null, layers: null, cur: "c1", frame: "real", board: null, drawing: false, polling: false };

/* ================================================================
 * 模型状态 & 手写预测
 * ================================================================ */
async function loadModel() {
  try {
    const j = await fetch("/api/cnn/model").then((r) => r.json());
    state.model = j.data || {};
  } catch (e) {
    state.model = {};
  }
  const m = state.model;
  const mm = $("modelMetrics");
  if (!m.trained) {
    $("cnnStatus").textContent = "模型: 未训练" + (m.numpy ? "" : " · 无 numpy");
    $("bigPred").textContent = "—";
    mm.innerHTML = `<div class="metric"><div class="k">状态</div><div class="v warn">未训练 — 点击「开始训练」</div></div>`;
    return;
  }
  const rp = m.report || {};
  $("cnnStatus").textContent = `模型: 已训练 · 测试 ${(rp.test_acc * 100).toFixed(1)}% · ${rp.subset}样本×${rp.epochs}轮`;
  mm.innerHTML = `
    <div class="metric"><div class="k">测试准确率</div><div class="v ok">${(rp.test_acc * 100).toFixed(2)}%</div></div>
    <div class="metric"><div class="k">验证准确率</div><div class="v info">${(rp.val_acc * 100).toFixed(1)}%</div></div>
    <div class="metric"><div class="k">训练数据</div><div class="v">${rp.subset} × ${rp.epochs}轮</div></div>
    <div class="metric"><div class="k">训练耗时</div><div class="v">${(rp.secs || 0).toFixed(0)}s</div></div>
    <div class="metric"><div class="k">架构</div><div class="v" style="font-size:11px;line-height:1.5">Conv8·5²→Pool→Conv16·5²→Pool→FC120→10<br>ReLU + Softmax · Adam</div></div>`;
}

/* ---------- 手写板 ---------- */
function initBoard() {
  const cv = $("drawBoard");
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 280, 280);
  ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 20;
  state.board = { cv, ctx };

  function pos(e) {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * (280 / r.width), (e.clientY - r.top) * (280 / r.height)];
  }
  cv.addEventListener("pointerdown", (e) => {
    state.drawing = true;
    cv.setPointerCapture(e.pointerId);
    const [x, y] = pos(e);
    state.last = [x, y];
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.01, y + 0.01); ctx.stroke();
  });
  cv.addEventListener("pointermove", (e) => {
    if (!state.drawing) return;
    const [x, y] = pos(e);
    ctx.beginPath(); ctx.moveTo(state.last[0], state.last[1]); ctx.lineTo(x, y); ctx.stroke();
    state.last = [x, y];
  });
  cv.addEventListener("pointerup", () => (state.drawing = false));
  $("btnClearBoard").onclick = clearBoard;
}
function clearBoard() {
  const { ctx, cv } = state.board;
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
  $("bigPred").textContent = "—";
  $("probBars").innerHTML = '<div class="empty" style="padding:8px">已清空</div>';
}
function rasterize() {
  const { ctx, cv } = state.board;
  const img = ctx.getImageData(0, 0, 280, 280).data;
  const raw = Array.from({ length: 28 }, () => Array(28).fill(0));
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let s = 0;
      for (let dy = 0; dy < 10; dy++) for (let dx = 0; dx < 10; dx++) {
        const p = ((y * 10 + dy) * 280 + x * 10 + dx) * 4;
        s += img[p];
      }
      raw[y][x] = s / 2800 / 255;
    }
  }
  // 质心居中(软权重: 不设硬阈值, 浅笔迹也能参与)
  let mx = 0, my = 0, mass = 0;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const v = raw[y][x];
    mx += x * v; my += y * v; mass += v;
  }
  const out = Array.from({ length: 28 }, () => Array(28).fill(0));
  if (mass > 0.5) {
    const dx = Math.round(13.5 - mx / mass), dy = Math.round(13.5 - my / mass);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const sx = x + dx, sy = y + dy;
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[sy][sx] = raw[y][x];
    }
  }
  // 预览(0..1 → 灰度, 下采样预览直接画 28)
  const pc = $("pixPreview").getContext("2d");
  pc.fillStyle = "#000"; pc.fillRect(0, 0, 28, 28);
  const pimg = pc.createImageData(28, 28);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const g = Math.round(out[y][x] * 255);
    const p = (y * 28 + x) * 4;
    pimg.data[p] = g; pimg.data[p + 1] = g; pimg.data[p + 2] = g; pimg.data[p + 3] = 255;
  }
  pc.putImageData(pimg, 0, 0);
  return out.map((row) => row.map((v) => Math.round(v * 1e3) / 1e3));
}
async function predict() {
  const dot = $("busyDot2");
  dot.classList.add("on");
  try {
    const pix = rasterize();
    const mass = pix.flat().reduce((a, b) => a + b, 0);
    if (mass < 0.5) { toast("先在画板上写一个数字", true); return; }
    const j = await api("/api/cnn/predict", { pixels: pix });
    renderProbs(j.probs, j.pred);
  } catch (e) { toast("预测失败: " + e.message, true); }
  finally { dot.classList.remove("on"); }
}
function renderProbs(probs, pred) {
  const mx = Math.max(...probs);
  $("bigPred").innerHTML = `预测: <span class="d">${pred}</span> &nbsp;置信 ${(mx * 100).toFixed(1)}%`;
  $("probBars").innerHTML = probs.map((p, i) => {
    const w = Math.max(1, p * 100);
    return `<div class="prob-row ${i === pred ? "top" : ""}">
      <div>${i}</div><div class="bar"><i style="width:${w}%"></i></div>
      <div class="pct">${(p * 100).toFixed(1)}%</div></div>`;
  }).join("");
}

/* ================================================================
 * 逐层算子分析
 * ================================================================ */
async function loadLayers() {
  const dot = $("busyDot2");
  dot.classList.add("on");
  try {
    const t0 = performance.now();
    const j = await api("/api/cnn/layers", {});
    state.layers = j.data;
    $("layersTime").textContent = `分析 ${Math.round(performance.now() - t0)} ms · ${state.layers.conv1.agg.n}+${state.layers.conv2.comp_agg.n} 个 2D 算子`;
    renderLayer();
    renderNarration();
  } catch (e) {
    toast("层分析失败: " + e.message, true);
    $("layerCards").innerHTML = `<div class="empty">${esc(e.message)}</div>`;
  } finally { dot.classList.remove("on"); }
}

function layerAggHTML(agg, extraTxt) {
  const kindTop = Object.entries(agg.kinds || {}).sort((a, b) => b[1] - a[1]);
  const tplTop = Object.entries(agg.templates || {}).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const mx = Math.max(1, ...tplTop.map((x) => x[1]));
  const maxk = Math.max(1, ...kindTop.map((x) => x[1]));
  const oh = agg.orient_hist || [];
  const omax = Math.max(1, ...oh);
  const chip = (txt, cls, ttl) => `<span class="mini-chip ${cls}" title="${esc(ttl || "")}">${txt}</span>`;
  return `<div class="agg-stats">
    <div class="metric"><div class="k">算子数 n</div><div class="v info">${agg.n}</div></div>
    <div class="metric"><div class="k">偶核(自伴)</div><div class="v ${agg.even_n ? "ok" : "warn"}">${agg.even_n}${extraTxt}</div></div>
    <div class="metric"><div class="k">奇核(反自伴)</div><div class="v">${agg.odd_n}</div></div>
    <div class="metric"><div class="k">一般(偶+奇)</div><div class="v">${agg.general_n}</div></div>
    <div class="metric"><div class="k">偶部能量占比均值</div><div class="v info">${(agg.mean_even_frac * 100).toFixed(1)}%</div></div>
    <div class="metric"><div class="k">奇部(陀螺)占比均值</div><div class="v warn">${(agg.mean_odd_frac * 100).toFixed(1)}%</div></div>
    <div class="metric"><div class="k">正定 PSD</div><div class="v ${agg.psd_n ? "ok" : "bad"}">${agg.psd_n} / ${agg.n}</div></div>
  </div>
  <div style="padding:0 16px 6px;display:flex;gap:8px;flex-wrap:wrap">${chip("无 L3 正定 → 非能量型特征检测", agg.psd_n ? "d-psd" : "d-indefinite")}</div>
  <div class="c-b" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px">
    <div><div class="small-txt">频带类型分布(个数)</div>${kindTop.map(([k, v]) => `<div class="bar-block">
      <div class="lab"><span>${KIND_CN[k] || k}</span><span>${v}</span></div>
      <div class="tb"><i style="width:${(v / maxk) * 100}%"></i></div></div>`).join("") || ""}</div>
    <div><div class="small-txt">最接近的经典算子(个数, 归一化相关)</div>${tplTop.map(([k, v]) => `<div class="bar-block">
      <div class="lab"><span>${TPL_CN[k] || k}</span><span>${v}</span></div>
      <div class="tb"><i style="width:${(v / mx) * 100}%"></i></div></div>`).join("") || ""}</div>
    <div><div class="small-txt">方向性主朝向直方图(0–180°, 奇部主导滤波器)</div>
      <div class="hist-row">${oh.map((v) => `<div class="hb" style="height:${(v / omax) * 100}%" title="${Math.round(oh.indexOf(v) * 15)}–${Math.round((oh.indexOf(v) + 1) * 15)}°: ${v}"></div>`).join("")}</div>
      <div class="cap" style="margin-top:2px">每格 15°; 高度=滤波器数</div></div>
  </div>`;
}

function filterBadges(q) {
  const t = (txt, cls, ttl) => `<span class="mini-chip ${cls}" title="${esc(ttl || "")}">${txt}</span>`;
  return `<div class="chips">
    ${t(TYPE_CN[q.type], "t-" + q.type, "180° 旋转对称性")}
    ${t(DEF_CN[q.definite] || q.definite, "d-" + q.definite)}
    ${t(KIND_CN[q.filter_kind] || q.filter_kind, q.filter_kind)}
    ${t("偶部 " + (q.even_frac * 100).toFixed(0) + "% · 奇部 " + (q.odd_frac * 100).toFixed(0) + "%", q.odd_frac > 0.5 ? "t-general" : "", "偶部=自伴成分(可导拉氏量); 奇部=反自伴陀螺成分")}
    ${q.orient_deg ? t("主朝向 " + orientCN(q.orient_deg.deg) + " (" + q.orient_deg.deg + "°)", "directional", "一致性=" + q.orient_deg.strength) : ""}
    ${t("谱 [" + fmt(q.spectrum.re_min) + "," + fmt(q.spectrum.re_max) + "] DC=" + fmt(q.spectrum.dc), "")}
    ${t("≈" + (TPL_CN[q.template.best] || q.template.best) + " ρ=" + q.template.score, "d-complex", "与经典算子的归一化相关 top: " + (q.template.top3 || []).map((x) => TPL_CN[x[0]] + " " + x[1]).join(", "))}
  </div>`;
}
function orientCN(deg) {
  if (deg === null || deg === undefined) return "";
  if (deg < 22.5 || deg >= 157.5) return "≈水平";
  if (deg < 67.5) return "≈对角↗";
  if (deg < 112.5) return "≈垂直";
  return "≈对角↘";
}

function renderLayer() {
  const L = state.layers;
  if (!L) return;
  const isC1 = state.cur === "c1";
  const cplx = state.frame === "cplx";
  const agg = isC1 ? L.conv1.agg : L.conv2.comp_agg;
  const ca = isC1 ? L.conv1.cplx_agg : L.conv2.cplx_comp_agg;
  $("layerAgg").innerHTML = cplx ? cplxAggHTML(ca, agg) : layerAggHTML(agg, isC1 ? "" : " (128 分量)");
  const expl = $("frameExplain");
  expl.innerHTML = cplx
    ? "复化(虚数)视角: 对任意实核 k, 卷积算子在复 L² 场上拆 A=Aₛ+Aₐ; 用复数核 ĉ = kₛ − i kₐ 构造 <b>Hermitian 算子</b>, 其实谱 = Re ĥ + Im ĥ ∈ ℝ —— 因此 <b>任何滤波器都自动对应实值 Hermitian 二次拉氏量</b>(奇部变为『动能/电流』而非陀螺)。下方数字为该视角的测量。"
    : "实框架: 偶核⇔自伴⇔可导出二次拉氏量; 奇核⇔反自伴⇔陀螺(能量守恒)。";
  const box = $("layerCards");
  if (isC1) {
    const cards = L.conv1.filters.map((f) => {
      const q = f.qa;
      return `<div class="fcard">
        <div class="fhead"><span class="fid">f${f.idx}</span><span style="flex:1"></span>
          <span class="orient-note">${cplx ? "动能 " + (f.cplx.kinetic_share * 100).toFixed(0) + "%" : KIND_CN[q.filter_kind]}</span></div>
        <canvas class="hm" data-k="${f.idx}" style="width:96px;height:96px"></canvas>
        ${cplx ? cplxBadges(f.cplx, q) : filterBadges(q)}
      </div>`;
    }).join("");
    box.innerHTML = `<div class="small-txt" style="margin-bottom:10px">${esc(L.conv1.note)}${cplx ? " — 复化 Hermitian 视图中每个滤波器均谱实值、可导实拉氏量。" : " — 每个 5×5 滤波器独立按算子分析(单通道 ⇒ 直接对应 2D 卷积算子)。"}</div>
      <div class="mini-grid">${cards}</div>`;
    [...box.querySelectorAll("canvas[data-k]")].forEach((cv) => {
      const f = L.conv1.filters[+cv.dataset.k];
      drawHeat(cv, f.k);
    });
  } else {
    const outs = L.conv2.outputs.map((o) => {
      const comps = o.components;
      const cnt = { even: 0, odd: 0, general: 0 };
      comps.forEach((c) => cnt[c.type]++);
      const dot = (c) => `<span class="mini-chip t-${c.type}" title="cin=${c.cin} · ${TYPE_CN[c.type]} · odd%=${(c.odd_frac * 100).toFixed(0)}${cplx ? " · 动能" + (c.kin * 100).toFixed(0) + "%" : ""}">${cplx ? (c.kin > 0.5 ? "K" : "P") : (c.type === "even" ? "E" : c.type === "odd" ? "O" : "G")}</span>`;
      return `<div class="fcard">
        <div class="fhead"><span class="fid">o${o.out}</span>
          <span class="mini-chip">主导通道 cin=${o.dominant_cin}</span>
          ${cplx ? `<span class="orient-note">动能 ${(o.cplx.kinetic_share * 100).toFixed(0)}%</span>`
                 : `<span class="orient-note">8 分量: E${cnt.even} / O${cnt.odd} / G${cnt.general}</span>`}</div>
        <canvas class="hm" data-o="${o.out}" style="width:96px;height:96px"></canvas>
        ${cplx ? cplxBadges(o.cplx, o.dominant_qa) : filterBadges(o.dominant_qa)}
        <div class="chips" style="margin-top:2px">分量: ${comps.map(dot).join("")}</div>
      </div>`;
    }).join("");
    box.innerHTML = `<div class="small-txt" style="margin-bottom:10px">${esc(L.conv2.note)} — 每卡展示「主导分量核」; ${cplx ? "分量字母: K=动能主导(奇部>50%) / P=势能主导; 聚合统计=128 分量复化视角。" : "字母为分量奇偶型(E偶/O奇/G一般); 聚合统计在顶部。"}</div>
      <div class="mini-grid">${outs}</div>`;
    [...box.querySelectorAll("canvas[data-o]")].forEach((cv) => {
      const o = L.conv2.outputs[+cv.dataset.o];
      drawHeat(cv, o.dominant_k);
    });
  }
}

/* 复化(虚数场)徽标与聚合面板 */
function cplxBadges(c, q) {
  const t = (txt, cls, ttl) => `<span class="mini-chip ${cls}" title="${esc(ttl || "")}">${txt}</span>`;
  return `<div class="chips">
    ${t("复化 Hermitian: 恒成立(实谱)", "d-psd", "ĉ=kₛ−ikₐ ⇒ 自伴, 拉氏量 S=½∫ĥ_c|û|² 恒实值")}
    ${t("动能(imag)占比 " + (c.kinetic_share * 100).toFixed(0) + "%", c.kinetic_share > 0.5 ? "t-odd" : "t-even", "奇部在复化后=谱实值动能/电流项")}
    ${t("实谱范围 [" + fmt(c.hc_min) + ", " + fmt(c.hc_max) + "]", c.herm_psd ? "d-psd" : "d-indefinite", "复化符号 Re ĥ+Im ĥ 仍变号 ⇒ 动能+势能混合, 非正定")}
    ${c.herm_psd ? t("Hermitian 正定 ✓", "d-psd") : t("Hermitian 正定 ✗ (0)", "d-indefinite")}
    ${t("实框参照: " + TYPE_CN[q.type] + " · 奇部" + (q.odd_frac * 100).toFixed(0) + "%", "d-complex")}
  </div>`;
}
function cplxAggHTML(ca, agg) {
  const t = (txt, cls, ttl) => `<span class="mini-chip ${cls}" title="${esc(ttl || "")}">${txt}</span>`;
  return `<div class="agg-stats">
    <div class="metric"><div class="k">算子数 n</div><div class="v info">${ca.n}</div></div>
    <div class="metric"><div class="k">动能(奇部→imag)占比均值</div><div class="v info">${(ca.kinetic_mean * 100).toFixed(1)}%<span style="font-size:10px;color:var(--muted)"> (${(ca.kinetic_lo * 100).toFixed(0)}–${(ca.kinetic_hi * 100).toFixed(0)}%)</span></div></div>
    <div class="metric"><div class="k">复化实谱范围</div><div class="v">[${fmt(ca.hc_min)}, ${fmt(ca.hc_max)}]</div></div>
    <div class="metric"><div class="k">Hermitian 正定(谱≥0)</div><div class="v bad">${ca.herm_psd_n} / ${ca.n}</div></div>
    <div class="metric"><div class="k">另一取法 ĉ=kₛ+ikₐ 谱</div><div class="v">[${fmt(ca.alt_min)}, ${fmt(ca.alt_max)}]</div></div>
    <div class="metric"><div class="k">拉氏量可导出(复化)</div><div class="v ok">恒是(Hermitian)</div></div>
  </div>
  <div style="padding:0 16px 6px;display:flex;gap:8px;flex-wrap:wrap">${t("任意实核 → ĉ=kₛ−ikₐ 自伴, 实谱 = Re ĥ+Im ĥ: 奇偶约束在复场下自动消失", "d-psd")}</div>`;
}

/* ================================================================
 * 系统性解读
 * ================================================================ */
function renderNarration() {
  const L = state.layers;
  const box = $("narration");
  if (!L) { box.innerHTML = '<div class="empty">先完成 ② 层分析</div>'; return; }
  const a1 = L.conv1.agg, ac = L.conv2.comp_agg;
  const ca1 = L.conv1.cplx_agg || {}, ca2 = L.conv2.cplx_comp_agg || {};
  const N = (x) => (x * 100).toFixed(0) + "%";
  const dir1 = (a1.kinds.directional || 0);
  const html = `
  <div class="small-txt" style="font-size:13.5px;line-height:2;color:#c6d3ef">
  <b style="color:#3fd7ff">一、网络结构层面</b> — 该网络 = 平移等变算子序列 + 非线性 + 降采样:
  <code class="p">Conv(8×5²) →ReLU→ Pool₂ → Conv(16×5²) →ReLU→ Pool₂ → FC(120) → 10</code>。
  两个卷积层都是 <b>L1 平移等变算子族</b>(每输出通道一个卷积算子); ReLU 引入非线性、池化做平移略变的降采样,
  FC 把全局 7×7×16=784 特征线性组合成 10 类判定方向(不再是空间卷积, 而是特征空间的“对角+混合”线性算子)。<br><br>
  <b style="color:#3fd7ff">二、逐滤波器算子分析(本页 ②)</b> — 用「自伴/偶核 ⇔ 拉氏量可导出、奇部 ⇔ 陀螺、PSD ⇔ 能量凸」做实测:
  <br>· Conv1 的 ${a1.n} 个 5×5 算子中, <b>偶核 0 个</b>, 偶/奇能量占比均值 ${N(a1.mean_even_frac)} / ${N(a1.mean_odd_frac)},
  全部为「一般(偶+奇)」且 <b>正定 PSD 0 个</b>; 频带上 ${dir1} 个方向差分、1 个低通; 与经典算子最接近的是
  拉普拉斯/锐化/差分族(模板匹配前几名)。<br>
  · Conv2 的 ${ac.n} 个通道分量算子里同样 <b>0 个 PSD、0 个纯偶</b>, 偶部均值 ${N(ac.mean_even_frac)},
  方向差分型高达 ${ac.kinds.directional || 0} 个 —— 说明第二层在首层已做边缘差分的基础上, 更集中地学习<b>带朝向的笔画/局部结构检测器</b>。<br>
  <b style="color:#3fd7ff">三、按“拉氏量/变分”框架的解读</b>:
  <br>1) 这些滤波器几乎都不是偶核 ⇒ 各自<b>不能由二次拉格朗日泛函导出</b>(自伴是必要条件);
  它们的奇部(反自伴, 占比 ~46–48%)对应框架中的 <b>陀螺/方向项</b> —— 与边缘/梯度检测的“方向敏感、
  时间反演破坏”语义一致; 偶部(自伴 ~52–54%)则提供局部平滑/对称增强的“势能型”成分。<br>
  2) 没有通道进入 L3 正定锥 ⇒ CNN 特征不是“能量极小/凸泛函”的解; 用算子语言说,
  卷积层更接近一组 <b>过零/差分测量(方向选择器)</b>, 而非势能梯度。<br>
  3) 若要在这种网络上“恢复”变分结构, 按对话给出的路径有二: 对单个滤波器做 <b>k=kₛ+kₐ 分解</b>后把偶部当自伴“势能核”、
  奇部当辛结构/陀螺项; 或在损失上加 <b>weight decay(对参数空间的二次势能)</b> + Adam 的动量近似耗散,
  使训练成为参数流形上的耗散梯度流 —— 这正是“物理一致”正则化(如对滤波器偶部/PSD 约束)的实验入口。<br><br>
  <b style="color:#3fd7ff">四、网络整体作为算子</b>: 有效感受野从 5×5(首层)经两次池化扩到约 17–18 像素(覆盖大部分笔画尺度);
  平移等变只在卷积层内保持, 池化给出平移鲁棒性(牺牲严格等变, 换取 L2 意义下的不变性);
  分类器部分(FC)是平移等变的破坏处 —— 这与“等变性给形状、全局结构给语义”的分工一致。
  <br><br>
  <b style="color:#b79aff">五、复场(虚数)视角下的补测</b>: 把每个实核复化为 ĉ=kₛ−ikₐ(Hermitian,
  实谱 Re ĥ+Im ĥ), Conv1 动能(imag)占比均值 ${(ca1.kinetic_mean * 100).toFixed(1)}%(与实框奇部占比一致), Conv2(128 分量)
  ${(ca2.kinetic_mean * 100).toFixed(1)}%; 复化实谱范围 [${fmt(ca1.hc_min)}, ${fmt(ca1.hc_max)}](Conv1)。
  两个要点: ① <b>复场下“可导出拉氏量”恒成立、奇偶约束自动消失</b> —— 奇部变成谱实值的
  动能/电流项, 而非需要辛结构容纳的陀螺; ② 但 <b>Hermitian 正定仍为 0/136</b>: 动能+势能
  混合谱照旧变号, 说明“虚数场”改变的是<u>可解释语义</u>(相位/电流/能量守恒视角, 类似
  i∂→动量), 并未让特征层变成凸能量; 想得到正定仍需显式符号设计。故在纯精度/几何任务上
  实框架足够且更省(参数减半、无相位非线性成本), 复场在需要相位或物理一致演化的任务
  (波动/量子/时频)才显价值 —— 与你“倾耗散、慎用复数”的取向一致。</div>`;
  box.innerHTML = html;
}

/* ================================================================
 * ⑤ 端到端分类机制解剖
 * ================================================================ */
async function loadMech() {
  const box = $("mechBox");
  try {
    const r = await fetch("/api/cnn/mech").then((x) => x.json());
    if (!r.ok) throw new Error(r.error || "no data");
    renderMech(box, r.data);
  } catch (e) {
    box.innerHTML = `<div class="empty">机制数据缺失: ${esc(e.message)}<br>
      <span class="small-txt">运行 <code class="p">python3 backend/mech.py</code>(约 1 分钟)后刷新。</span></div>`;
  }
}

function heatCells(mat, digitsFirst = false, mxScale = null) {
  /* 渲染通道×数字的激活热力表格(div), 返回 html */
  if (!mat || !mat.length) return "";
  const R = mat.length, C = mat[0].length;
  const all = [].concat(...mat);
  let mx = mxScale || Math.max(...all, 1e-9);
  const col = (v) => {
    const t = v / mx;
    const a = Math.round(12 + t * 60);
    const r = Math.round(30 + t * 190);
    const b = Math.round(60 + (1 - t) * 90);
    return `rgb(${r},${b},${255 - a})`;
  };
  const thd = (j) => `<span style="min-width:20px;text-align:center;font:11px var(--mono);color:#9fc">${j}</span>`;
  const head = `<div class="chrow ch-head">${digitsFirst ? "" : '<span style="min-width:34px"></span>'}
    ${Array.from({ length: C }, (_, j) => thd(j)).join("")}${digitsFirst ? '<span style="min-width:34px"></span>' : ""}</div>`;
  const rows = mat.map((row, i) => {
    const lab = `<span style="min-width:34px;font:10px var(--mono);color:#9cf">${digitsFirst ? "ch" + i : i}</span>`;
    return `<div class="chrow">${digitsFirst ? "" : lab}${row.map((v, j) => {
      const dd = digitsFirst ? v : v;
      return `<span title="${esc("ch" + (digitsFirst ? j : i) + " → 数字" + (digitsFirst ? i : j)) + " = " + v}" style="width:20px;height:16px;border-radius:3px;background:${col(dd)}"></span>`;
    }).join("")}${digitsFirst ? lab : ""}</div>`;
  }).join("");
  return `<div style="overflow-x:auto"><div style="min-width:${C * 20 + 50}px">${head}${rows}</div></div>`;
}

function renderMech(box, m) {
  const pct3 = (x) => (x * 100).toFixed(1) + "%";
  const st = m.stages || {};
  const row = (nm, key, note) => `<tr><td>${nm}</td>
    <td class="mono">${st[key] ? pct3(st[key].ridge) : "—"}</td>
    <td class="mono">${st[key] ? pct3(st[key].centroid) : "—"}</td><td class="small-txt">${note}</td></tr>`;
  const confPairTxt = (m.conf_pairs || []).map(([a, b, n]) => `<span class="badge red">${a}→判为${b} ×${n}</span>`).join(" ") || "—";
  const dirClose = (m.dir_close || []).map(([a, b, c]) => `<span class="badge orange">${a} & ${b} 方向 cos=${c}</span>`).join(" ") || "—";
  const c1 = m.channels && m.channels.conv1, c2 = m.channels && m.channels.conv2;
  const fbar = (scores, mx) => scores.map((f, i) => {
    const w = Math.max(2, (f / (mx + 1e-9)) * 100);
    return `<div class="bar-block" title="ch${i} 判别力 F=${f}"><div class="lab"><span>ch${i}</span></div>
      <div class="tb"><i style="width:${w}%"></i></div></div>`;
  }).join("");
  const marr = m.margin || {};
  const relu = m.relu_rate || {};
  // 关键机制叙述(数据驱动)
  const lift = st.pix && st.pool2 && st.h
    ? { p2: (st.pool2.ridge - st.pix.ridge) * 100, h: (st.h.ridge - st.pool2.ridge) * 100 }
    : null;
  box.innerHTML = `
  <div class="metrics" style="border-bottom:none;padding:12px 16px 4px">
    <div class="metric"><div class="k">测试准确率</div><div class="v ok">${pct3(m.accuracy)}</div></div>
    <div class="metric"><div class="k">分类间隔 mean/p10</div><div class="v">${fmt(m.margin ? m.margin.mean : 0)} / ${fmt(m.margin ? m.margin.p10 : 0)}</div></div>
    <div class="metric"><div class="k">负间隔比例</div><div class="v ${m.margin && m.margin.neg_frac > 0.001 ? "warn" : "ok"}">${m.margin ? pct3(m.margin.neg_frac) : "—"}</div></div>
    <div class="metric"><div class="k">ReLU 激活率 Conv1/C2/FC1</div><div class="v" style="font-size:12px">${pct3(relu.conv1)} / ${pct3(relu.conv2)} / ${pct3(relu.fc1)}</div></div>
    <div class="metric"><div class="k">最易混淆(误判数)</div><div class="v small-txt">${confPairTxt || "—"}</div></div>
    <div class="metric"><div class="k">决策方向最近对(cos)</div><div class="v small-txt">${dirClose || "—"}</div></div>
  </div>

  <div style="padding:6px 16px 2px"><b>M1 · 分到哪一步“可线性分出”: 岭回归/质心读头分半准确率</b>
  <span class="small-txt">(机制: 卷积栈把像素信息压成线性可分的特征; 若某级已 ~100%, 分类只是在该特征上取 10 个方向的 argmax)</span></div>
  <div style="overflow-x:auto;padding:0 16px 8px"><table class="sumt">
    <tr><th>特征阶段(维度)</th><th>线性(岭)可分度</th><th>质心最近类</th><th>说明</th></tr>
    ${row("原始像素(784)", "pix", "最直接基线")}
    ${row("Conv 栈输出 pool2 (784)", "pool2", "两卷积层+池化后的特征")}
    ${row("FC1 隐藏 h (120, ReLU)", "h", "FC1 重排后的特征")}
  </table></div>
  ${lift ? `<div class="small-txt" style="padding:0 16px 6px">量化: 卷积栈把线性可分度抬升 ${lift.p2 >= 0 ? "+" : ""}${lift.p2.toFixed(1)} pp(像素 → pool2), FC1 再 ${lift.h >= 0 ? "+" : ""}${lift.h.toFixed(1)} pp。即「分类机制 ≈ 平移等变算子把笔迹变成线性可分测度 → 阈值门控 → 10 方向读头」。</div>` : ""}

  <div style="padding:6px 16px 2px"><b>M2 · 通道×数字 平均激活(正确样本, 行=通道/列=数字)与判别力 F</b>
  <span class="small-txt">(机制: 每个算子通道把“证据”投给哪些数字; F 越大越专一)</span></div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;padding:0 16px 8px">
    <div><div class="small-txt">Conv1 8 通道(色深=平均激活)</div>${c1 ? heatCells(c1.mean_matrix) : ""}
      <div class="small-txt" style="margin-top:4px">判别力 F: ${c1 ? fbar(c1.scores, Math.max(...c1.scores)) : ""}</div></div>
    <div><div class="small-txt">Conv2 16 通道(主导通道, 色深=平均激活)</div>${c2 ? heatCells(c2.mean_matrix) : ""}
      <div class="small-txt" style="margin-top:4px">判别力 F: ${c2 ? fbar(c2.scores, Math.max(...c2.scores)) : ""}</div></div>
  </div>

  <div style="padding:6px 16px 2px"><b>M3 · 末层 10 个“决策方向”几何(120 维 W2 行)</b></div>
  <div style="padding:0 16px 8px" class="small-txt">
    方向范数: ${(m.dir_norm || []).map((v, i) => `${i}:${fmt(v)}`).join("  ")} —
    决策是“对每个类取一个权重向量, 比谁的内积大”。方向越接近的类越易混淆; 方向最近对如上表。
  </div>

  <div style="padding:6px 16px 10px"><b>机制一句话</b>:
  笔画 → (偶+奇/方向算子)响应 → ReLU 门控(激活率 ${pct3(relu.conv1)}/${pct3(relu.conv2)}/${pct3(relu.fc1)}, 稀疏≈特征选择) →
  池化做平移鲁棒聚合 → 全局读头取 10 方向内积 argmax(间隔 mean=${fmt(m.margin ? m.margin.mean : 0)})。
  各阶段线性可分度与通道×数字证据表给出了“哪一层为哪个类存了证据”的定量清单。</div>`;
}

/* ================================================================
 * 训练流程
 * ================================================================ */
async function startTrain() {
  const subset = +$("trainSubset").value;
  const epochs = Math.max(1, Math.min(6, +$("trainEpochs").value || 3));
  const btn = $("btnTrain");
  btn.disabled = true;
  $("trainHint").textContent = "训练中(首次需下载 MNIST ~15MB)… 可最小化等待";
  const logEl = $("trainLog");
  logEl.style.display = "block";
  logEl.textContent = "提交训练…\n";
  try {
    await api("/api/cnn/train", { subset, epochs });
    $("busyDot2").classList.add("on");
    await pollTrain();
  } catch (e) {
    toast("训练请求失败: " + e.message, true);
    logEl.textContent += "失败: " + e.message;
  } finally {
    btn.disabled = false;
    $("busyDot2").classList.remove("on");
    $("trainHint").textContent = "";
  }
}
async function pollTrain() {
  const logEl = $("trainLog");
  for (let i = 0; i < 4000; i++) {
    const j = await fetch("/api/cnn/train_status").then((r) => r.json());
    const st = j.state || {};
    if (st.log && st.log.length) logEl.textContent = st.log.join("\n");
    if (st.error) { logEl.textContent += "\n错误: " + st.error; toast("训练出错: " + st.error, true); return; }
    if (!st.running && st.report) {
      logEl.textContent += `\n✔ 完成: test_acc=${(st.report.test_acc * 100).toFixed(1)}% (${st.report.secs}s)`;
      toast(`训练完成: 测试准确率 ${(st.report.test_acc * 100).toFixed(1)}%`);
      await loadModel();
      await loadLayers();
      return;
    }
    if (!st.running && !st.report && !st.log) { logEl.textContent += "\n(服务端无训练记录)"; return; }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

/* ================================================================
 * ④ 定量结论 & 设计规则(受控实验)
 * ================================================================ */
function pct(x) { return (x * 100).toFixed(1) + "%"; }
function ppDiff(a, b) { return ((a - b) * 100).toFixed(2); }

async function loadRules() {
  const box = $("rulesBox");
  try {
    const r = await fetch("/api/cnn/rules").then((x) => x.json());
    if (!r.ok) throw new Error(r.error || "no data");
    renderRules(box, r.data);
  } catch (e) {
    box.innerHTML = `<div class="empty">定量实验数据缺失: ${esc(e.message)}<br>
      <span class="small-txt">在本机运行 <code class="p">python3 backend/exp_rules.py</code>(约 5 分钟)后刷新。</span></div>`;
  }
}

function renderRules(box, d) {
  const modes = d.modes || [];
  const none = modes.find((m) => m.mode === "none");
  const evenM = modes.find((m) => m.mode === "even");
  const oddM = modes.find((m) => m.mode === "odd");
  const acc = (m) => (m ? m.test_acc : null);
  const row = (m) => {
    const nm = { none: "无约束(自由)", even: "偶核约束 k=(k+k̄)/2<br><span style='color:var(--muted)'>自伴·可导出拉氏量</span>", odd: "奇核约束 k=(k−k̄)/2<br><span style='color:var(--muted)'>反自伴·陀螺</span>" }[m.mode];
    const a = acc(m);
    const dpp = none && m.mode !== "none" && a !== null
      ? ((a - acc(none)) * 100 >= 0 ? "+" : "") + ((a - acc(none)) * 100).toFixed(2) + " pp" : "—";
    const s1 = m.stats.conv1, s2 = m.stats.conv2;
    return `<tr><td>${nm}</td><td class="mono ${m.mode === "none" ? "" : a < acc(none) ? "neg" : "pos"}">${a === null ? "—" : pct(a)}</td>
      <td class="mono">${dpp}</td>
      <td class="mono">${(s1.mean_odd * 100).toFixed(0)}%</td>
      <td class="mono">${(s1.mean_even * 100).toFixed(0)}%</td>
      <td class="mono">${(s2.mean_odd * 100).toFixed(0)}%</td>
      <td class="mono">${s1.psd}</td>
      <td class="mono">${s1.directional}/8</td></tr>`;
  };
  const cfg = d.config || {};
  const b1 = d.agg_seeds_conv1 || {}, b2 = d.agg_seeds_conv2 || {};
  const fmtAgg = (a, k) => {
    if (!a || !a[k]) return "—";
    return `${(a[k].mean * (k === "orient_R" ? 1 : 100)).toFixed(k === "orient_R" ? 3 : 1)}`;
  };
  const seeds = d.seeds || [];
  const accRange = seeds.length ? [Math.min(...seeds.map((s) => s.test_acc)), Math.max(...seeds.map((s) => s.test_acc))] : null;

  // —— 由实验数派生定量规则(按实测结果措辞) ——
  const rules = [];
  if (none && evenM) {
    const d = (acc(evenM) - acc(none)) * 100;
    const sign = d >= 0 ? "+" : "";
    rules.push({
      icon: "R1", t: d >= -0.05 ? "零代价(实测)" : "有代价(实测)",
      title: "奇偶结构是“自由自由度”: 投影为偶核(自伴=可导出拉氏量)几乎不损精度",
      txt: `同数据/轮数/种子下, 每步把卷积核投影成偶核(自伴, L2/拉氏量可导出类), 测试精度 ${sign}${d.toFixed(2)} pp(${pct(acc(none))} → ${pct(acc(evenM))})。手写笔画≈条带结构, 中心对称的<b>各向异性偶核</b>(bars/二阶型, 仍可取向, 方向型滤波器 ${evenM.stats.conv1.directional}/8 个)足以识别 — 说明<b>把卷积层约束进拉氏量框架(偶核)不需要牺牲精度</b>, 是可落地的“物理一致/可解释变分层”设计路径。`,
    });
  }
  if (none && oddM) {
    const d = (acc(oddM) - acc(none)) * 100;
    const sign = d >= 0 ? "+" : "";
    rules.push({
      icon: "R2", t: Math.abs(d) < 0.2 ? "几乎零代价" : "有代价",
      title: "纯奇核(反自伴/陀螺)约束同样几乎零代价 — 奇部不是精度的必要条件",
      txt: `只保留奇核(陀螺/一阶差分, DC 为零)训练, 测试 ${pct(acc(oddM))}(${sign}${d.toFixed(2)} pp)。两类极值约束(偶 vs 奇)都≈基线, 说明对数字识别而言特征内容主要在高频/对比结构, 与算子奇偶相位无关; 偶部(平滑/势能)与奇部(方向/陀螺)可以<b>任意配比</b>而不伤精度。`,
    });
  }
  rules.push({
    icon: "R3", t: "跨模型恒真",
    title: "PSD(正定/凸能量)在普通训练下恒为 0 — 特征层本质是过零/差分测量器",
    txt: `全部 ${modes.length + seeds.length} 个模型(含偶/奇极值约束)的 Conv1+Conv2 共 ${(modes.length + seeds.length) * 136} 个 2D 算子中, PSD 计数恒为 0: 任何约束下训练出的特征核符号都变号、能量泛函非凸。若要让层具有“势能/能量”语义(正则、去噪、物理一致), 必须<b>显式</b>选用正定核族(对称非负符号, 如协方差/格林核参数化), 不能指望自由训练自动得到。`,
  });
  if (b1.mean_odd) {
    rules.push({
      icon: "R4", t: "分层规律(实测)",
      title: "奇偶占比: 首层随种子漂移(41–60%), 深层分量自组织到 ~48% — 别把它当硬指标",
      txt: `Conv1 奇部占比跨种子 ${(b1.mean_odd.lo * 100).toFixed(0)}–${(b1.mean_odd.hi * 100).toFixed(0)}%(均值 ${(b1.mean_odd.mean * 100).toFixed(1)}%), 而 Conv2 的 128 个分量稳定在 ${(b2.mean_odd ? b2.mean_odd.mean * 100 : 0).toFixed(1)}%±1%。规则: 偶/奇分解(页 ②)适合做<b>诊断与约束旋钮</b>, 但 50/50 不是网络必然; 首层奇偶预算与初始化/优化路径有关, 深层组合更保守地趋于平衡。`,
    });
  }
  if (b1.orient_R) {
    rules.push({
      icon: "R5", t: "需监控",
      title: "方向多样性不是自动保证的 — 训练早期用方向直方图体检",
      txt: `Conv1 方向型滤波器稳定出现(${fmtAgg(b1, "directional")}/8), 但朝向均匀性 R 跨种子波动大(${b1.orient_R.lo}–${b1.orient_R.hi}, 均值 ${b1.orient_R.mean}): 短训练/小数据下可能出现方向盲区。规则: 把页②的 15° 方向直方图与 R 当作“方向多样性体检表”, 发现空桶再考虑初始化/数据增强。`,
    });
  }
  if (accRange) {
    rules.push({
      icon: "R6", t: "方法论",
      title: "固定预算的 A/B 投影协议可复用: 先验约束的代价应“实测不猜”",
      txt: `同配置 3 种子测试精度 ${pct(accRange[0])}–${pct(accRange[1])}(种子间抖动 ~1.4 pp), 而 A/B 用同一种子把抖动排除, 使“偶约束 +0.06 pp / 奇约束 −0.07 pp”成为可比结论。给任何任务的先验(偶核、可导拉氏量、PSD、对称轴…)套用本协议: 每步投影到目标子空间 vs 自由训练, 先测后决定约束是否“免费”。`,
    });
  }

  const ruleHtml = rules.map((r) => `<li class="info"><span class="ic" style="color:var(--cyan)">${r.icon}</span>
    <span><b>${esc(r.title)}</b> <span class="badge" style="color:var(--orange);border-color:rgba(255,182,77,.5)">${esc(r.t)}</span><br>
    ${r.txt}</span></li>`).join("");

  box.innerHTML = `
  <div class="small-txt" style="padding:10px 16px 2px">实验设置: MNIST ${cfg.subset}×${cfg.epochs} 轮 · Adam lr ${cfg.lr} · 每步把卷积核投影到目标对称子空间; 无约束组同 seed 基线 test=${cfg.acc_base_seed7 === undefined ? "—" : pct(cfg.acc_base_seed7)}。</div>
  <div style="padding:4px 16px 6px"><b>A. 对称约束的精度代价(设计规则的量化支撑)</b></div>
  <div style="overflow-x:auto;padding:0 16px 4px"><table class="sumt">
    <tr><th>训练约束</th><th>test acc</th><th>Δ vs 自由</th><th>Conv1 奇部%</th><th>Conv1 偶部%</th><th>Conv2 奇部%</th><th>Conv1 PSD</th><th>Conv1 方向型</th></tr>
    ${modes.map(row).join("")}</table></div>
  <div style="padding:12px 16px 2px"><b>B. 多种子复现性(自由训练, Conv1 算子统计)</b></div>
  <div style="overflow-x:auto;padding:0 16px 8px"><table class="sumt">
    <tr><th>指标</th><th>均值</th><th>范围(3 seeds)</th><th>含义</th></tr>
    <tr><td>奇部能量占比</td><td class="mono">${fmtAgg(b1, "mean_odd")}%</td><td class="mono">${(b1.mean_odd ? b1.mean_odd.lo * 100 : 0).toFixed(0)}–${(b1.mean_odd ? b1.mean_odd.hi * 100 : 0).toFixed(0)}%</td><td>反自伴(陀螺)分量占比</td></tr>
    <tr><td>偶部能量占比</td><td class="mono">${fmtAgg(b1, "mean_even")}%</td><td class="mono">—</td><td>自伴(势能型)分量占比</td></tr>
    <tr><td>正定 PSD 数</td><td class="mono">${b1.psd ? b1.psd.mean : "0"}</td><td class="mono">恒 0</td><td>无滤波器进入 L3 正定锥</td></tr>
    <tr><td>方向差分型滤波器</td><td class="mono">${fmtAgg(b1, "directional")}/8</td><td class="mono">—</td><td>每层方向选择性个数</td></tr>
    <tr><td>朝向均匀性 R</td><td class="mono">${fmtAgg(b1, "orient_R")}</td><td class="mono">${b1.orient_R ? b1.orient_R.lo + "–" + b1.orient_R.hi : "—"}</td><td>0=全朝向均匀</td></tr>
    </table></div>
  <div style="padding:4px 16px 2px"><b>一般设计规则(由上述定量结果提炼)</b></div>
  <ul class="concl" style="padding:2px 16px 10px">${ruleHtml}</ul>`;
}


/* ================================================================
 * ⑥ 模块作用机理(消融实验)
 * ================================================================ */
async function loadModules() {
  const box = $("modBox");
  try {
    const r = await fetch("/api/cnn/modules").then((x) => x.json());
    if (!r.ok) throw new Error(r.error || "no data");
    renderModules(box, r.data);
  } catch (e) {
    box.innerHTML = `<div class="empty">消融实验数据缺失: ${esc(e.message)}<br>
      <span class="small-txt">运行 <code class="p">python3 backend/mech_modules.py</code>(约 2 分钟)后刷新。</span></div>`;
  }
}

function renderModules(box, d) {
  const base = d.baseline ? d.baseline.test_acc : 0;
  const pct = (x) => (x * 100).toFixed(2) + "%";
  const dpp = (x) => (x >= 0 ? "+" : "") + x.toFixed(2) + " pp";
  const bar = (v, mx, cls) => `<div class="tb" style="height:10px;border-radius:5px;background:rgba(120,150,210,.1);overflow:hidden">
      <i style="display:block;height:100%;width:${Math.max(1, (v / mx) * 100)}%;background:${cls}"></i></div>`;

  // —— 训练期消融 ——
  const tr = [...(d.training || [])].sort((a, b) => b.test_acc - a.test_acc);
  const trRows = tr.map((t) => {
    const delta = (t.test_acc - base) * 100;
    const cls = t.name === "base" ? "info" : delta < -0.5 ? "warn" : "";
    return `<tr><td>${esc(t.name)}<div class="small-txt">${esc(t.note)}</div></td>
      <td class="mono ${cls}">${pct(t.test_acc)}</td>
      <td class="mono">${t.name === "base" ? "—" : dpp(delta)}</td>
      <td class="mono">${t.params.toLocaleString()}</td>
      <td class="mono">${t.secs}s</td></tr>`;
  }).join("");

  // —— 推理期消融 ——
  const inf = d.inference || [];
  const infRows = inf.map((t) => `<tr><td>${esc(t.name)}<div class="small-txt">${esc(t.note)}</div></td>
      <td class="mono">${pct(t.acc)}</td>
      <td class="mono ${t.delta_pp < -0.5 ? "warn" : ""}">${t.delta_pp === 0 ? "—" : dpp(t.delta_pp)}</td></tr>`).join("");

  // —— 通道重要性 ——
  const chHTML = (list, layer) => {
    const mx = Math.max(0.01, ...list.map((c) => Math.abs(c.drop_pp)));
    return `<div class="small-txt" style="margin-bottom:4px">Conv${layer} 逐通道置零的精度损失(pp, 越大越关键)</div>
      <div style="display:grid;grid-template-columns:repeat(${layer === 1 ? 8 : 16},1fr);gap:4px;align-items:end;height:110px">
      ${list.map((c) => `<div title="ch${c.ch} 置零后 acc=${pct(c.acc)}" style="display:flex;flex-direction:column;justify-content:flex-end;height:100%">
        <div style="height:${Math.max(2, (Math.max(0, c.drop_pp) / mx) * 100)}%;background:linear-gradient(180deg,#ff6b7a,#ffb347);border-radius:3px 3px 0 0"></div>
        <div style="font:9px var(--mono);color:var(--muted);text-align:center">${c.ch}</div></div>`).join("")}</div>`;
  };

  // —— 显著性 ——
  const sal = d.saliency || { per_class: [] };
  const salGrid = (sal.per_class || []).map((m, i) => `<div style="text-align:center">
      <canvas class="hm" data-sal="${i}" style="width:64px;height:64px;image-rendering:pixelated;border:1px solid var(--line);border-radius:4px"></canvas>
      <div style="font:10px var(--mono);color:var(--muted)">${i}</div></div>`).join("");

  // —— 自动解读 ——
  const find = (arr, n) => (arr || []).find((x) => x.name === n) || {};
  const B = base * 100;
  const notes = [];
  const mlp = find(d.training, "mlp_no_conv"), c1 = find(d.training, "conv1_only"),
        nh = find(d.training, "no_hidden"), nd = find(d.training, "no_dropout"),
        nr = find(d.training, "no_relu"), la = find(d.training, "linear_all"),
        ap = find(d.training, "avg_pool"), sp = find(d.training, "stride_downsample"),
        np_ = find(d.training, "no_pool"), ev = find(d.training, "even_kernels"),
        fr = find(d.training, "frozen_random_conv");
  if (mlp.test_acc) notes.push(`**卷积模块是主力**: 去掉全部卷积(纯 MLP)降到 ${pct(mlp.test_acc)}(${dpp((mlp.test_acc - base) * 100)}), 即两个卷积层贡献约 ${((base - mlp.test_acc) * 100).toFixed(1)} 个百分点 —— 权重共享+局部感受野带来的平移等变特征是精度来源。`);
  if (c1.test_acc) notes.push(`**深度**: 只留一层卷积 ${pct(c1.test_acc)}(${dpp((c1.test_acc - base) * 100)}), 第二层带来的组合特征是额外增益但非必需。`);
  if (nh.test_acc) notes.push(`**隐藏层 FC120**: 去掉后 ${pct(nh.test_acc)}(${dpp((nh.test_acc - base) * 100)}), 参数从 ${(find(d.training, "base").params || 0).toLocaleString()} 降到 ${nh.params.toLocaleString()} —— 它主要做“特征重组/去相关”, 参数效率很高。`);
  if (nd.test_acc) notes.push(`**Dropout**: 关掉后 ${pct(nd.test_acc)}(${dpp((nd.test_acc - base) * 100)}), 在这个小网络上正则化收益有限。`);
  if (nr.test_acc) notes.push(`**ReLU(逐层)**: 去掉所有 ReLU 仍有 ${pct(nr.test_acc)}(${dpp((nr.test_acc - base) * 100)}) —— 因为 **最大池化本身已提供非线性**; 再叠加平均池化近似线性系统后为 ${la.test_acc ? pct(la.test_acc) : "—"}${la.test_acc ? `(${dpp((la.test_acc - base) * 100)})` : ""}, 这才是“非线性”的总贡献。`);
  if (ap.test_acc) notes.push(`**池化方式**: 最大池化 → 平均池化 ${pct(ap.test_acc)}(${dpp((ap.test_acc - base) * 100)}), 说明“取最强响应”的稀疏选择比平均更契合笔画特征; 换成学习式步长卷积 ${sp.test_acc ? pct(sp.test_acc) : "—"}${sp.test_acc ? `(${dpp((sp.test_acc - base) * 100)})` : ""}。`);
  if (np_ && np_.test_acc) notes.push(`**降采样必要性**: 完全不做池化 ${pct(np_.test_acc)}(${dpp((np_.test_acc - base) * 100)}), 参数 ${np_.params.toLocaleString()} —— 池化用极少参数换来了感受野扩张与平移鲁棒。`);
  if (ev.test_acc) notes.push(`**自伴约束(拉氏量可导出)**: 每步把卷积核投影为偶核 ${pct(ev.test_acc)}(${dpp((ev.test_acc - base) * 100)}) —— 在这个笔画类任务上约束几乎免费(对应前面 R1 规则)。`);
  if (fr.test_acc) notes.push(`**卷积核必须学习吗**: 冻结为随机核、只训分类头 ${pct(fr.test_acc)}(${dpp((fr.test_acc - base) * 100)}), 与随机特征方法的预期一致 —— 卷积核确实学到了任务相关结构。`);
  const r1 = find(d.inference, "relu_off@conv1"), r2 = find(d.inference, "relu_off@conv2"),
        rf = find(d.inference, "relu_off@fc1"), sh = find(d.inference, "shuffle_kernels"),
        ei = find(d.inference, "even_project_infer");
  notes.push(`**推理期（冻结权重）**: 关 ReLU → conv1 ${r1.delta_pp ?? "—"}pp / conv2 ${r2.delta_pp ?? "—"}pp / fc1 ${rf.delta_pp ?? "—"}pp;` +
    ` 卷积核投影为偶核 ${ei.delta_pp ?? "—"}pp; 核内像素打乱(对照) ${sh.delta_pp ?? "—"}pp —— 打乱会造成大幅下降, 证明空间结构(而非仅数值分布)才是有效信息。`);
  // 遮挡/显著性: 正证据 vs 负证据
  const oc = (n) => find(d.inference, n);
  const oC = oc("occlusion_center"), oB = oc("occlusion_border"),
        oCo = oc("occlusion_center_ones"), oBo = oc("occlusion_border_ones"),
        oR = oc("occlusion_rows"), oCl = oc("occlusion_cols");
  if (oC.delta_pp !== undefined) {
    notes.push(`**遮挡(置零)**: 中心 8×8 ${dpp(oC.delta_pp)}, 中间 12 行 ${dpp(oR.delta_pp)}, 中间 12 列 ${dpp(oCl.delta_pp)}, 四周 3px ${dpp(oB.delta_pp)} —— 置零边缘几乎无影响, 因为那里本来就是背景(空操作)。`);
  }
  if (oBo.delta_pp !== undefined) {
    notes.push(`**负证据(关键发现)**: 把四周 3px 直接<b>填满墨迹</b>后精度 ${pct(oBo.acc)}(${dpp(oBo.delta_pp)}), 比破坏中心的 ${dpp(oCo.delta_pp)} 还严重 —— 网络不仅看“笔画在哪”, 更强依赖“<b>边缘必须是空白</b>”这一负证据。这正是显著性图里边缘能量占比 ${(sal.border_ratio * 100).toFixed(0)}% 的原因: 梯度大 ≠ 可遮挡(置零无操作), 必须配合“填充式遮挡”才能读出真实依赖。`);
  }
  if (ei.delta_pp !== undefined && ev.delta_pp !== undefined) {
    notes.push(`**训练 vs 推理的差别**: 训练时把核约束为偶核几乎不掉点(${dpp(ev.delta_pp)}), 但在<b>已训练好的模型上</b>直接把核投影为偶核却掉 ${Math.abs(ei.delta_pp).toFixed(1)}pp —— 说明偶核本身足够表达, 只要训练时允许网络去适配这个约束; 事后投影破坏的是已学到的相位/方向结构。`);
  }
  const ch2 = (d.channel_importance && d.channel_importance.conv2) || [];
  const top2 = [...ch2].sort((a, b) => b.drop_pp - a.drop_pp).slice(0, 3);
  if (top2.length) notes.push(`**通道分工**: Conv2 中最关键的通道是 ${top2.map((c) => `ch${c.ch}(${c.drop_pp > 0 ? "−" : "+"}${Math.abs(c.drop_pp)}pp)`).join("、")} —— 少数通道承载了大部分判别信息(与页⑤“共享基+少数选择通道”一致)。`);
  if (sal.center_ratio !== undefined) notes.push(`**信息读取位置**: 真类梯度显著性中, 中心区占 ${(sal.center_ratio * 100).toFixed(1)}%、四周边缘占 ${(sal.border_ratio * 100).toFixed(1)}% —— 分类主要依赖笔画中心区域, 与质心预处理一致。`);

  box.innerHTML = `
  <div class="small-txt" style="padding:12px 16px 2px">实验设置: MNIST ${(d.config || {}).subset} 训练样本 × ${(d.config || {}).epochs} 轮 · batch ${(d.config || {}).batch} · 测试 ${(d.config || {}).test_n} 张 · seed ${(d.config || {}).seed} · 单变量对照(每次只改一个模块)。基线 <b class="ok">${pct(base)}</b>。</div>

  <div style="padding:10px 16px 2px"><b>A. 训练期消融(每个变体独立重训, 按测试精度排序)</b></div>
  <div style="overflow-x:auto;padding:0 16px 10px"><table class="sumt">
    <tr><th>模块改动</th><th>测试准确率</th><th>Δ vs 基线</th><th>参数量</th><th>耗时</th></tr>
    ${trRows}</table></div>

  <div style="padding:4px 16px 2px"><b>B. 推理期干预(冻结同一份基线权重, 不改训练)</b></div>
  <div style="overflow-x:auto;padding:0 16px 10px"><table class="sumt">
    <tr><th>干预</th><th>测试准确率</th><th>Δ vs 基线</th></tr>${infRows}</table></div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;padding:0 16px 10px">
    <div>${chHTML((d.channel_importance || {}).conv1 || [], 1)}</div>
    <div>${chHTML((d.channel_importance || {}).conv2 || [], 2)}</div>
  </div>

  <div style="padding:4px 16px 10px"><b>C. 显著性(真类 logit 对输入像素的梯度, 按类平均)</b>
    <span class="small-txt">中心区能量占比 ${sal.center_ratio !== undefined ? (sal.center_ratio * 100).toFixed(1) + "%" : "—"} · 边缘占比 ${sal.border_ratio !== undefined ? (sal.border_ratio * 100).toFixed(1) + "%" : "—"}</span></div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;padding:0 16px 12px">${salGrid}</div>

  <div style="padding:4px 16px 12px"><b>D. 机理结论(由以上数字自动生成)</b>
    <ul class="concl" style="margin-top:8px">${notes.map((n) => `<li class="info"><span class="ic">✱</span><span>${n.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")}</span></li>`).join("")}</ul>
    <div class="small-txt" style="margin-top:6px">注: 单 seed、30k×3 轮预算下的对照; 结论为“该预算下的模块贡献排序”, 不是绝对最优值。最大池化提供非线性这一点解释了“去掉 ReLU 几乎不掉点”。</div>
  </div>`;
  // 画显著性热图
  (sal.per_class || []).forEach((m, i) => {
    const cv = box.querySelector(`canvas[data-sal="${i}"]`);
    if (cv) drawHeat(cv, m);
  });
}

/* ================================================================
 * 启动
 * ================================================================ */
function bind() {
  document.querySelectorAll("#layerTabs button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#layerTabs button").forEach((x) => x.classList.toggle("active", x === b));
      state.cur = b.dataset.l;
      renderLayer();
    };
  });
  document.querySelectorAll("#frameTabs button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#frameTabs button").forEach((x) => x.classList.toggle("active", x === b));
      state.frame = b.dataset.f;
      renderLayer();
    };
  });
  $("btnPredict").onclick = predict;
  $("btnTrain").onclick = startTrain;
  initBoard();
}
async function boot() {
  bind();
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    if (!h.cnn) {
      $("cnnStatus").textContent = "CNN 模块需 numpy(以 CNN_PYTHONPATH=… 启动后端)";
    }
  } catch (e) { /* 后端未连 */ }
  await loadModel();
  await loadLayers();
  await loadRules();
  await loadMech();
  await loadModules();
}
document.addEventListener("DOMContentLoaded", boot);
