/* ============================================================================
 * CNN 手写数字识别演示台
 *  · 纯 JS 前向引擎(权重来自训练脚本导出的 JSON, 与 PyTorch/numpy 权重同布局)
 *  · 逐层演示: 卷积滑窗 → ReLU → 池化 → 再卷积 → 再池化 → 展平 → FC120 → Dropout → Softmax
 *  · 3D 特征图堆栈(Canvas2D 仿射投影, 可旋转缩放)
 *  · KaTeX 公式; 可选 TF.js 交叉验证; 拉格朗日算子分析面板
 * ========================================================================== */
"use strict";

/* ---------------------------------------------------------------- 基础工具 */
const $ = (s) => document.querySelector(s);
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "—"
  : (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)) ? x.toExponential(2) : (+x.toFixed(d)).toString();
function toast(msg, err) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast show" + (err ? " err" : "");
  clearTimeout(t._t); t._t = setTimeout(() => (t.className = "toast"), 3200);
}
function renderTex(el, tex, cap) {
  const K = window.katex;
  const html = (K && K.renderToString)
    ? (() => { try { return K.renderToString(tex, { throwOnError: false, displayMode: true }); } catch (e) { return null; } })()
    : null;
  el.innerHTML = (cap ? `<div class="cap">${cap}</div>` : "") +
    (html || `<code>${tex.replace(/\\/g, "").replace(/[{}]/g, "")}</code>`);
}

/* ---------------------------------------------------------------- 热图绘制 */
function heat(canvas, mat, mode = "div", labels = null) {
  /* mat: 2D 数组 */
  const R = mat.length, C = mat[0].length;
  canvas.width = C; canvas.height = R;
  canvas.style.imageRendering = "pixelated";
  const ctx = canvas.getContext("2d");
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const v = mat[i][j]; if (v < mn) mn = v; if (v > mx) mx = v; }
  const span = Math.max(Math.abs(mn), Math.abs(mx), 1e-9);
  const img = ctx.createImageData(C, R);
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const v = mat[i][j]; let r, g, b;
    if (mode === "gray") { const t = mn === mx ? 0 : (v - mn) / (mx - mn); r = g = b = Math.round(255 * t); }
    else {
      const t = Math.max(-1, Math.min(1, v / span));
      if (t < 0) { r = Math.round(40 + 40 * t); g = Math.round(90 + 70 * t); b = 255; }
      else { r = 255; g = Math.round(110 + 145 * (1 - t)); b = Math.round(60 * (1 - t) + 30); }
    }
    const p = (i * C + j) * 4; img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return { mn, mx };
}

/* ---------------------------------------------------------------- 前向引擎 */
const eng = {
  W: null, meta: null, trace: null, input: null, modelName: "",
};

function toF32(nested) { return nested instanceof Float32Array ? nested : Float32Array.from(nested.flat(Infinity)); }

function conv2d(inp, C, H, W, ker, O, K, bias) {
  const out = new Float32Array(O * H * W);
  const pad = (K - 1) >> 1;
  for (let o = 0; o < O; o++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let s = bias ? bias[o] : 0;
        for (let c = 0; c < C; c++) {
          const ib = c * H * W, kb = (o * C + c) * K * K;
          for (let a = 0; a < K; a++) {
            const yy = y + a - pad; if (yy < 0 || yy >= H) continue;
            for (let b = 0; b < K; b++) {
              const xx = x + b - pad; if (xx < 0 || xx >= W) continue;
              s += ker[kb + a * K + b] * inp[ib + yy * W + xx];
            }
          }
        }
        out[o * H * W + y * W + x] = s;
      }
    }
  }
  return out;
}
function relu(a) { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] > 0 ? a[i] : 0; return o; }
function maxpool(a, C, H, W) {
  const OH = H >> 1, OW = W >> 1, o = new Float32Array(C * OH * OW);
  for (let c = 0; c < C; c++) for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) {
    let m = -Infinity;
    for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
      const v = a[c * H * W + (2 * y + dy) * W + (2 * x + dx)]; if (v > m) m = v;
    }
    o[c * OH * OW + y * OW + x] = m;
  }
  return o;
}
function dense(x, W, b, O, I) {
  const o = new Float32Array(O);
  for (let j = 0; j < O; j++) { let s = b[j]; for (let i = 0; i < I; i++) s += W[j * I + i] * x[i]; o[j] = s; }
  return o;
}
function softmax(z) {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
function slice(a, off, len) { return a.slice(off, off + len); }
function as2d(a, off, H, W) { const out = []; for (let y = 0; y < H; y++) { const row = []; for (let x = 0; x < W; x++) row.push(a[off + y * W + x]); out.push(row); } return out; }

function forward(input28) {
  const W = eng.W;
  const x = toF32(input28);
  const c1pre = conv2d(x, 1, 28, 28, W.c1w, 8, 5, W.c1b);
  const c1 = relu(c1pre);
  const p1 = maxpool(c1, 8, 28, 28);
  const c2pre = conv2d(p1, 8, 14, 14, W.c2w, 16, 5, W.c2b);
  const c2 = relu(c2pre);
  const p2 = maxpool(c2, 16, 14, 14);
  const flat = p2;                                  // 16*7*7 = 784, 与训练同序(C,H,W)
  const f1pre = dense(flat, W.f1w, W.f1b, 120, 784);
  const f1 = relu(f1pre);
  const logits = dense(f1, W.f2w, W.f2b, 10, 120);
  const probs = softmax(Array.from(logits));
  eng.trace = { x, c1pre, c1, p1, c2pre, c2, p2, flat, f1pre, f1, logits, probs };
  return eng.trace;
}

/* ---------------------------------------------------------------- 画板输入 */
const pad = { canvas: null, ctx: null, drawing: false, last: null, size: 336, cell: 12 };
function initPad() {
  pad.canvas = $("#pad"); pad.ctx = pad.canvas.getContext("2d");
  pad.ctx.fillStyle = "#000"; pad.ctx.fillRect(0, 0, pad.size, pad.size);
  const pos = (e) => {
    const r = pad.canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (pad.size / r.width), (e.clientY - r.top) * (pad.size / r.height)];
  };
  pad.canvas.addEventListener("pointerdown", (e) => {
    pad.drawing = true; pad.canvas.setPointerCapture(e.pointerId);
    const [x, y] = pos(e); pad.last = [x, y];
    drawSeg(x, y, x + 0.01, y + 0.01);
  });
  pad.canvas.addEventListener("pointermove", (e) => {
    if (!pad.drawing) return; const [x, y] = pos(e); drawSeg(pad.last[0], pad.last[1], x, y); pad.last = [x, y];
  });
  pad.canvas.addEventListener("pointerup", () => (pad.drawing = false));
  pad.canvas.addEventListener("pointerleave", () => (pad.drawing = false));
}
function drawSeg(x0, y0, x1, y1) {
  const c = pad.ctx;
  c.strokeStyle = "#fff"; c.lineWidth = 22; c.lineCap = "round"; c.lineJoin = "round";
  c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
}
function clearPad() {
  pad.ctx.fillStyle = "#000"; pad.ctx.fillRect(0, 0, pad.size, pad.size);
  $("#trueLab").textContent = "—"; $("#predLab").textContent = "—"; $("#confVal").textContent = "—";
  $("#inputMeta").textContent = "待写入"; $("#thumb").getContext("2d").clearRect(0, 0, 112, 112);
  setStagesEnabled(false);
}
function rasterize() {
  const img = pad.ctx.getImageData(0, 0, pad.size, pad.size).data;
  const cell = pad.cell, raw = [];
  for (let y = 0; y < 28; y++) {
    const row = [];
    for (let x = 0; x < 28; x++) {
      let s = 0;
      for (let dy = 0; dy < cell; dy++) for (let dx = 0; dx < cell; dx++) {
        s += img[((y * cell + dy) * pad.size + x * cell + dx) * 4];
      }
      row.push(s / (cell * cell * 255));
    }
    raw.push(row);
  }
  let mx = 0, my = 0, mass = 0;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) { const v = raw[y][x]; mx += x * v; my += y * v; mass += v; }
  let out = raw;
  if ($("#chkCenter").checked && mass > 0.4) {
    const dx = Math.round(13.5 - mx / mass), dy = Math.round(13.5 - my / mass);
    out = Array.from({ length: 28 }, () => Array(28).fill(0));
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const sx = x + dx, sy = y + dy;
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[sy][sx] = raw[y][x];
    }
  }
  return out;
}
function showThumb(m) { heat($("#thumb"), m, "gray"); }

/* ---------------------------------------------------------------- 模型载入 */
async function loadModel(name) {
  const r = await fetch(`/api/cnn/weights?model=${encodeURIComponent(name)}`).then((x) => x.json());
  if (!r.ok) throw new Error(r.error || "载入失败");
  const w = r.data.weights;
  eng.W = {
    c1w: toF32(w.c1w), c1b: Float32Array.from(w.c1b),
    c2w: toF32(w.c2w), c2b: Float32Array.from(w.c2b),
    f1w: toF32(w.f1w), f1b: Float32Array.from(w.f1b),
    f2w: toF32(w.f2w), f2b: Float32Array.from(w.f2b),
  };
  eng.meta = r.data.report || {};
  eng.modelName = r.model;
  $("#modelName").textContent = r.model;
  $("#ledModel").className = "led on";
  const acc = eng.meta.test_acc;
  $("#accTag").textContent = acc ? `test_acc=${(acc * 100).toFixed(2)}% · ${eng.meta.framework || ""}` : "";
  renderArch(r.data.arch);
  return r.data;
}
function renderArch(arch) {
  const layers = (arch && arch.layers) || [];
  $("#archTable").innerHTML = layers.map((l) => `<tr>
    <td>${l.no}</td>
    <td><span class="t">${l.type}</span>${l.in !== undefined ? ` ${l.in}→${l.out ?? ""}` : ""}${l.k ? ` k=${l.k}` : ""}${l.p ? ` p=${l.p}` : ""}</td>
    <td class="p">${l.out_size || ""}${l.params ? `<br>${l.params}` : ""}</td></tr>`).join("") ||
    `<tr><td colspan="3">结构信息不可用(运行训练脚本后重试)</td></tr>`;
}

/* ---------------------------------------------------------------- 阶段与公式 */
const STAGES = [
  { id: "input", name: "输入 28×28" },
  { id: "conv1", name: "卷积1 · 8×5×5" },
  { id: "pool1", name: "ReLU + 池化1" },
  { id: "conv2", name: "卷积2 · 16×5×5×8" },
  { id: "pool2", name: "ReLU + 池化2" },
  { id: "flat", name: "展平 784" },
  { id: "fc1", name: "FC120 + ReLU + Dropout" },
  { id: "softmax", name: "FC10 + Softmax" },
];
let stage = 0, playing = false, playTimer = null;

const FORMULAS = {
  input: ["x\\in[0,1]^{28\\times28}", "输入是单通道灰度图: 每个像素一个 0–1 实数, 不再展平(保留空间结构)"],
  conv1: ["(K*u)_{i,j}=\\sum_{a=0}^{4}\\sum_{b=0}^{4}K_{a,b}\\,u_{\\,i+a-2,\\;j+b-2}+\\beta",
    "5×5 卷积核在 28×28 灰度图上滑动; 每个位置做 25 次乘加再求和(+偏置 β) → 输出特征图同尺寸 28×28。8 个不同核 → 8 张特征图。"],
  pool1: ["\\mathrm{ReLU}(z)=\\max(0,z)\\;\\;\\Longrightarrow\\;\\;\\mathrm{MaxPool}_{2\\times2}(h)_{i,j}=\\max_{0\\le a,b<2}h_{\\,2i+a,\\,2j+b}",
    "先 ReLU 把负响应压成 0(保留“有该特征”的正证据), 再用 2×2 窗口取最大, 尺寸减半 → 14×14×8, 获得平移鲁棒性。"],
  conv2: ["h^{(k)}=\\mathrm{ReLU}\\!\\Big(\\sum_{c=1}^{8}K^{(k,c)}*p^{(c)}+\\beta_k\\Big),\\quad k=1\\dots16",
    "16 个 5×5×8 的三维卷积核: 对 8 个输入通道分别卷积后求和(再加偏置), 得到 14×14×16 特征图。"],
  pool2: ["p^{(k)}_{i,j}=\\max_{0\\le a,b<2}h^{(k)}_{\\,2i+a,\\,2j+b}\\;\\Longrightarrow\\;7\\times7\\times16",
    "同样 ReLU + 2×2 最大池化: 再降采样一半, 特征变得“更抽象、更全局”。"],
  flat: ["v=\\mathrm{vec}(P)\\in\\mathbb{R}^{7\\cdot7\\cdot16}=\\mathbb{R}^{784}",
    "把 7×7×16 张量拉直成 784 维向量, 交给全连接层(空间结构在此结束, 只剩“模式—强度”)。"],
  fc1: ["a=\\mathrm{ReLU}\\!\\big(W^{(1)}v+b^{(1)}\\big),\\qquad \\tilde a=\\frac{a\\odot m}{1-p},\\;\\;m\\sim\\mathrm{Bernoulli}(1-p)",
    "784→120 全连接 + ReLU(稀疏门: 负值被关掉)。Dropout 训练时按概率 p 随机丢弃神经元并放大存活项; 推理时关闭(等价于集成)。"],
  softmax: ["z=W^{(2)}a+b^{(2)},\\qquad \\mathrm{softmax}(z)_i=\\frac{e^{z_i}}{\\sum_{j=1}^{10}e^{z_j}},\\qquad \\sum_{i=1}^{10}p_i=1",
    "120→10 得到 10 个 logit, Softmax 归一化成概率分布: 10 类相加正好为 1, 最大者即预测数字。"],
};

function renderFormula() {
  const [tex, cap] = FORMULAS[STAGES[stage].id];
  renderTex($("#formulaBox"), tex, cap);
}

/* ---------------------------------------------------------------- 3D 渲染 */
const cam = { rx: -0.32, ry: 0.62, zoom: 1 };
function project(x, y, z, W, H) {
  const cy = Math.cos(cam.ry), sy = Math.sin(cam.ry);
  const x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
  const cx = Math.cos(cam.rx), sx = Math.sin(cam.rx);
  const y1 = y * cx - z1 * sx, z2 = y * sx + z1 * cx;
  const d = 1400 / (1400 + z2);
  return [W / 2 + x1 * cam.zoom * d, H / 2 + y1 * cam.zoom * d, z2];
}
function mapCanvas(data, H, W, scale = 1) {
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const ctx = c.getContext("2d");
  let mx = 1e-9; for (let i = 0; i < data.length; i++) mx = Math.max(mx, Math.abs(data[i]));
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    const v = data[i * W + j] / mx;
    const t = Math.max(-1, Math.min(1, v));
    let r, g, b;
    if (t < 0) { r = 40 + 40 * t; g = 90 + 70 * t; b = 255; }
    else { r = 255; g = 110 + 145 * (1 - t); b = 60 * (1 - t) + 30; }
    const p = (i * W + j) * 4; img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 245;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
function drawStack(view, maps, H, W, opts = {}) {
  /* maps: [{data,label}] → 沿 z 轴排列的 3D 卡片堆栈 */
  const ctx = view.getContext("2d");
  const { W: VW, H: VH } = view;
  const scale = opts.scale || 3.2;
  const gap = opts.gap || 46;
  const n = maps.length;
  const items = maps.map((m, i) => {
    const z = (i - (n - 1) / 2) * gap;
    const c = project(0, 0, z, VW, VH);
    return { ...m, z, cz: c[2], canvas: mapCanvas(m.data, H, W) };
  }).sort((a, b) => b.cz - a.cz);   // 远的先画
  for (const it of items) {
    const s = (W / 2) * scale;
    const p00 = project(-s, -s, it.z, VW, VH), p10 = project(s, -s, it.z, VW, VH), p01 = project(-s, s, it.z, VW, VH);
    ctx.save();
    ctx.setTransform((p10[0] - p00[0]) / W, (p10[1] - p00[1]) / W, (p01[0] - p00[0]) / H, (p01[1] - p00[1]) / H, p00[0], p00[1]);
    ctx.globalAlpha = 0.96;
    ctx.drawImage(it.canvas, 0, 0);
    ctx.strokeStyle = it.hl ? "#3ddc97" : "rgba(120,160,220,.5)";
    ctx.lineWidth = it.hl ? 2.5 : 1;
    ctx.strokeRect(0, 0, W, H);
    ctx.restore();
    const lc = project(s, s, it.z, VW, VH);
    ctx.fillStyle = it.hl ? "#3ddc97" : "#7d90a8";
    ctx.font = "12px Consolas, monospace";
    ctx.fillText(it.label || "", lc[0] + 4, lc[1] + 14);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

/* ---------------------------------------------------------------- 各阶段绘制 */
const anim = { t: 0, timer: null };

function currentSelection() {
  /* 自动选“响应最强”的滤波器/通道, 让演示更有代表性 */
  const tr = eng.trace; if (!tr) return { f: 0, o: 0, c: 0 };
  const peak = (arr, C, H, W) => { const out = []; for (let c = 0; c < C; c++) { let m = 0; for (let i = 0; i < H * W; i++) m = Math.max(m, Math.abs(arr[c * H * W + i])); out.push(m); } return out; };
  const s1 = peak(tr.c1, 8, 28, 28);
  const s2 = peak(tr.c2, 16, 14, 14);
  const f = s1.indexOf(Math.max(...s1));
  const o = s2.indexOf(Math.max(...s2));
  const c = (eng.W && eng.W.c2w) ? (() => { let best = 0, bv = -1; for (let i = 0; i < 8; i++) { let s = 0; for (let k = 0; k < 25; k++) s += Math.abs(eng.W.c2w[(o * 8 + i) * 25 + k]); if (s > bv) { bv = s; best = i; } } return best; })() : 0;
  return { f, o, c };
}

function draw() {
  const view = $("#view"), ctx = view.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#070b10"; ctx.fillRect(0, 0, view.width, view.height);
  const tr = eng.trace;
  if (!tr) {
    ctx.fillStyle = "#42566e"; ctx.font = "15px Consolas, monospace";
    ctx.fillText("等待输入: 在左侧面板手写数字, 点击「确认演示」", 40, view.height / 2);
    return;
  }
  const threeD = $("#chk3d").checked;
  const sel = currentSelection();
  const id = STAGES[stage].id;
  const mapsFor = (arr, C, H, W, labels) => Array.from({ length: C }, (_, c) => ({ data: arr.slice(c * H * W, (c + 1) * H * W), label: labels ? labels(c) : `#${c}` }));

  if (id === "input") {
    drawMatrix(ctx, tr.x, 28, 28, 40, 60, 320, "输入 x (28×28, 灰度)");
    drawMatrix(ctx, tr.x, 28, 28, 400, 60, 320, "同一矩阵 = 像素强度");
    ctx.fillStyle = "#9fb3cc"; ctx.font = "13px Consolas, monospace";
    ctx.fillText(`非零像素 ${Array.from(tr.x).filter((v) => v > 0.05).length} / 784`, 40, 430);
    ctx.fillText(`质心居中后交给网络; 单通道`, 40, 452);
  } else if (id === "conv1") {
    drawConvAnim(ctx, view, tr, sel.f, 28, 28, eng.W.c1w, 1, 8);
  } else if (id === "pool1") {
    drawMaps3D(ctx, view, mapsFor(tr.c1, 8, 28, 28, (c) => `f${c} ReLU`), 28, 28, `ReLU 后 8×28×28`, threeD, sel.f);
  } else if (id === "conv2") {
    drawConv2(ctx, view, tr, sel.o, sel.c);
  } else if (id === "pool2") {
    drawMaps3D(ctx, view, mapsFor(tr.c2, 16, 14, 14, (c) => `k${c}`), 14, 14, `ReLU 后 16×14×14`, threeD, sel.o);
  } else if (id === "flat") {
    drawFlat(ctx, view, tr);
  } else if (id === "fc1") {
    drawFC1(ctx, view, tr);
  } else if (id === "softmax") {
    drawSoftmax(ctx, view, tr);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  $("#stageLabel").textContent = `STAGE ${stage + 1}/${STAGES.length} · ${STAGES[stage].name}`;
  document.querySelectorAll("#stageBar button").forEach((b, i) => {
    b.className = i === stage ? "active" : (i < stage ? "done" : "");
  });
}
function drawMatrix(ctx, data, H, W, x0, y0, size, title) {
  const cell = size / Math.max(H, W);
  let mx = 1e-9; for (let i = 0; i < data.length; i++) mx = Math.max(mx, Math.abs(data[i]));
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    const v = Math.max(0, Math.min(1, data[i * W + j] / mx));
    ctx.fillStyle = `rgb(${Math.round(255 * v)},${Math.round(255 * v)},${Math.round(255 * v)})`;
    ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
  }
  ctx.strokeStyle = "#2a3d54"; ctx.strokeRect(x0, y0, W * cell, H * cell);
  ctx.fillStyle = "#9fb3cc"; ctx.font = "12px Consolas, monospace"; ctx.fillText(title, x0, y0 - 8);
}
function drawMatrixColor(ctx, data, H, W, x0, y0, size, title, sel = null) {
  const cell = size / Math.max(H, W);
  let mx = 1e-9; for (let i = 0; i < data.length; i++) mx = Math.max(mx, Math.abs(data[i]));
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    const t = Math.max(-1, Math.min(1, data[i * W + j] / mx));
    let r, g, b;
    if (t < 0) { r = 40 + 40 * t; g = 90 + 70 * t; b = 255; } else { r = 255; g = 110 + 145 * (1 - t); b = 60 * (1 - t) + 30; }
    ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
    ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
    if (sel && sel[0] === i && sel[1] === j) { ctx.strokeStyle = "#3ddc97"; ctx.lineWidth = 2; ctx.strokeRect(x0 + j * cell, y0 + i * cell, cell, cell); ctx.lineWidth = 1; }
  }
  ctx.strokeStyle = "#2a3d54"; ctx.strokeRect(x0, y0, W * cell, H * cell);
  ctx.fillStyle = "#9fb3cc"; ctx.font = "12px Consolas, monospace"; ctx.fillText(title, x0, y0 - 8);
}
function drawConvAnim(ctx, view, tr, f, H, W, kernels, inC, outC) {
  const K = 5, pad = 2;
  const ink = tr.x;                            // 输入(Float32Array)
  const ker = kernels.slice(f * inC * K * K, (f * inC + 1) * K * K);
  const total = H * W;
  const pos = Math.floor(anim.t * total) % total;
  const oy = Math.floor(pos / W), ox = pos % W;
  const CS = 11;                                // 输入显示单元
  drawMatrix(ctx, ink, 28, 28, 40, 70, 320, `输入 x + 滑动窗口 (核 f${f})`);
  ctx.strokeStyle = "#3ddc97"; ctx.lineWidth = 2;
  ctx.strokeRect(40 + (ox - pad) * CS, 70 + (oy - pad) * CS, K * CS, K * CS); ctx.lineWidth = 1;
  // 核矩阵
  const kx = 400, ky = 70;
  ctx.fillStyle = "#9fb3cc"; ctx.font = "12px Consolas, monospace";
  ctx.fillText(`卷积核 f${f} (5×5)`, kx, ky - 8);
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) {
    const v = ker[a * K + b];
    const t = Math.max(-1, Math.min(1, v / (Math.max(...ker.map(Math.abs)) || 1)));
    ctx.fillStyle = t < 0 ? `rgba(90,130,255,${0.25 + 0.7 * -t})` : `rgba(255,150,80,${0.2 + 0.75 * t})`;
    ctx.fillRect(kx + b * 34, ky + a * 34, 32, 32);
    ctx.fillStyle = "#dbe7f7"; ctx.font = "9px Consolas"; ctx.fillText(v.toFixed(2), kx + b * 34 + 3, ky + a * 34 + 19);
  }
  // 逐项乘 → 求和
  const px = 400, py = 260;
  ctx.fillStyle = "#9fb3cc"; ctx.font = "12px Consolas, monospace";
  ctx.fillText(`逐项相乘 → 求和 Σ`, px, py - 8);
  let sum = 0;
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) {
    const yy = oy + a - 2, xx = ox + b - 2;
    const u = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? ink[yy * W + xx] : 0;
    const p = ker[a * K + b] * u; sum += p;
    const t = Math.max(-1, Math.min(1, p / 2));
    ctx.fillStyle = t < 0 ? `rgba(90,130,255,${0.2 + 0.6 * -t})` : `rgba(255,150,80,${0.15 + 0.6 * t})`;
    ctx.fillRect(px + b * 34, py + a * 34, 32, 32);
    ctx.fillStyle = "#dbe7f7"; ctx.font = "9px Consolas"; ctx.fillText(p.toFixed(2), px + b * 34 + 3, py + a * 34 + 19);
  }
  const outv = sum + eng.W.c1b[f];
  ctx.fillStyle = "#dbe7f7"; ctx.font = "14px Consolas, monospace";
  ctx.fillText(`Σ(K·u) = ${sum.toFixed(3)}   + β = ${outv.toFixed(3)}`, px, py + 200);
  ctx.fillStyle = "#3ddc97";
  ctx.fillText(`→ 特征图位置 (${oy},${ox})`, px, py + 224);
  // 输出特征图(已算部分)
  const outMap = new Float32Array(H * W);
  for (let i = 0; i <= pos; i++) outMap[i] = tr.c1[f * H * W + i];
  drawMatrixColor(ctx, outMap, H, W, 660, 70, 230, `特征图 f${f} (实时填充)`, [oy, ox]);
}
function drawMaps3D(ctx, view, maps, H, W, title, threeD, hl) {
  if (threeD) { drawStack(view, maps.map((m, i) => ({ ...m, hl: i === hl })), H, W, { scale: H === 28 ? 2.6 : 3.4, gap: 40 }); }
  else {
    const cols = Math.min(8, maps.length), size = 92;
    maps.forEach((m, i) => {
      const x = 30 + (i % cols) * (size + 16), y = 50 + Math.floor(i / cols) * (size + 40);
      drawMatrixColor(ctx, m.data, H, W, x, y, size, m.label, i === hl ? [0, 0] : null);
    });
  }
  ctx.fillStyle = "#9fb3cc"; ctx.font = "13px Consolas, monospace";
  ctx.fillText(title + (threeD ? " · 3D 堆栈(可拖动旋转)" : ""), 30, 28);
}
function drawConv2(ctx, view, tr, o, cin) {
  const CS = 11;
  // 输入 8 通道
  ctx.fillStyle = "#9fb3cc"; ctx.font = "12px Consolas, monospace";
  ctx.fillText(`输入 8 通道 → 输出通道 k${o}: 逐通道卷积后求和`, 30, 26);
  for (let c = 0; c < 8; c++) {
    const x = 30 + (c % 4) * 118, y = 44 + Math.floor(c / 4) * 150;
    drawMatrixColor(ctx, tr.p1.slice(c * 14 * 14, (c + 1) * 14 * 14), 14, 14, x, y, 96, `p1 #${c}`, c === cin ? [0, 0] : null);
  }
  // 参与相加的通道高亮
  const kerC = eng.W.c2w.slice((o * 8 + cin) * 25, (o * 8 + cin + 1) * 25);
  drawMatrixColor(ctx, kerC, 5, 5, 520, 60, 150, `核 K[${o},${cin}] (5×5)`);
  const contrib = conv2d(tr.p1.slice(cin * 196, (cin + 1) * 196), 1, 14, 14, kerC, 1, 5, [0]);
  drawMatrixColor(ctx, contrib, 14, 14, 700, 60, 190, `通道 ${cin} 的贡献`);
  const outMap = tr.c2.slice(o * 196, (o + 1) * 196);
  drawMatrixColor(ctx, outMap, 14, 14, 700, 290, 190, `∑ 全部通道 + β → ReLU = k${o}`);
  ctx.fillStyle = "#dbe7f7"; ctx.font = "13px Consolas, monospace";
  ctx.fillText(`对 8 个通道各做一次 5×5 卷积, 再逐点相加 → 1 张 14×14 特征图; 16 个输出通道各有一套 (8×5×5) 核`, 30, 500);
}
function drawFlat(ctx, view, tr) {
  ctx.fillStyle = "#9fb3cc"; ctx.font = "12px Consolas, monospace";
  ctx.fillText("7×7×16 = 784 维向量(热力条, 按通道 c 优先排列)", 30, 26);
  const n = 784, cw = (view.width - 60) / n;
  let mx = 1e-9; for (let i = 0; i < n; i++) mx = Math.max(mx, tr.flat[i]);
  for (let i = 0; i < n; i++) {
    const t = Math.max(0, Math.min(1, tr.flat[i] / mx));
    ctx.fillStyle = `rgb(${(40 + 215 * t) | 0},${(90 + 120 * t) | 0},${(200 - 100 * t) | 0})`;
    ctx.fillRect(30 + i * cw, 60, Math.max(1, cw + 0.4), 120);
  }
  // 通道刻度
  for (let c = 0; c <= 16; c++) {
    const x = 30 + (c * 49) * cw;
    ctx.strokeStyle = "rgba(120,160,220,.35)"; ctx.beginPath(); ctx.moveTo(x, 180); ctx.lineTo(x, 190); ctx.stroke();
    if (c < 16) { ctx.fillStyle = "#7d90a8"; ctx.font = "10px Consolas"; ctx.fillText("c" + c, x + 2, 204); }
  }
  ctx.fillStyle = "#dbe7f7"; ctx.font = "13px Consolas, monospace";
  ctx.fillText(`非零分量 ${Array.from(tr.flat).filter((v) => v > 0.01).length} / 784 · 最大 ${mx.toFixed(3)}`, 30, 250);
  ctx.fillText("展平后送入 FC120: 每个隐藏神经元看到全部 784 个值(权重 784 个)", 30, 276);
}
function drawFC1(ctx, view, tr) {
  const n = 120, bw = (view.width - 80) / n;
  let mx = 1e-9; for (let i = 0; i < n; i++) mx = Math.max(mx, tr.f1[i]);
  let active = 0;
  ctx.fillStyle = "#9fb3cc"; ctx.font = "12px Consolas, monospace";
  ctx.fillText("120 个隐藏神经元(ReLU 后) — 负值被关闭, 形成稀疏门控", 30, 26);
  for (let i = 0; i < n; i++) {
    const h = 150 * (tr.f1[i] / mx), x = 40 + i * bw;
    if (tr.f1[i] > 0) active++;
    ctx.fillStyle = tr.f1[i] > 0 ? "#3ddc97" : "#26384d";
    ctx.fillRect(x, 200 - h, Math.max(1, bw - 1), Math.max(2, h));
  }
  ctx.fillStyle = "#dbe7f7"; ctx.font = "13px Consolas, monospace";
  ctx.fillText(`激活 ${active}/120 (${(100 * active / n).toFixed(0)}%) · 峰值 ${mx.toFixed(3)}`, 30, 240);
  ctx.fillText("Dropout: 训练时随机 mask 并 /(1-p) 缩放, 推理时全部打开 —— 本演示为推理, 故 dropout 不生效", 30, 266);
  // dropout 示意
  ctx.fillText("示意(训练时): ", 30, 300);
  let seed = 7; const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let i = 0; i < 40; i++) {
    const on = rnd() > 0.25;
    ctx.fillStyle = on ? "#3ddc97" : "#2a3a4d";
    ctx.fillRect(120 + i * 14, 288, 11, 14);
  }
  ctx.fillStyle = "#7d90a8"; ctx.fillText("绿=保留, 暗=丢弃(p=0.25)", 120, 320);
}
function drawSoftmax(ctx, view, tr) {
  const probs = tr.probs, top = probs.indexOf(Math.max(...probs));
  ctx.fillStyle = "#9fb3cc"; ctx.font = "13px Consolas, monospace";
  ctx.fillText("10 个 logit → Softmax → 概率分布(和为 1)", 30, 26);
  const bw = (view.width - 140) / 10;
  for (let i = 0; i < 10; i++) {
    const x = 90 + i * bw, h = 300 * probs[i];
    ctx.fillStyle = i === top ? "#3ddc97" : "#2f7fa5";
    ctx.fillRect(x, 380 - h, bw - 14, h);
    ctx.fillStyle = "#dbe7f7"; ctx.font = "13px Consolas, monospace";
    ctx.fillText(String(i), x + bw / 2 - 12, 404);
    ctx.fillStyle = "#9fb3cc"; ctx.font = "10px Consolas"; ctx.fillText((probs[i] * 100).toFixed(1) + "%", x, 370 - h);
  }
  ctx.fillStyle = "#dbe7f7"; ctx.font = "14px Consolas, monospace";
  ctx.fillText(`预测 = ${top}   置信度 = ${(probs[top] * 100).toFixed(2)}%   Σp = ${probs.reduce((a, b) => a + b, 0).toFixed(4)}`, 30, 450);
  ctx.fillStyle = "#7d90a8"; ctx.font = "12px Consolas, monospace";
  ctx.fillText(`logits: [${Array.from(tr.logits).map((v) => v.toFixed(2)).join(", ")}]`, 30, 474);
}

/* ---------------------------------------------------------------- 概率面板 */
function renderProbs(probs) {
  const top = probs.indexOf(Math.max(...probs));
  $("#probBox").innerHTML = probs.map((p, i) => `<div class="prow ${i === top ? "top" : ""}">
    <span>${i}</span><span class="pbar"><i style="width:${(p * 100).toFixed(2)}%"></i></span>
    <span>${(p * 100).toFixed(2)}%</span></div>`).join("");
  $("#sumTag").textContent = `Σ=${probs.reduce((a, b) => a + b, 0).toFixed(4)}`;
}

/* ---------------------------------------------------------------- 运行 */
function setStagesEnabled(on) {
  document.querySelectorAll("#stageBar button").forEach((b) => (b.disabled = !on));
  $("#btnPlay").disabled = !on; $("#btnNext").disabled = !on; $("#btnPrev").disabled = !on;
}
function doRun(input28, meta = {}) {
  eng.lastInput = input28;
  const t0 = performance.now();
  const tr = forward(input28);
  const ms = performance.now() - t0;
  const top = tr.probs.indexOf(Math.max(...tr.probs));
  $("#predLab").textContent = top;
  $("#confVal").textContent = (tr.probs[top] * 100).toFixed(2) + "%";
  $("#msVal").textContent = ms.toFixed(1) + " ms";
  if (meta.true !== undefined) $("#trueLab").textContent = meta.true;
  renderProbs(tr.probs);
  setStagesEnabled(true);
  if (meta.keepStage !== true) { stage = 0; }
  anim.t = 0;
  draw();
  startAnim();
  updateReadout();
  renderFormula();
  // 可选: TF.js 交叉验证
  tfCrossCheck(input28, tr.probs).catch(() => {});
}
function updateReadout() {
  const tr = eng.trace; if (!tr) return;
  const sel = currentSelection();
  const top = tr.probs.indexOf(Math.max(...tr.probs));
  const txt = {
    input: `输入: 28×28×1 = 784 个像素; 非零 ${Array.from(tr.x).filter((v) => v > 0.05).length}`,
    conv1: `Conv1: 8 个 5×5 核 × 784 个位置 = 6272 次滑窗乘加 → 8×28×28; 当前展示 f${sel.f}(响应最强)`,
    pool1: `ReLU+Pool1: 8×(28×28) → 8×(14×14); 空间减半, 保留最大响应`,
    conv2: `Conv2: 16 通道 × (8×5×5) 核 → 16×14×14; 展示 k${sel.o} 的通道 ${sel.c} 贡献`,
    pool2: `ReLU+Pool2: 16×(14×14) → 16×(7×7)`,
    flat: `Flatten: 16×7×7 = 784 维; 非零 ${Array.from(tr.flat).filter((v) => v > 0.01).length}`,
    fc1: `FC120: 784→120 (${(784 * 120 + 120).toLocaleString()} 参数); 激活 ${Array.from(tr.f1).filter((v) => v > 0).length}/120`,
    softmax: `FC10: 120→10; 预测 ${top}, 置信 ${(tr.probs[top] * 100).toFixed(2)}%, Σp=${tr.probs.reduce((a, b) => a + b, 0).toFixed(4)}`,
  }[STAGES[stage].id];
  $("#readout").textContent = txt + "\n" + (eng.meta && eng.meta.test_acc ? `模型 ${eng.modelName} · test_acc ${(eng.meta.test_acc * 100).toFixed(2)}%` : "");
}
function startAnim() {
  stopAnim();
  if (!$("#chkAnim").checked) return;
  const tr = eng.trace; if (!tr) return;
  anim.timer = setInterval(() => {
    anim.t = (anim.t + 0.02) % 1;
    if (STAGES[stage].id === "conv1") draw();
  }, 40);
}
function stopAnim() { if (anim.timer) clearInterval(anim.timer); anim.timer = null; }
function goStage(i) {
  if (!eng.trace) return;
  stage = Math.max(0, Math.min(STAGES.length - 1, i));
  anim.t = 0; draw(); renderFormula(); updateReadout(); startAnim();
}

/* ---------------------------------------------------------------- TF.js 交叉验证 */
async function tfCrossCheck(input28, jsProbs) {
  if (!window.tf) { $("#tfState").textContent = "不可用(离线跳过)"; $("#ledTf").className = "led warn"; return; }
  const W = eng.W;
  const probs = tf.tidy(() => {
    const k1 = tf.tensor4d(Array.from(W.c1w), [8, 1, 5, 5]).transpose([2, 3, 1, 0]);   // → [5,5,1,8]
    const k2 = tf.tensor4d(Array.from(W.c2w), [16, 8, 5, 5]).transpose([2, 3, 1, 0]);  // → [5,5,8,16]
    let x = tf.tensor4d(input28.flat(), [1, 28, 28, 1]);
    x = tf.relu(tf.add(tf.conv2d(x, k1, 1, "same"), tf.tensor1d(Array.from(W.c1b))));
    x = tf.maxPool(x, 2, 2, "valid");
    x = tf.relu(tf.add(tf.conv2d(x, k2, 1, "same"), tf.tensor1d(Array.from(W.c2b))));
    x = tf.maxPool(x, 2, 2, "valid");
    // 展平: JS/PyTorch 为 (C,H,W) 序, TF 为 (H,W,C) 序 → 交换 FC1 权重列
    const perm = new Int32Array(784);
    for (let h = 0; h < 7; h++) for (let w = 0; w < 7; w++) for (let c = 0; c < 16; c++) perm[h * 112 + w * 16 + c] = c * 49 + h * 7 + w;
    const f1w = tf.tensor2d(Array.from(W.f1w), [120, 784]);
    const f1wT = tf.transpose(f1w);                        // [784,120]
    const f1wPerm = tf.gather(f1wT, tf.tensor1d(Array.from(perm), "int32")); // [784,120] 重排行
    const flat = tf.reshape(x, [1, 784]);
    let y = tf.relu(tf.add(tf.matMul(flat, f1wPerm), tf.tensor1d(Array.from(W.f1b))));
    y = tf.add(tf.matMul(y, tf.transpose(tf.tensor2d(Array.from(W.f2w), [10, 120]))), tf.tensor1d(Array.from(W.f2b)));
    return Array.from(tf.softmax(y, 1).dataSync());
  });
  const diff = Math.max(...probs.map((p, i) => Math.abs(p - jsProbs[i])));
  const agree = probs.indexOf(Math.max(...probs)) === jsProbs.indexOf(Math.max(...jsProbs));
  $("#tfState").textContent = `一致 ${agree ? "✓" : "✗"} · max|Δp|=${diff.toExponential(1)}`;
  $("#ledTf").className = "led " + (agree && diff < 0.02 ? "on" : "warn");
}

/* ---------------------------------------------------------------- 拉格朗日面板 */
async function loadLagrangian() {
  try {
    const r = await fetch("/api/cnn/layers", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }).then((x) => x.json());
    if (!r.ok) throw new Error(r.error || "no data");
    const L = r.data, agg = L.conv1.agg;
    $("#lagTag").textContent = `偶核 ${agg.even_n}/8 · PSD ${agg.psd_n}/8 · 奇部均值 ${(agg.mean_odd_frac * 100).toFixed(1)}%`;
    const cards = L.conv1.filters.map((f) => {
      const q = f.qa, e = q.even_frac, o = q.odd_frac;
      const badge = q.definite === "psd" ? "g" : q.definite === "complex" ? "v" : "a";
      return `<div class="lag-card">
        <canvas class="k" data-k="${f.idx}"></canvas>
        <div class="t"><span>f${f.idx} ${q.type}</span><span class="badge ${badge}">${q.definite}</span></div>
        <div class="evbar" title="偶部(自伴) / 奇部(陀螺)"><i class="e" style="width:${(e * 100).toFixed(0)}%"></i><i class="o" style="width:${(o * 100).toFixed(0)}%"></i></div>
        <div class="t"><span>偶 ${(e * 100).toFixed(0)}%</span><span>奇 ${(o * 100).toFixed(0)}%</span></div>
        <div class="t"><span>谱 [${fmt(q.spectrum.re_min, 2)}, ${fmt(q.spectrum.re_max, 2)}]</span></div>
        <div class="t"><span>≈${q.template.best}</span></div>
      </div>`;
    }).join("");
    const kk = L.conv1.agg.kinds || {};
    $("#lagBox").innerHTML = `
      <div id="lagFormula"></div>
      <div class="small" style="font:11.5px/1.7 var(--mono);color:#9fb3cc;margin:6px 0 10px">
        8 个第一层卷积核按「算子×拉格朗日设计空间」逐个体检:
        偶部 kₛ(自伴 ⇒ 可导出二次拉氏量) / 奇部 kₐ(反自伴 ⇒ 陀螺·方向);
        PSD(ĥ≥0, 凸能量)在自由训练下恒为 0 —— 特征层是“过零/差分测量器”, 不是能量泛函。
        频带分布: ${Object.entries(kk).map(([k, v]) => `${k}×${v}`).join(", ")}
      </div>
      <div class="lag-grid">${cards}</div>`;
    L.conv1.filters.forEach((f) => {
      const cv = document.querySelector(`#lagBox canvas[data-k="${f.idx}"]`);
      if (cv) heat(cv, f.k, "div");
    });
    renderTex($("#lagFormula"),
      "S[u]=\\tfrac12\\!\\int u\\,(K*u)\\,dx,\\qquad \\frac{\\delta S}{\\delta u}=K*u \\quad\\Longleftrightarrow\\quad K \\text{ 自伴} \\;(k(x)=k(-x))",
      "可导出拉氏量的条件: 核为偶函数(自伴)");
    $("#lagBox").dataset.loaded = "1";
    katexAppend($("#lagBox"), [
      ["k(x)=k_s(x)+k_a(x),\\quad k_s=\\tfrac12[k(x)+k(-x)],\\;\\; k_a=\\tfrac12[k(x)-k(-x)]", "偶/奇分解: 势能(自伴) + 陀螺(反自伴)"],
      ["\\hat k(\\omega)\\ge 0\\;\\forall\\omega \\iff S \\text{ 凸(正定 / L3 锥)};\\qquad R[v]=\\tfrac12\\!\\int v\\,(D*v)\\,dx", "正定 ⇔ 凸能量; 负谱经 Rayleigh 耗散函数纳入变分"],
      ["\\hat h_c(\\omega)=\\operatorname{Re}\\hat h(\\omega)+\\operatorname{Im}\\hat h(\\omega)\\in\\mathbb{R}", "复化(虚数场)视角: ĉ=kₛ−ikₐ ⇒ Hermitian, 奇部=动能/电流项"],
    ]);
  } catch (e) {
    $("#lagBox").innerHTML = `<div style="font:11.5px var(--mono);color:#7d90a8">算子分析不可用: ${e.message}(需后端运行且已训练)</div>`;
  }
}
function katexAppend(box, items) {
  let anchor = box.querySelector("#lagFormula");
  items.forEach(([tex, cap]) => {
    const d = document.createElement("div");
    d.className = "formula";
    if (anchor && anchor.parentNode) anchor.parentNode.insertBefore(d, anchor.nextSibling);
    else box.appendChild(d);
    renderTex(d, tex, cap);
    anchor = d;
  });
}

/* ---------------------------------------------------------------- 交互绑定 */
function initStageBar() {
  $("#stageBar").innerHTML = STAGES.map((s, i) => `<button data-i="${i}">${i + 1}. ${s.name}</button>`).join("");
  document.querySelectorAll("#stageBar button").forEach((b) => (b.onclick = () => { stopPlay(); goStage(+b.dataset.i); }));
  setStagesEnabled(false);
}
function stopPlay() {
  playing = false; clearInterval(playTimer); playTimer = null; $("#btnPlay").textContent = "▷ 自动播放";
}
function initControls() {
  $("#btnRun").onclick = async () => {
    const m = rasterize();
    const mass = m.flat().reduce((a, b) => a + b, 0);
    if (mass < 0.5) return toast("先在面板上写一个数字", true);
    showThumb(m);
    $("#inputMeta").textContent = `已写入 · 质量 ${mass.toFixed(1)}`;
    doRun(m);
    toast("前向推理完成");
  };
  $("#btnClear").onclick = () => { clearPad(); stopPlay(); eng.trace = null; stage = 0; draw(); $("#readout").textContent = "等待输入…"; renderFormula(); };
  $("#btnSample").onclick = async () => {
    try {
      const i = Math.floor(Math.random() * 10000);
      const r = await fetch(`/api/cnn/sample?i=${i}&model=${encodeURIComponent(eng.modelName)}`).then((x) => x.json());
      if (!r.ok) throw new Error(r.error);
      const img = r.data.img;
      // 画到画板上
      pad.ctx.fillStyle = "#000"; pad.ctx.fillRect(0, 0, pad.size, pad.size);
      const cell = pad.size / 28;
      for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
        const v = img[y][x]; pad.ctx.fillStyle = `rgb(${(v * 255) | 0},${(v * 255) | 0},${(v * 255) | 0})`;
        pad.ctx.fillRect(x * cell, y * cell, cell, cell);
      }
      showThumb(img);
      $("#inputMeta").textContent = `测试样本 #${r.data.index}`;
      doRun(img, { true: r.data.true });
      toast(`载入测试样本 #${r.data.index}(真值 ${r.data.true})`);
    } catch (e) { toast("载入失败: " + e.message, true); }
  };
  $("#btnNext").onclick = () => { stopPlay(); goStage(stage + 1); };
  $("#btnPrev").onclick = () => { stopPlay(); goStage(stage - 1); };
  $("#btnPlay").onclick = () => {
    if (playing) return stopPlay();
    playing = true; $("#btnPlay").textContent = "⏸ 暂停";
    goStage(0);
    playTimer = setInterval(() => { if (stage >= STAGES.length - 1) return stopPlay(); goStage(stage + 1); }, 2600);
  };
  $("#chk3d").onchange = draw;
  $("#chkAnim").onchange = () => (anim.timer ? stopAnim() : startAnim());
  // 3D 旋转
  const view = $("#view");
  let drag = null;
  view.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; view.setPointerCapture(e.pointerId); });
  view.addEventListener("pointermove", (e) => {
    if (!drag) return;
    cam.ry += (e.clientX - drag.x) * 0.01; cam.rx += (e.clientY - drag.y) * 0.008;
    cam.rx = Math.max(-1.3, Math.min(1.3, cam.rx));
    drag = { x: e.clientX, y: e.clientY }; draw();
  });
  view.addEventListener("pointerup", () => (drag = null));
  view.addEventListener("wheel", (e) => { e.preventDefault(); cam.zoom = Math.max(0.4, Math.min(2.6, cam.zoom * (e.deltaY > 0 ? 0.94 : 1.06))); draw(); }, { passive: false });
}
async function initModels() {
  const sel = $("#modelSel");
  const names = ["cnn_torch.json", "cnn_mnist.json"];
  const avail = [];
  for (const n of names) {
    try { const r = await fetch(`/api/cnn/weights?model=${n}`).then((x) => x.json()); if (r.ok) avail.push({ n, acc: r.data.report && r.data.report.test_acc }); } catch (e) { /* skip */ }
  }
  if (!avail.length) { toast("没有可用模型, 请先运行训练脚本", true); $("#engineState").textContent = "无模型"; return; }
  sel.innerHTML = avail.map((a) => `<option value="${a.n}">${a.n}${a.acc ? ` (test ${(a.acc * 100).toFixed(2)}%)` : ""}</option>`).join("");
  sel.onchange = async () => { await loadModel(sel.value); toast("已切换模型: " + sel.value); };
  await loadModel(avail[0].n);
  $("#engineState").textContent = "就绪(纯 JS)";
  $("#ledEngine").className = "led on";
}

/* ---------------------------------------------------------------- 可选 CDN(KaTeX / TF.js) */
function loadCss(href) {
  const l = document.createElement("link"); l.rel = "stylesheet"; l.href = href; document.head.appendChild(l);
}
function loadScript(src, timeoutMs) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    const tm = setTimeout(() => { s.onerror = s.onload = null; reject(new Error("timeout")); }, timeoutMs || 8000);
    s.src = src; s.async = true;
    s.onload = () => { clearTimeout(tm); resolve(true); };
    s.onerror = () => { clearTimeout(tm); reject(new Error("load error")); };
    document.head.appendChild(s);
  });
}
async function loadExternal() {
  // 1) KaTeX: 优先本地 vendor(离线可用), 失败再退回 CDN
  const katexCss = "vendor/katex/katex.min.css", katexJs = "vendor/katex/katex.min.js";
  const cdnCss = "https://unpkg.com/katex@0.16.9/dist/katex.min.css";
  const cdnJs = "https://unpkg.com/katex@0.16.9/dist/katex.min.js";
  const useKatex = async (css, js) => {
    loadCss(css);
    await loadScript(js, 7000);
    renderFormula();
    if ($("#lagBox") && $("#lagBox").dataset.loaded) loadLagrangian();
  };
  try { await useKatex(katexCss, katexJs); }
  catch (e) { try { await useKatex(cdnCss, cdnJs); } catch (e2) { /* 保持等宽降级 */ } }
  // 2) TensorFlow.js: 仅用于交叉验证, 多镜像尝试, 全部失败则跳过
  const tfUrls = [
    "https://unpkg.com/@tensorflow/tfjs@4.20.0/dist/tf.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/tensorflow/4.20.0/tf.min.js",
  ];
  for (const u of tfUrls) {
    try {
      await loadScript(u, 15000);
      $("#tfState").textContent = "已加载 · 待运行验证";
      if (eng.lastInput && eng.trace) tfCrossCheck(eng.lastInput, eng.trace.probs).catch(() => {});
      return;
    } catch (e) { /* 试下一个 */ }
  }
  $("#tfState").textContent = "不可用(离线跳过)";
  $("#ledTf").className = "led warn";
}

/* ---------------------------------------------------------------- 启动 */
async function boot() {
  initPad(); initStageBar(); initControls();
  renderFormula(); draw();
  try { await initModels(); } catch (e) { toast("模型载入失败: " + e.message, true); }
  loadLagrangian();
  loadExternal();
}
window.__demo = {  // 供自动化测试
  run: (img, meta) => { showThumb(img); doRun(img, meta || {}); },
  setStage: (i) => { stopPlay(); goStage(i); },
  state: () => ({ model: eng.modelName, probs: eng.trace ? Array.from(eng.trace.probs) : null, stage, tf: $("#tfState").textContent }),
  ready: () => !!eng.W,
};
document.addEventListener("DOMContentLoaded", boot);
