#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MNIST 手写数字 CNN —— 完整训练脚本 + 实时 Web 进度看板 (PyTorch)

用法:
  python3 backend/train_web.py                       # 默认 15 轮, 3 epochs, 端口 8020
  python3 backend/train_web.py --epochs 5 --subset 30000 --port 8020
  python3 backend/train_web.py --out ../data/cnn_mnist.json
然后浏览器打开 http://127.0.0.1:8020 实时查看:
  · 训练/验证损失与验证准确率曲线(实时)
  · epoch/step 进度条 + ETA
  · 8 张固定测试样本的实时预测(写数字真值 + 预测 + 概率)
  · 第一层 8 个 5×5 卷积核热图(随训练演化)
  · 控制台日志

依赖: torch(必需), numpy(必需); torchvision 可选(没有则用仓库内置 MNIST 解析/缓存)。
训练结束把权重导出为 data/*.json(与 numpy 版引擎/演示页兼容的格式)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnn  # noqa: E402  (复用 MNIST 下载/解析缓存与数据目录)

ARCH_DOC = {
    "name": "LeNet 风格 MNIST CNN (PyTorch)",
    "input": "28×28×1 灰度",
    "layers": [
        {"no": 1, "type": "Conv2d", "in": 1, "out": 8, "k": 5, "stride": 1, "pad": 2, "out_size": "28×28×8",
         "params": "8×1×5×5+8 = 208", "note": "8 个 5×5 卷积核滑窗 → 8 张特征图"},
        {"no": 2, "type": "ReLU", "note": "非线性激活(负值置 0)"},
        {"no": 3, "type": "MaxPool2d", "k": 2, "stride": 2, "out_size": "14×14×8", "note": "2×2 取最大, 降采样"},
        {"no": 4, "type": "Conv2d", "in": 8, "out": 16, "k": 5, "stride": 1, "pad": 2, "out_size": "14×14×16",
         "params": "16×8×5×5+16 = 3216", "note": "16 个 5×5×8 卷积核(逐通道求和) → 16 张特征图"},
        {"no": 5, "type": "ReLU"},
        {"no": 6, "type": "MaxPool2d", "k": 2, "stride": 2, "out_size": "7×7×16", "note": "再池化"},
        {"no": 7, "type": "Flatten", "out_size": "784", "note": "展平 7×7×16=784 维向量"},
        {"no": 8, "type": "Dropout", "p": 0.25, "note": "训练时随机丢弃(正则, 推理关闭)"},
        {"no": 9, "type": "Linear", "in": 784, "out": 120, "params": "784×120+120 = 94200"},
        {"no": 10, "type": "ReLU"},
        {"no": 11, "type": "Dropout", "p": 0.25},
        {"no": 12, "type": "Linear", "in": 120, "out": 10, "params": "120×10+10 = 1210"},
        {"no": 13, "type": "Softmax", "note": "10 类概率分布, 和为 1"},
    ],
}

# ------------------------------------------------------------------ 状态
STATE = {
    "running": False, "done": False, "error": None,
    "epoch": 0, "epochs": 0, "step": 0, "total_steps": 0,
    "config": {}, "t0": None, "elapsed": 0.0, "eta": None,
    "hist": [], "log": [], "samples": None, "kernels": None,
    "report": None, "device": str(torch.device("cpu")),
}
LOCK = threading.Lock()


def log(msg):
    line = time.strftime("%H:%M:%S ") + str(msg)
    with LOCK:
        STATE["log"].append(line)
        STATE["log"] = STATE["log"][-200:]
    print(line, flush=True)


class Net(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 5, padding=2)
        self.c2 = nn.Conv2d(8, 16, 5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 16, 120)
        self.fc2 = nn.Linear(120, 10)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, keep=None):
        """keep: 可选 dict, 收集中间张量(供进度页/演示解释)"""
        h1 = torch.relu(self.c1(x))
        p1 = torch.max_pool2d(h1, 2)
        h2 = torch.relu(self.c2(p1))
        p2 = torch.max_pool2d(h2, 2)
        flat = p2.flatten(1)
        d = self.drop(flat) if self.training else flat
        f1 = torch.relu(self.fc1(d))
        d2 = self.drop(f1) if self.training else f1
        z = self.fc2(d2)
        if keep is not None:
            keep.update(h1=h1, p1=p1, h2=h2, p2=p2, flat=flat, f1=f1, logits=z)
        return z


def export_json(model, report, out_path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    W = {
        "c1w": sd["c1.weight"], "c1b": sd["c1.bias"],
        "c2w": sd["c2.weight"], "c2b": sd["c2.bias"],
        "f1w": sd["fc1.weight"], "f1b": sd["fc1.bias"],
        "f2w": sd["fc2.weight"], "f2b": sd["fc2.bias"],
    }
    obj = {"arch": ARCH_DOC, "weights": {k: np.round(v, 6).tolist() for k, v in W.items()},
           "report": report}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return os.path.abspath(out_path)


def snapshot(model, Xte_t, Yte_t, idx=None):
    """采样: 固定测试样本的预测 + 第一层卷积核热图"""
    model.eval()
    if idx is None:
        idx = list(range(10))
    with torch.no_grad():
        x = Xte_t[idx]
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    samples = [{
        "img": [[round(float(v), 3) for v in row] for row in Xte_t[i, 0].tolist()],
        "true": int(Yte_t[i]),
        "pred": int(probs[k].argmax()),
        "conf": round(float(probs[k].max()), 4),
        "probs": [round(float(v), 4) for v in probs[k]],
    } for k, i in enumerate(idx)]
    k1 = model.c1.weight.detach().cpu().numpy()  # (8,1,5,5)
    kernels = [[[round(float(v), 4) for v in k1[f, 0, a].tolist()] for a in range(5)] for f in range(8)]
    model.train()
    return samples, kernels


def worker(args):
    try:
        device = torch.device("cpu")
        Xtr, Ytr, Xte, Yte = cnn.ensure_mnist()
        rng = np.random.RandomState(args.seed)
        n = min(args.subset, len(Xtr))
        idx = rng.choice(len(Xtr), size=n, replace=False)
        Xs, Ys = Xtr[idx], Ytr[idx]
        nval = min(args.val, n // 5)
        Xv, Yv = Xs[:nval], Ys[:nval]
        Xt, Yt = Xs[nval:], Ys[nval:]

        Xte_t = torch.tensor(Xte[:2000].astype(np.float32))[:, None, :, :]
        Yte_t = torch.tensor(Yte[:2000].astype(np.int64))
        Xv_t = torch.tensor(Xv.astype(np.float32))[:, None, :, :]
        Yv_t = torch.tensor(Yv.astype(np.int64))

        ds = TensorDataset(torch.tensor(Xt.astype(np.float32))[:, None, :, :],
                           torch.tensor(Yt.astype(np.int64)))
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)
        total = len(dl) * args.epochs
        with LOCK:
            STATE.update(epochs=args.epochs, total_steps=total, t0=time.time(),
                         running=True, config={
                             "subset": n, "val": nval, "epochs": args.epochs, "batch": args.batch,
                             "lr": args.lr, "dropout": args.dropout, "seed": args.seed,
                             "optimizer": "Adam", "loss": "CrossEntropyLoss", "device": "cpu",
                             "arch": "Conv8·5²→P2→Conv16·5²→P2→FC120→FC10"})
        log(f"数据就绪: 训练 {len(Xt)} / 验证 {len(Xv)} / 测试 2000; 共 {total} steps")

        model = Net(args.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 2), gamma=0.5)
        crit = nn.CrossEntropyLoss()
        model.train()
        step = 0
        best = (0.0, None)
        for ep in range(1, args.epochs + 1):
            run_loss, seen = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                z = model(xb)
                loss = crit(z, yb)
                loss.backward()
                opt.step()
                run_loss += float(loss) * len(xb)
                seen += len(xb)
                step += 1
                if step % args.log_every == 0 or step == total:
                    model.eval()
                    with torch.no_grad():
                        vacc = float((model(Xv_t).argmax(1) == Yv_t).float().mean())
                    model.train()
                    el = time.time() - STATE["t0"]
                    with LOCK:
                        STATE.update(step=step, epoch=ep, elapsed=round(el, 1),
                                     eta=round(el / step * (total - step), 1) if step else None,
                                     hist=STATE["hist"] + [{
                                         "step": step, "epoch": ep,
                                         "loss": round(run_loss / max(seen, 1), 4),
                                         "val_acc": round(vacc, 4)}])
                    log(f"epoch {ep} step {step}/{total} loss={run_loss/max(seen,1):.4f} val_acc={vacc:.4f}")
                    run_loss, seen = 0.0, 0
                if step % args.snap_every == 0 or step == total:
                    s, k = snapshot(model, Xte_t, Yte_t)
                    with LOCK:
                        STATE["samples"], STATE["kernels"] = s, k
            sched.step()
            vacc = float((model(Xv_t).argmax(1) == Yv_t).float().mean())
            if vacc > best[0]:
                best = (vacc, {k: v.detach().clone() for k, v in model.state_dict().items()})
        # 用验证最优权重
        if best[1] is not None:
            model.load_state_dict(best[1])
        model.eval()
        with torch.no_grad():
            tacc = float((model(Xte_t).argmax(1) == Yte_t).float().mean())
        report = {"val_acc": round(best[0], 4), "test_acc": round(tacc, 4),
                  "subset": int(n), "epochs": args.epochs, "batch": args.batch,
                  "lr": args.lr, "dropout": args.dropout, "seed": args.seed,
                  "secs": round(time.time() - STATE["t0"], 1),
                  "framework": f"pytorch-{torch.__version__}",
                  "history": STATE["hist"][-60:]}
        path = export_json(model, report, args.out)
        with LOCK:
            STATE.update(running=False, done=True, report=report, out=path,
                         samples=snapshot(model, Xte_t, Yte_t)[0])
        log(f"训练完成: test_acc={tacc:.4f} val_acc={best[0]:.4f} → {path}")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        with LOCK:
            STATE.update(running=False, done=True, error=repr(e))
        log("训练失败: " + repr(e))


# ------------------------------------------------------------------ Web
PAGE = r"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>MNIST CNN 训练进度 · 实时看板</title>
<style>
:root{--bg:#0b0f14;--panel:#111823;--line:#1e2a3a;--text:#d7e3f4;--muted:#7c8ea6;--cyan:#37c8f0;--amber:#ffb347;--green:#3ddc97;--red:#ff6b6b}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:13px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;padding:16px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:12px 16px;margin-bottom:12px}
h1{font-size:16px;letter-spacing:.5px}
.led{width:10px;height:10px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green)}
.led.run{background:var(--amber);box-shadow:0 0 10px var(--amber);animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.3}}
.meta{margin-left:auto;display:flex;gap:16px;flex-wrap:wrap;font-family:Consolas,monospace;color:var(--muted)}
.meta b{color:var(--text)}
.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:12px}
@media (max-width:980px){.grid{grid-template-columns:1fr}}
.panel{border:1px solid var(--line);background:var(--panel);border-radius:8px;overflow:hidden}
.ph{padding:8px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;display:flex;justify-content:space-between}
.pb{padding:12px}
canvas{width:100%;display:block;background:#080c12;border:1px solid var(--line);border-radius:6px}
.bar{height:14px;border:1px solid var(--line);border-radius:7px;overflow:hidden;background:#080c12;margin:8px 0 4px}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .3s}
.samples{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.smp{text-align:center;font-family:Consolas,monospace}
.smp canvas{image-rendering:pixelated;border-color:var(--line)}
.smp .lab{margin-top:4px;font-size:12px}
.smp .ok{color:var(--green)}.smp .bad{color:var(--red)}
.kern{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.kern canvas{image-rendering:pixelated}
pre.log{max-height:220px;overflow:auto;font:11.5px/1.6 Consolas,monospace;color:#9fb3cc;background:#080c12;border:1px solid var(--line);border-radius:6px;padding:8px;white-space:pre-wrap}
.chiplist{display:flex;gap:8px;flex-wrap:wrap;font-family:Consolas,monospace}
.chip{border:1px solid var(--line);border-radius:20px;padding:2px 10px;color:var(--muted)}
.chip b{color:var(--cyan)}
</style></head><body>
<header>
  <span id="led" class="led"></span>
  <h1>MNIST CNN 训练进度 · 实时看板</h1>
  <div class="meta">
    <span>状态 <b id="mStatus">连接中…</b></span>
    <span>epoch <b id="mEpoch">-</b></span>
    <span>step <b id="mStep">-</b></span>
    <span>loss <b id="mLoss">-</b></span>
    <span>val_acc <b id="mAcc">-</b></span>
    <span>用时 <b id="mEl">-</b> / ETA <b id="mEta">-</b></span>
  </div>
</header>
<div class="chiplist" id="cfg" style="margin:0 0 12px 2px"></div>
<div class="grid">
  <div style="display:flex;flex-direction:column;gap:12px">
    <div class="panel"><div class="ph"><span>损失 / 准确率曲线</span><span id="curveInfo"></span></div>
      <div class="pb"><canvas id="chart" height="320"></canvas>
      <div class="bar"><i id="prog"></i></div>
      <div style="display:flex;justify-content:space-between;font-family:Consolas,monospace;color:var(--muted)">
        <span id="pEpoch">epoch -/-</span><span id="pStep">step -/-</span></div></div></div>
    <div class="panel"><div class="ph"><span>测试样本实时预测</span><span>真值 / 预测 / 置信度</span></div>
      <div class="pb samples" id="samples"></div></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:12px">
    <div class="panel"><div class="ph"><span>第一层卷积核 (8×5×5, 实时演化)</span></div>
      <div class="pb kern" id="kern"></div></div>
    <div class="panel"><div class="ph"><span>控制台</span></div>
      <div class="pb"><pre class="log" id="log"></pre></div></div>
    <div class="panel"><div class="ph"><span>网络结构</span></div>
      <div class="pb" style="font-family:Consolas,monospace;font-size:12px;color:#9fb3cc" id="arch"></div></div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
const DIV=[[-1,[50,80,220]],[-.25,[90,150,240]],[0,[10,14,24]],[.25,[250,180,90]],[1,[255,110,60]]];
function ramp(t){t=Math.max(-1,Math.min(1,t));for(let i=0;i<DIV.length-1;i++){if(t>=DIV[i][0]&&t<=DIV[i+1][0]){const a=DIV[i],b=DIV[i+1];const f=(t-a[0])/(b[0]-a[0]||1);return[0,1,2].map(j=>a[1][j]+(b[1][j]-a[1][j])*f)}}return DIV[4][1]}
function heat(cv,mat){const R=mat.length,C=mat[0].length;cv.width=C;cv.height=R;const ctx=cv.getContext('2d');let mx=1e-9;mat.forEach(r=>r.forEach(v=>mx=Math.max(mx,Math.abs(v))));const img=ctx.createImageData(C,R);for(let i=0;i<R;i++)for(let j=0;j<C;j++){const c=ramp(mat[i][j]/mx);const p=(i*C+j)*4;img.data[p]=c[0];img.data[p+1]=c[1];img.data[p+2]=c[2];img.data[p+3]=255}ctx.putImageData(img,0,0)}
function fmtT(s){if(s==null)return'-';s=Math.round(s);return (s/60|0)+'m'+(s%60)+'s'}
let hist=[];
function chart(){const cv=$('#chart'),ctx=cv.getContext('2d');const W=cv.width=cv.clientWidth*2,H=cv.height=320*2;ctx.clearRect(0,0,W,H);
 if(!hist.length)return;const losses=hist.map(h=>h.loss),accs=hist.map(h=>h.val_acc);const n=hist.length;
 const pad=44;const X=i=>pad+(W-pad*2)*(n<2?0:i/(n-1));
 const lmax=Math.max(...losses,1e-6)*1.1,lmin=0;
 const Y=v=>pad+(H-pad*2)*(1-(v-lmin)/(lmax-lmin));
 const Ya=v=>pad+(H-pad*2)*(1-v);
 ctx.strokeStyle='#1e2a3a';ctx.lineWidth=1;for(let g=0;g<=4;g++){const y=pad+(H-pad*2)*g/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();ctx.fillStyle='#7c8ea6';ctx.font='20px Consolas';ctx.fillText((lmax*(1-g/4)).toFixed(2),6,y+6);ctx.fillText((1-g/4).toFixed(2),W-pad+8,y+6)}
 ctx.strokeStyle='#ffb347';ctx.lineWidth=3;ctx.beginPath();losses.forEach((v,i)=>{i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v))});ctx.stroke();
 ctx.strokeStyle='#3ddc97';ctx.beginPath();accs.forEach((v,i)=>{i?ctx.lineTo(X(i),Ya(v)):ctx.moveTo(X(i),Ya(v))});ctx.stroke();
 ctx.fillStyle='#ffb347';ctx.font='22px Consolas';ctx.fillText('loss',pad+6,26);ctx.fillStyle='#3ddc97';ctx.fillText('val_acc',pad+90,26)}
function renderSamples(s){const box=$('#samples');box.innerHTML='';s.forEach(o=>{const d=document.createElement('div');d.className='smp';d.innerHTML=`<canvas></canvas><div class="lab ${o.pred===o.true?'ok':'bad'}">真 ${o.true} · 预 ${o.pred} · ${(o.conf*100).toFixed(1)}%</div>`;box.appendChild(d);heat(d.querySelector('canvas'),o.img);})}
function renderKern(k){const box=$('#kern');box.innerHTML='';k.forEach(m=>{const d=document.createElement('div');d.className='smp';d.innerHTML='<canvas style="height:64px"></canvas>';box.appendChild(d);heat(d.querySelector('canvas'),m);})}
async function tick(){try{const s=await (await fetch('/api/status')).json();
 $('#led').className='led'+(s.running?' run':'');
 $('#mStatus').textContent=s.error?('失败: '+s.error):(s.done?'已完成':'训练中');
 $('#mEpoch').textContent=s.epoch+'/'+s.epochs;$('#mStep').textContent=s.step+'/'+s.total_steps;
 const last=s.hist[s.hist.length-1]||{};$('#mLoss').textContent=last.loss??'-';$('#mAcc').textContent=last.val_acc??'-';
 $('#mEl').textContent=fmtT(s.elapsed);$('#mEta').textContent=s.done?'—':fmtT(s.eta);
 $('#prog').style.width=(s.total_steps?100*s.step/s.total_steps:0)+'%';
 $('#pEpoch').textContent='epoch '+s.epoch+'/'+s.epochs;$('#pStep').textContent='step '+s.step+'/'+s.total_steps;
 hist=s.hist||[];chart();
 if(s.samples)renderSamples(s.samples);if(s.kernels)renderKern(s.kernels);
 $('#log').textContent=(s.log||[]).slice(-60).join('\n');
 $('#cfg').innerHTML=Object.entries(s.config||{}).map(([k,v])=>`<span class="chip">${k} <b>${v}</b></span>`).join('');
 if(s.report&&!s._archShown){$('#arch').textContent=(s.report.framework||'')+'  test_acc='+s.report.test_acc+'  val_acc='+s.report.val_acc+'  '+s.report.secs+'s';s._archShown=1}
}catch(e){$('#mStatus').textContent='服务未响应'}}
fetch('/api/arch').then(r=>r.json()).then(a=>{$('#arch').textContent=(a.layers||[]).map(l=>`${l.no}. ${l.type}${l.k?' k='+l.k:''}${l.out?' '+l.out:''}${l.out_size?' → '+l.out_size:''}${l.params?'  ['+l.params+']':''}`).join('\n')});
setInterval(tick,1000);tick();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _json(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/status"):
            with LOCK:
                snap = {k: STATE[k] for k in ("running", "done", "error", "epoch", "epochs", "step",
                                              "total_steps", "config", "elapsed", "eta", "hist",
                                              "samples", "kernels", "report", "log")}
            return self._json(snap)
        if self.path.startswith("/api/arch"):
            return self._json(ARCH_DOC)
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):  # 静默
        pass


def main():
    ap = argparse.ArgumentParser(description="MNIST CNN 训练 + 实时 Web 进度看板")
    ap.add_argument("--subset", type=int, default=30000, help="训练样本数 (<=60000)")
    ap.add_argument("--val", type=int, default=1000, help="验证集大小")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log-every", type=int, default=50, help="每多少 step 记一次曲线")
    ap.add_argument("--snap-every", type=int, default=100, help="每多少 step 更新样本/卷积核快照")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--out", default=os.path.join(cnn.DATA_DIR, "cnn_torch.json"))
    args = ap.parse_args()

    t = threading.Thread(target=worker, args=(args,), daemon=True)
    print(f"训练启动: subset={args.subset} epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} dropout={args.dropout}")
    print(f"进度看板: http://{args.host}:{args.port}  (训练结束自动停止数据更新, Ctrl+C 退出)")
    t.start()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
