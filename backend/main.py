# -*- coding: utf-8 -*-
"""
算子生成与分析工作台 —— 本地 Web 服务(纯标准库, 零依赖)
  python3 backend/main.py [port]
默认 http://127.0.0.1:8017 ; 可用环境变量 PORT / HOST 覆盖。
静态前端位于 ../frontend; JSON API:
  POST /api/analyze     {kernel: [[...]], ...}   完整分析报告
  POST /api/parametric  {family, m, params}      参数化核族生成
  POST /api/random      {m, kind, intensity?, seed}  按类别随机生成
  GET  /api/health
"""
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernelmath as km  # noqa: E402

try:
    import cnn  # noqa: E402
    CNN_OK = cnn.NP_OK
except Exception:  # noqa: BLE001
    cnn = None
    CNN_OK = False

_train_state = {"running": False, "started_at": None, "log": [], "report": None, "error": None}


def _cnn_train_worker(params):
    """后台线程执行 CNN 训练。"""
    def log(s):
        _train_state["log"].append(s)
        print("[cnn-train]", s, flush=True)
    try:
        W, rep = cnn.train(**params)
        cnn.save_model(W, rep)
        _train_state["report"] = rep
        log(f"训练完成: test_acc={rep.get('test_acc')} ({(rep.get('secs', 0))}s)")
    except Exception as e:  # noqa: BLE001
        _train_state["error"] = str(e)
        traceback.print_exc()
    finally:
        _train_state["running"] = False

FRONTEND = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend"))
MAX_BODY = 512 * 1024
MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


def _validate_kernel(kernel):
    if not isinstance(kernel, list) or not kernel:
        raise ValueError("kernel 必须是二维数组")
    M = len(kernel)
    if M % 2 == 0:
        raise ValueError("内核边长需为奇数(编辑面板强制奇数尺寸)")
    if M > 33:
        raise ValueError("内核过大(≤33)")
    out = []
    for row in kernel:
        if not isinstance(row, list) or len(row) != M:
            raise ValueError("内核必须是正方形")
        r = []
        for v in row:
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"非法数值: {v!r}")
            if not (abs(f) <= 1e6):
                raise ValueError("数值超出范围")
            r.append(f)
        out.append(r)
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "OperatorLab/1.0"

    # ---- helpers ------------------------------------------------------
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, msg, code=400):
        self._send_json({"ok": False, "error": msg}, code)

    def _read_body(self):
        ln = int(self.headers.get("Content-Length") or 0)
        if ln <= 0:
            return {}
        if ln > MAX_BODY:
            raise ValueError("请求体过大")
        raw = self.rfile.read(ln)
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else {}

    # ---- routing ------------------------------------------------------
    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._send_json({
                "ok": True,
                "name": "operator-lab",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "engine": "kernelmath pure-python",
                "cnn": CNN_OK,
            })
        if path == "/api/cnn/model":
            return self._cnn_json(lambda: cnn.model_status())
        if path == "/api/cnn/train_status":
            return self._send_json({"ok": True, "state": {
                "running": _train_state["running"],
                "started_at": _train_state["started_at"],
                "log": _train_state["log"][-30:],
                "report": _train_state["report"],
                "error": _train_state["error"],
            }})
        if path == "/api/cnn/rules":
            # 定量实验结论(backend/exp_rules.py 生成)
            fp = os.path.join(cnn.DATA_DIR, "exp_rules.json") if cnn else None
            if fp and os.path.isfile(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    return self._send_json({"ok": True, "data": json.load(f)})
            return self._send_json({"ok": False, "error": "exp_rules.json 不存在, 先运行 python3 backend/exp_rules.py"})
        if path == "/api/cnn/mech":
            # 端到端分类机制解剖(backend/mech.py 生成)
            fp = os.path.join(cnn.DATA_DIR, "mech.json") if cnn else None
            if fp and os.path.isfile(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    return self._send_json({"ok": True, "data": json.load(f)})
            return self._send_json({"ok": False, "error": "mech.json 不存在, 先运行 python3 backend/mech.py"})
        # 静态文件(仅限 frontend 目录内)
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        fp = os.path.normpath(os.path.join(FRONTEND, rel))
        if not fp.startswith(FRONTEND + os.sep) and fp != os.path.join(FRONTEND, "index.html"):
            return self._send_error_json("forbidden", 403)
        if not os.path.isfile(fp):
            return self._send_error_json("not found", 404)
        ext = os.path.splitext(fp)[1].lower()
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _cnn_json(self, fn):
        if not CNN_OK:
            return self._send_error_json(
                "CNN 模块需要 numpy: 请以 CNN_PYTHONPATH=<numpy所在目录> 启动后端", 503)
        try:
            return self._send_json({"ok": True, "data": fn()})
        except ValueError as e:
            return self._send_error_json(str(e))

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/analyze":
                body = self._read_body()
                k = _validate_kernel(body.get("kernel"))
                t0 = time.time()
                result = km.classify(k)
                ms = round((time.time() - t0) * 1000)
                return self._send_json({"ok": True, "ms": ms, "result": result})
            if path == "/api/parametric":
                body = self._read_body()
                family = str(body.get("family", ""))
                m = int(body.get("m", 9))
                params = body.get("params") or {}
                if m % 2 == 0 or not (3 <= m <= 33):
                    raise ValueError("m 需为 3..33 的奇数")
                kernel, note = km.parametric_kernel(family, m, params)
                return self._send_json({"ok": True, "kernel": kernel, "note": note})
            if path == "/api/random":
                body = self._read_body()
                m = int(body.get("m", 9))
                kind = str(body.get("kind", "general"))
                intensity = float(body.get("intensity", 1.0))
                seed = body.get("seed")
                if m % 2 == 0 or not (3 <= m <= 33):
                    raise ValueError("m 需为 3..33 的奇数")
                kernel = km.random_kernel(m, kind, intensity, seed)
                return self._send_json({"ok": True, "kernel": kernel})
            if path == "/api/cnn/predict":
                if not CNN_OK:
                    return self._send_error_json("CNN 模块需要 numpy 环境", 503)
                body = self._read_body()
                pix = body.get("pixels")
                if not isinstance(pix, list) or len(pix) != 28 or \
                        not all(isinstance(r, list) and len(r) == 28 for r in pix):
                    raise ValueError("pixels 需为 28×28 数值数组")
                probs, pred = cnn.predict_pixels(pix)
                return self._send_json({"ok": True, "probs": probs, "pred": pred})
            if path == "/api/cnn/layers":
                return self._cnn_json(cnn.layer_analyses)
            if path == "/api/cnn/train":
                if not CNN_OK:
                    return self._send_error_json("CNN 模块需要 numpy 环境", 503)
                if _train_state["running"]:
                    return self._send_error_json("训练已在进行中")
                body = self._read_body()
                params = {
                    "subset": int(body.get("subset", 20000)),
                    "epochs": int(body.get("epochs", 3)),
                    "batch": int(body.get("batch", 64)),
                    "lr": float(body.get("lr", 2e-3)),
                    "seed": int(body.get("seed", 2026)),
                }
                if not (1000 <= params["subset"] <= 60000):
                    raise ValueError("subset 需在 1000..60000")
                _train_state.update(running=True, started_at=time.time(),
                                    log=["开始训练 subset=%d epochs=%d" % (
                                        params["subset"], params["epochs"])],
                                    report=None, error=None)
                t = threading.Thread(target=_cnn_train_worker, args=(params,), daemon=True)
                t.start()
                return self._send_json({"ok": True, "started": True})
            if path == "/api/cnn/train_status":
                return self._send_json({"ok": True, "state": {
                    "running": _train_state["running"],
                    "started_at": _train_state["started_at"],
                    "log": _train_state["log"][-30:],
                    "report": _train_state["report"],
                    "error": _train_state["error"],
                }})
            return self._send_error_json("unknown endpoint: " + path, 404)
        except (ValueError, json.JSONDecodeError) as e:
            return self._send_error_json(str(e))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send_error_json("internal error: " + repr(e), 500)

    def log_message(self, fmt, *args):  # noqa: A003
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else "8017"))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"算子生成与分析工作台: http://{host}:{port}")
    print(f"静态前端目录: {FRONTEND}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
