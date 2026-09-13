#!/usr/bin/env bash
# 启动算子生成与分析工作台
# 用法: bash run.sh            # 默认 http://127.0.0.1:8017
#       PORT=9000 bash run.sh  # 自定义端口
set -uo pipefail
cd "$(dirname "$0")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8017}"
echo "[run.sh] 算子生成与分析工作台 → http://${HOST}:${PORT}"
echo "[run.sh] 后端为纯 Python 标准库(零依赖)。Ctrl+C 停止。"
exec python3 backend/main.py
