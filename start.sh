#!/usr/bin/env bash
# ============================================================================
# 启动/停止 CNN 演示服务
#   bash start.sh          启动主服务(实验室 + CNN 分析 + 演示台), 端口 8017
#   bash start.sh board    额外启动训练进度看板, 端口 8020
#   bash start.sh stop     停止
#   bash start.sh status   查看状态
# 说明: 服务以 setsid + nohup 完全脱离当前终端/会话, 日志写入 logs/,
#       监听 0.0.0.0 以便 Windows 浏览器通过 localhost 或 WSL IP 访问。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
PORT="${PORT:-8017}"
BOARD_PORT="${BOARD_PORT:-8020}"
export CNN_PYTHONPATH="${CNN_PYTHONPATH:-/home/dunaandone/dsvideo/pylib}"
mkdir -p logs

wsl_ip() { hostname -I 2>/dev/null | awk '{print $1}'; }

stop_one() { # $1=pidfile $2=name
  local pf="logs/$1.pid"
  if [[ -f "$pf" ]]; then
    local pid; pid=$(cat "$pf")
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null; echo "已停止 $2 (pid $pid)"; else echo "$2 未在运行"; fi
    rm -f "$pf"
  else
    echo "$2 无 pid 记录"
  fi
}

start_main() {
  if [[ -f logs/server.pid ]] && kill -0 "$(cat logs/server.pid)" 2>/dev/null; then
    echo "主服务已在运行 (pid $(cat logs/server.pid))"; return 0
  fi
  setsid nohup env HOST=0.0.0.0 PORT="$PORT" CNN_PYTHONPATH="$CNN_PYTHONPATH" \
    "$PY" backend/main.py >> logs/server.log 2>&1 < /dev/null &
  echo $! > logs/server.pid
  sleep 2
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/api/health" >/dev/null; then
    echo "✔ 主服务已启动 (pid $(cat logs/server.pid))"
  else
    echo "✘ 主服务启动失败, 见 logs/server.log"; tail -5 logs/server.log
  fi
}

start_board() {
  if [[ -f logs/board.pid ]] && kill -0 "$(cat logs/board.pid)" 2>/dev/null; then
    echo "训练看板已在运行 (pid $(cat logs/board.pid))"; return 0
  fi
  # 注意: 看板训练的输出写到独立文件, 避免覆盖演示页正在用的 data/cnn_torch.json
  setsid nohup "$PY" backend/train_web.py --host 0.0.0.0 --port "$BOARD_PORT" \
    --out "${BOARD_OUT:-data/cnn_board.json}" >> logs/board.log 2>&1 < /dev/null &
  echo $! > logs/board.pid
  sleep 2
  if curl -s --max-time 5 "http://127.0.0.1:$BOARD_PORT/api/status" >/dev/null; then
    echo "✔ 训练看板已启动 (pid $(cat logs/board.pid))"
  else
    echo "✘ 看板启动失败, 见 logs/board.log"; tail -5 logs/board.log
  fi
}

urls() {
  local ip; ip=$(wsl_ip)
  echo
  echo "  实验室(卷积核/设计空间)  http://127.0.0.1:$PORT/            http://$ip:$PORT/"
  echo "  CNN 分析(层/机制/规则)   http://127.0.0.1:$PORT/cnn.html     http://$ip:$PORT/cnn.html"
  echo "  ★ CNN 演示台(手写识别)    http://127.0.0.1:$PORT/demo/        http://$ip:$PORT/demo/"
  [[ -f logs/board.pid ]] && echo "  训练进度看板              http://127.0.0.1:$BOARD_PORT/          http://$ip:$BOARD_PORT/"
  echo
  echo "  提示: Windows 浏览器若 127.0.0.1 打不开, 就用上面的 WSL IP ($ip) 地址;"
  echo "        都不通时检查 Windows 防火墙是否放行 WSL, 或执行 wsl --shutdown 后重开(需重新运行本脚本)。"
}

case "${1:-start}" in
  start)  start_main; urls ;;
  board)  start_main; start_board; urls ;;
  stop)   stop_one server "主服务"; stop_one board "训练看板" ;;
  status)
    for n in server board; do
      pf="logs/$n.pid"
      if [[ -f $pf ]] && kill -0 "$(cat $pf)" 2>/dev/null; then echo "$n: 运行中 (pid $(cat $pf))"; else echo "$n: 未运行"; fi
    done
    ss -ltnp 2>/dev/null | grep -E ":$PORT|:$BOARD_PORT" || true
    ;;
  restart) stop_one server "主服务"; stop_one board "训练看板"; sleep 1; start_main; urls ;;
  *) echo "用法: bash start.sh [start|board|stop|status|restart]"; exit 1 ;;
esac
