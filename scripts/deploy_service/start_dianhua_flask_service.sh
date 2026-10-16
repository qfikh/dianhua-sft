#!/bin/bash
LOG_DIR="./logs"
mkdir -p $LOG_DIR

. venv_sft_py311/bin/activate

# 日志时间戳使用北京时间（gunicorn/logging 取 localtime）
export TZ=Asia/Shanghai

# 停掉旧服务
# pkill -f "dianhua_sft_interface" 2>/dev/null
# sleep 2
# echo "旧服务已停止"

# 启动新服务
LOG_FILE="$LOG_DIR/dianhua_flask_$(date +%Y%m%d).log"
nohup gunicorn \
    -w 4 \
    -b 0.0.0.0:5011 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    dianhua_sft_interface:app >> $LOG_FILE 2>&1 &

echo "新服务已启动，日志: $LOG_FILE"