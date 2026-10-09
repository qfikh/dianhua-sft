#!/bin/bash
LOG_DIR="./logs"
mkdir -p $LOG_DIR

. venv_sft_py311/bin/activate

# 按天生成日志文件名（每次启动时计算）
LOG_FILE="$LOG_DIR/dianhua_flask_v2_$(date +%Y%m%d).log"

nohup gunicorn \
    -w 2 \
    -b 0.0.0.0:5013 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    dianhua_sft_interface_v2:app >> $LOG_FILE 2>&1 &
