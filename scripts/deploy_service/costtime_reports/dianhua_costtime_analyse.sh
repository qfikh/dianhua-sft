#!/bin/bash

# 1. 切换到工作目录（解决 crontab 默认在家目录执行的问题）
cd /data1/xujianjun/workspace/dianhua_sft/costtime_reports || exit 1

# 2. 激活 Python 虚拟环境
. ../venv_sft_py311_v2/bin/activate


# 3. 执行 Python 分析脚本
python3 dianhua_costtime_analyse.py >> dianhua_costtime.log 2>&1
cp dianhua_costtime.log /data1/xiaohy/file_8003/file/dianhua_costtime.log

# 可选：打印执行时间，方便日志排查
# echo "============================[$(date '+%Y-%m-%d %H:%M:%S')] 脚本执行完成============================" >> dianhua_costtime.log 2>&1






# 定时任务

# 0 8 * * * sh /data1/xujianjun/workspace/dianhua_sft/costtime_reports/dianhua_costtime_analyse.sh 
