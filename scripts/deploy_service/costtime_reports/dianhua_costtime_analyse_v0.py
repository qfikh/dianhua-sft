#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dianhua 服务每日耗时统计

每天 08:30 由 cron 触发，统计「前一天 00:00:00 ~ 23:59:59」每个请求的
vllm 调用耗时，输出 Excel：dianhua_costtime_YYYYMMDD.xlsx（明细页 + 汇总页）。

日志文件定位说明：
    日志按「服务启动日期」命名（dianhua_flask_YYYYMMDD.log），且以 >> 追加写入，
    因此服务不重启时同一个文件会持续累积多天日志（例如 dianhua_flask_20261012.log
    可能包含 10-12 之后的所有日志）。凡「文件名日期 <= 统计日期」的日志文件都可能含
    该日日志（当天重启会把日志切分到新、旧两个文件），故全部读取，再按行内时间戳过滤。

部署（脚本放在 /data1/xujianjun/workspace/dianhua_sft/ 下）：
    根目录已写死为 /data1/xujianjun/workspace/dianhua_sft，日志目录默认 = 根目录/logs，
    因此脚本放在该目录的任意层级都能正常执行；本地或其他环境可用环境变量
    DIANHUA_ROOT_DIR / DIANHUA_LOG_DIR / DIANHUA_COSTTIME_OUTPUT_DIR 覆盖。

用法：
    python trace_costtime_analyse.py                  # 统计昨天（北京时间）
    python trace_costtime_analyse.py 2026-10-15       # 统计指定日期（补数）
    python trace_costtime_analyse.py 2026-10-15 -d /path/to/logs -o ./reports

cron（服务器每天 08:30 执行，脚本内部为绝对路径，无需 cd）：
    30 8 * * * /data1/xujianjun/workspace/dianhua_sft/venv_sft_py311_v2/bin/python \\
        /data1/xujianjun/workspace/dianhua_sft/trace_costtime_analyse.py >> \\
        /data1/xujianjun/workspace/dianhua_sft/logs/costtime_cron.log 2>&1
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 部署根目录：服务器上直接写死，脚本放在任意位置都能正确找到 logs/；可用环境变量覆盖。
# 目录不存在时（本地调试等）回退到脚本所在目录。
ROOT_DIR = os.getenv("DIANHUA_ROOT_DIR", "/data1/xujianjun/workspace/dianhua_sft")
if not os.path.isdir(ROOT_DIR):
    ROOT_DIR = SCRIPT_DIR

LOG_DIR = os.getenv("DIANHUA_LOG_DIR", os.path.join(ROOT_DIR, "logs"))
LOG_GLOB = os.getenv("DIANHUA_LOG_GLOB", "dianhua_flask_*.log")
OUTPUT_DIR = os.getenv("DIANHUA_COSTTIME_OUTPUT_DIR", os.path.join(ROOT_DIR, "costtime_reports"))

# 日志时间戳固定为北京时间（start 脚本 export TZ=Asia/Shanghai），此处保证「昨天」也按北京时间算
LOG_TZ = timezone(timedelta(hours=8))

# 从文件名中解析服务启动日期，如 dianhua_flask_20261012.log / dianhua_flask_v2_20261012.log
LOG_NAME_RE = re.compile(r"^dianhua_flask_(?:v\d+_)?(\d{8})\.log$")

# 单条业务日志（INFO in dianhua_sft_interface）的字段提取
LINE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}\]"  # 行首时间戳
    r".*?Session ID：(?P<sid>[^,]+),"                          # Session ID
    r".*?修改后垫话内容：(?P<dianhua>[^,]*),"                    # 垫话内容（以半角逗号收尾）
    r".*?vllm调用耗时:\s*(?P<vllm>[\d.]+)s"                    # vllm 调用耗时
)

COLUMNS = ["时间戳", "Session ID", "垫话内容", "垫话服务耗时(s)"]


def parse_args():
    parser = argparse.ArgumentParser(description="统计 dianhua 服务指定日期（默认昨天）的 vllm 调用耗时")
    parser.add_argument("date", nargs="?", help="统计日期，格式 YYYY-MM-DD，默认昨天")
    parser.add_argument("-d", "--log-dir", default=LOG_DIR, help=f"日志目录，默认 {LOG_DIR}")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR, help=f"输出目录，默认 {OUTPUT_DIR}")
    return parser.parse_args()


def resolve_target_date(raw):
    """未传日期时默认统计「昨天」（按北京时间）"""
    if not raw:
        return datetime.now(LOG_TZ).date() - timedelta(days=1)
    return datetime.strptime(raw, "%Y-%m-%d").date()


def pick_log_files(log_dir, target_day):
    """挑选待扫描的日志文件：文件名日期 <= 目标日期的全部文件。

    不能只取最新的一个：日志文件名只精确到天，若服务在目标日当天重启，目标日的日志会被
    切分到新、旧两个文件（旧文件为 00:00 至重启前，新文件为重启后），只读最新文件会漏数据。
    解析不到日期的文件一并纳入；全部解析不到时兜底返回全部。
    """
    files = sorted(glob.glob(os.path.join(log_dir, LOG_GLOB)))
    dated, undated = [], []
    for path in files:
        match = LOG_NAME_RE.match(os.path.basename(path))
        if match:
            dated.append((datetime.strptime(match.group(1), "%Y%m%d").date(), path))
        else:
            undated.append(path)
    eligible = [path for day, path in sorted(dated) if day <= target_day]
    return eligible + undated if eligible or undated else files


def parse_log_file(path, target_day):
    """按行解析日志，仅保留目标日期且成功提取到 vllm 耗时的记录"""
    prefix = target_day.strftime("%Y-%m-%d")
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "vllm调用耗时" not in line or prefix not in line:
                continue
            match = LINE_RE.match(line)
            if not match or not match.group("ts").startswith(prefix):
                continue
            group = match.groupdict()
            records.append(
                {
                    "时间戳": group["ts"],
                    "Session ID": group["sid"].strip(),
                    "垫话内容": group["dianhua"].strip(),
                    "垫话服务耗时(s)": float(group["vllm"]),
                }
            )
    return records


def write_excel(records, target_day, output_dir):
    """输出明细页 + 汇总页，返回 (文件路径, 汇总字典)"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(records, columns=COLUMNS)
    count = len(df)
    summary = {
        "日期": target_day.strftime("%Y-%m-%d"),
        "请求数": count,
        "垫话服务平均耗时(s)": round(df["垫话服务耗时(s)"].mean(), 4) if count else 0.0,
    }
    xlsx_path = os.path.join(output_dir, f"dianhua_costtime_{target_day.strftime('%Y%m%d')}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="明细", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="汇总", index=False)
    return xlsx_path, summary


def main():
    args = parse_args()
    target_day = resolve_target_date(args.date)

    files = pick_log_files(args.log_dir, target_day)
    if not files:
        return 1

    records = []
    for path in files:
        records.extend(parse_log_file(path, target_day))

    # 兜底：选定的日志没解析到记录时，扫描目录下其余日志（应对命名/轮转异常）
    if not records:
        others = [f for f in sorted(glob.glob(os.path.join(args.log_dir, LOG_GLOB))) if f not in files]
        for path in others:
            records.extend(parse_log_file(path, target_day))

    _, summary = write_excel(records, target_day, args.output_dir)
    now = datetime.now(LOG_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} 统计的 {target_day.strftime('%Y-%m-%d')}的 平均垫话服务耗时: {summary['垫话服务平均耗时(s)']:.4f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
