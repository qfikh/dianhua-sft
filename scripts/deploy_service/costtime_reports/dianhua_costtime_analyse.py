#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dianhua 服务每日耗时统计

每天 08:30 由 cron 触发，统计前一天 vllm 调用耗时，输出 Excel + 一行汇总。
日志按天滚动（dianhua_flask_YYYYMMDD.log），每行是 dict 字面量，含 vllm_latency 字段。
"""

import argparse
import ast
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.getenv("DIANHUA_ROOT_DIR", "/data1/xujianjun/workspace/dianhua_sft")
if not os.path.isdir(ROOT_DIR):
    ROOT_DIR = SCRIPT_DIR

LOG_DIR = os.getenv("DIANHUA_LOG_DIR", os.path.join(ROOT_DIR, "logs"))
OUTPUT_DIR = os.getenv("DIANHUA_COSTTIME_OUTPUT_DIR", os.path.join(ROOT_DIR, "costtime_reports"))
LOG_TZ = timezone(timedelta(hours=8))
COLUMNS = ["时间戳", "Session ID", "垫话内容", "垫话服务耗时(s)"]


def parse_log_file(path, target_day):
    """解析目标日日志文件，返回 vllm 耗时记录列表"""
    prefix = target_day.strftime("%Y-%m-%d")
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if "vllm_latency" not in line or prefix not in line:
                continue
            start = line.find("{")
            if start < 0:
                continue
            try:
                data = ast.literal_eval(line[start:].rstrip())
            except (ValueError, SyntaxError):
                continue
            if not str(data.get("request_time", "")).startswith(prefix):
                continue
            vllm = data.get("vllm_latency")
            if vllm is None:
                continue
            records.append({
                "时间戳": data["request_time"],
                "Session ID": data.get("session_id", ""),
                "垫话内容": data.get("final_dianhua", ""),
                "垫话服务耗时(s)": float(vllm),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="统计 dianhua 服务指定日期的 vllm 调用耗时")
    parser.add_argument("date", nargs="?", help="统计日期 YYYY-MM-DD，默认昨天")
    parser.add_argument("-d", "--log-dir", default=LOG_DIR)
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    target_day = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date \
        else datetime.now(LOG_TZ).date() - timedelta(days=1)

    log_path = os.path.join(args.log_dir, "dianhua_flask_%s.log" % target_day.strftime("%Y%m%d"))
    if not os.path.isfile(log_path):
        return 1

    records = parse_log_file(log_path, target_day)
    df = pd.DataFrame(records, columns=COLUMNS)
    count = len(df)
    avg = round(df["垫话服务耗时(s)"].mean(), 4) if count else 0.0

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "dianhua_costtime_%s.xlsx" % target_day.strftime("%Y%m%d"))
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="明细", index=False)
        pd.DataFrame([{"日期": target_day.strftime("%Y-%m-%d"), "请求数": count,
                       "垫话服务平均耗时(s)": avg}]).to_excel(writer, sheet_name="汇总", index=False)

    now = datetime.now(LOG_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now} 统计的 {target_day.strftime('%Y-%m-%d')}的 平均垫话服务耗时: {avg:.4f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
