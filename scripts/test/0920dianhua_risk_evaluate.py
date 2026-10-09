#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估测试题垫话生成脚本

读取附件5-评估测试题.xlsx 中的「应拒答测试题库」「非拒答测试题库」两个 sheet，
取「题目」列，调用垫话生成接口，将结果写回每个 sheet 的最后一列。
"""

import requests
import time
import os
import sys

import openpyxl
from openpyxl.utils import get_column_letter


# ============ 配置 ============
INPUT_XLSX = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0915 垫话安全围栏/附件5-评估测试题.xlsx"
OUTPUT_XLSX = os.path.join( os.path.dirname(INPUT_XLSX), "附件5-评估测试题_生成结果_V3_1009.xlsx" )

API_URL = "http://192.168.86.3:5011/generate"
SOURCE = "evaluate_test"
TIMEOUT = 30
MAX_RETRIES = 2
REQUEST_INTERVAL = 0.1  # 每次请求间隔秒数

# 客服开场白（放在题目之前，构造合法对话历史）
SERVICE_OPENING = "客服：您好，我是梅赛德斯奔驰的官方客服，请问有什么可以帮您？"

SHEET_NAMES = ["应拒答测试题库", "非拒答测试题库"]
QUESTION_COL_NAME = "题目"
RESULT_COL_NAME = "生成垫话"


# ============ 工具函数 ============
def call_generate_api(question: str, session_id: str = "eval") -> str:
    """调用垫话生成接口，返回垫话内容（失败返回空字符串）"""
    dialog_history = [SERVICE_OPENING, f"客户：{question}"]
    payload = {
        "source": SOURCE,
        "session_id": session_id,
        "dialog_history": dialog_history,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 200:
                return data.get("data", {}).get("dianhua", "")
            else:
                return f"[ERROR] {data.get('message', '')}"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            return f"[EXCEPTION] {str(e)}"
    return ""


def find_col_by_name(ws, col_name: str, header_row: int = 1) -> int:
    """在指定行中按列名查找列号（1-based），找不到返回 -1"""
    for col_idx in range(1, ws.max_column + 1):
        val = ws.cell(row=header_row, column=col_idx).value
        if val and str(val).strip() == col_name:
            return col_idx
    return -1


def process_sheet(ws):
    """处理单个 sheet：读取题目列 → 批量调用接口 → 写入最后一列"""
    q_col = find_col_by_name(ws, QUESTION_COL_NAME)
    if q_col < 0:
        print(f"  ⚠️  未找到「{QUESTION_COL_NAME}」列，跳过 sheet: {ws.title}")
        return

    # 结果列 = 当前最大列 + 1
    result_col = ws.max_column + 1

    # 写表头
    ws.cell(row=1, column=result_col, value=RESULT_COL_NAME)

    # 收集题目
    rows_questions = []  # (row_idx, question)
    for row_idx in range(2, ws.max_row + 1):
        q = ws.cell(row=row_idx, column=q_col).value
        if q is None or str(q).strip() == "":
            continue
        rows_questions.append((row_idx, str(q).strip()))

    total = len(rows_questions)
    print(f"  📖 共 {total} 条题目，开始调用接口 (每次间隔 {REQUEST_INTERVAL}s)...")

    # 串行调用，每次请求间隔 REQUEST_INTERVAL 秒
    results = {}  # row_idx -> dianhua
    for idx, (row_idx, q) in enumerate(rows_questions):
        try:
            dianhua = call_generate_api(q, f"{ws.title}_{row_idx}")
        except Exception as e:
            dianhua = f"[EXCEPTION] {str(e)}"
        results[row_idx] = dianhua
        print(f"row_idx={row_idx}, dianhua={dianhua}")

        done = idx + 1
        if done % 50 == 0 or done == total:
            print(f"     进度: {done}/{total}")

        # 最后一条不 sleep
        if done < total:
            time.sleep(REQUEST_INTERVAL)

    # 写回Excel
    for row_idx, _ in rows_questions:
        ws.cell(row=row_idx, column=result_col, value=results.get(row_idx, ""))

    print(f"  ✅ sheet [{ws.title}] 处理完成，结果写入第 {result_col} 列 ({get_column_letter(result_col)})")


# ============ 主流程 ============
def main():
    if not os.path.exists(INPUT_XLSX):
        print(f"❌ 文件不存在: {INPUT_XLSX}")
        sys.exit(1)

    print(f"📂 读取文件: {INPUT_XLSX}")
    wb = openpyxl.load_workbook(INPUT_XLSX)

    for sheet_name in SHEET_NAMES:
        if sheet_name not in wb.sheetnames:
            print(f"⚠️  未找到 sheet: {sheet_name}，跳过")
            continue
        print(f"\n========== 处理 sheet: {sheet_name} ==========")
        ws = wb[sheet_name]
        process_sheet(ws)

    print(f"\n💾 保存结果到: {OUTPUT_XLSX}")
    wb.save(OUTPUT_XLSX)
    print("🎉 全部处理完成")


if __name__ == "__main__":
    main()
