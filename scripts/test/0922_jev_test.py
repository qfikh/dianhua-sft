#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JEV (Typesafe AI) 接口测试脚本

封装 https://api.typesafe.ai/v1/systemone 调用，
支持 choice / noul / score 等题型，方便批量测试不同 case。
"""

import os
import json
import requests


# ============ 配置 ============
API_URL = "https://api.typesafe.ai/v1/systemone"
API_TOKEN = "apikey_230177e4c04ff704620b5999691d3b4bc57_30daf30960366f155121945476e85505f402aabfd0e30b635b6d7272b015ccb6"
DEFAULT_MODEL = "jev-latest"
TIMEOUT = 60


def call_jev(
    state: str,
    questions: dict,
    model: str = DEFAULT_MODEL,
    token: str = None,
) -> dict:
    """
    调用 JEV 接口，返回完整的 JSON 响应。

    Args:
        state: 用户输入的文本（如客户对话内容、投诉内容等）
        questions: 问题定义字典，key 为问题名，value 为题目的配置
            - type: "choice" | "noul" | "score"
            - instructions: 问题描述
            - criteria: 选项（choice 是 dict，score 是 list，noul 不需要）
        model: 模型名，默认 jev-latest
        token: API Token，不传则从环境变量 TYPESAFE_API_TOKEN 读取

    Returns:
        接口返回的 JSON 字典
    """
    if not token:
        token = API_TOKEN
    if not token:
        raise ValueError(
            "未设置 API Token，请设置环境变量 TYPESAFE_API_TOKEN "
            "或在调用时传入 token 参数"
        )

    payload = {
        "state": state,
        "model": model,
        "questions": questions,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = requests.post(API_URL, json=payload, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ============ 快捷题型构造 ============
def make_choice(name: str, instructions: str, criteria: dict) -> tuple:
    """构造一道选择题（返回 key, value 元组，方便用 dict() 组装）"""
    return name, {
        "type": "choice",
        "instructions": instructions,
        "criteria": criteria,
    }


def make_noul(name: str, instructions: str) -> tuple:
    """构造一道 noul 题（是/否类）"""
    return name, {
        "type": "noul",
        "instructions": instructions,
    }


def make_score(name: str, instructions: str, criteria: list) -> tuple:
    """构造一道打分/排序题"""
    return name, {
        "type": "score",
        "instructions": instructions,
        "criteria": criteria,
    }


# ============ 测试示例 ============
def demo_single_case():
    """示例：测试单个 case（与 curl 示例一致）"""
    state = "我昨天买的耳机，今天收到后发现左耳完全没有声音。已经换了两台手机测试，问题还是一样。我不想换货了。另外，你们客服一直没人回复。"

    questions = dict([
        make_choice(
            name="处理",
            instructions="客户主要希望如何处理？",
            criteria={
                "退款": "退款",
                "换货": "换货",
                "维修": "维修",
                "咨询": "咨询",
                "其他": "不是 退款、换货、维修、咨询 这些",
            },
        ),
        make_noul(
            name="情绪",
            instructions="客户的情绪是负面的么？",
        ),
        make_score(
            name="分配",
            instructions="应该分配给哪个团队？",
            criteria=["售前咨询", "售后支持", "物流支持"],
        ),
    ])

    result = call_jev(state=state, questions=questions)
    print("=" * 60)
    print("📝 输入:", state[:80] + "..." if len(state) > 80 else state)
    print("-" * 60)
    print("📊 结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    # 使用前请先设置环境变量:
    #   export TYPESAFE_API_TOKEN="你的token"
    demo_single_case()
