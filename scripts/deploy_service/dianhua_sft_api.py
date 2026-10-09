import requests
import re
import time
import os
import random
from flask import Flask, request, Response
import json
# from redis_helper import redis_get, redis_set

def jsonify(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype="application/json")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:5010")

# 阿里云 Qwen API 配置
DASHSCOPE_API_KEY = "sk-ws-H.EPHYEIR.ntZi.MEQCIDWSE2d903v9QtcCKKTGw5d08GWIx19AxrcIjZlGnAo-AiBC1JXSHg4-3kMZSt0NcbsVtrIWCE9jgAyYwR_yZbkkJg"
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.6-flash")

SYSTEM_PROMPT_V30_0810 = """
你是专业汽车客服话术指导师，核心任务：为客户每一段完整表述生成场景适配精准、分布绝对均衡、高度多样化的承接垫话，严格控制"我XX您的XX"类句式在合理区间，绝对禁止单个垫话过度集中使用，确保自然过渡、专业得体。

一、垫话定义
垫话：坐席正式解答客户问题前，用于承接客户表述、体现专业礼貌、平稳自然过渡至正式答复的规范承接语，不包含实际业务解答内容。禁止使用开场语、结束语等非承接类用语。

二、垫话判断核心依据
1. 优先原则：标注时以客户最后一句话为核心判断依据
2. 完整表述判定：若客户最后一句话语义完整、能够独立表达明确意图，仅参照最后一句话进行判断
3. 不完整表述判定：若客户最后一句话语义残缺、无法独立表达明确意图，需结合最后一句话及上文所有客户表述内容综合判断
4. 兜底原则：若通过上述方式仍无法明确判断客户表述所属类型，必须直接使用简短通用垫话处理

三、标准化标注操作步骤
1. 文本预处理：通读完整对话文本，严格区分客户表述与坐席表述，精准定位客户的最后一句表述
2. 依据判定：按照"垫话判断核心依据"，确定本次标注的参照范围（仅最后一句 / 最后一句 + 上文）
3. 垫话选择优先级与场景适配规则（必须严格遵守）：
  - 第一优先级（占比50%-60%）：从【高优先级简短通用垫话库】中均匀随机选择合适表达
    - 客户表述简短（10字以内）、意图明确：优先使用单字/两字垫话
    - 客户表述中等长度（10-30字）、内容清晰：优先使用组合式垫话
    - 客户表述较长（30字以上）、信息较多：优先使用稍长组合垫话
  - 第二优先级（占比20%-25%）：对照"垫话类型匹配规则"判断客户表述所属类型，使用对应的"我XX您的XX"类句式
    - 适用场景：客户表述包含明确的需求、问题描述、情绪表达或顾虑担忧时
    - 多样性要求：同批次标注中，"我XX您的XX"类句式内部必须均匀使用所有允许的表达，不得长期集中使用某1-2个
  - 第三优先级（占比5%-15%）：使用"百搭+简短结构"组合
4. 基础规则校验：逐项核对生成的垫话是否符合以下所有规则要求：
  - 格式与长度要求（1-12字，必须带标点）
  - 句式要求（仅陈述式，无问句、反问句、感叹句）
  - 禁止内容要求（绝对禁用话术、观点类词汇、语气词等）
  - 不与客户最后一句内容重复
5. 严格的多样性与分布校验（必须执行）：
  - 检查生成的垫话与同批次已生成的其他垫话是否存在句式、结构、用词重复，若存在重复则重新生成
  - 同批次标注中，单个垫话的使用次数不得超过总数量的12%（彻底解决"好的。"占比过高问题）
  - 确保高优先级简短通用垫话库中所有15个选项都有被使用，不得遗漏
  - 确保"我XX您的XX"类句式内部所有8个允许的表达都有被使用，不得遗漏
  - 严格控制各优先级占比在规定区间内，不得超出上下限
6. 结果输出：严格按照指定输出格式，直接输出唯一符合要求的垫话结果

四、高优先级简短通用垫话库（必须均匀使用所有选项）
嗯。/ 嗯嗯。/好的。/了解。

五、垫话类型匹配规则（第二优先级使用，必须均匀使用所有表达）
1. 客户对话为需求类、疑问类、表态类、指示类：可垫【意思 / 想法】相关内容。
2. 客户对话为客户描述类：可垫【情况 / 问题】相关内容。
3. 客户对话为情绪 / 不满类：可垫【心情 / 感受】相关内容。
4. 客户对话为担心 / 害怕类：可垫【顾虑 / 担忧】相关内容。
  
六、可使用的垫话句子结构
- 助词：嗯
- 动词+助词：明白了 / 好的
- 主谓：我明白、我了解
- 主谓+助词：我明白了。
- 主谓+定宾+助词：我明白您的意思了（仅第二优先级使用）
- 允许使用上述结构的倒装句
- 允许使用高优先级简短通用垫话库中的任意组合
  
七、格式与长度要求
有效垫话控制1–12字，必须带标点。仅使用陈述式承接语，严禁问句、反问句、感叹句。
  
八、输出格式
直接输出唯一结果，无多余文字、无序号、无解释、无格式，干净返回。
"""

# 售前咨询(车型配置、车型对比、价格)三个场景下加长垫话
pre_sales_cushions_v2 = ['收到，我马上为您查询相关信息。','正在查询相关内容，请您稍等','我帮您看一下，请稍等。','您稍等，我帮您查一查。','您稍等，我去后台查一下。','我正在为您查询，请稍候。','请稍候，我正在为您核实相关信息。','麻烦您稍等，我帮您查一下。','请稍等，我帮您仔细看下。','请稍等一下，我帮您核实下。','请您稍等，我来确认一下。','您稍等，我帮您核实一下。','请稍等，我核实一下。','我帮您核实一下。','我这就帮您看看，请稍等。','请您稍等一下，我来帮您确认。','正在查找相关信息，麻烦您等一下。','请稍等，我来帮您查查。']

# 垫话去重时的词语替换规则 (old, new)
DIANHUA_REPLACEMENTS = [
    ("嗯，", "嗯嗯，"),
    ("了解了。", "明白了。"),
    ("了解。", "明白。"),
    ("，好，", "，好的，"),
    ("我帮您看一下", "我帮您确认一下")
]

app = Flask(__name__)
import logging
app.logger.setLevel(logging.INFO)

def _clean_think_tags(text: str) -> str:
    """清理空的 <think> 标签
    如果 <think></think> 标签内部没有内容，则移除标签
    """
    # 移除空的 <think></think> 标签（包括空白内容）
    text = re.sub(r'<think>\s*</think>', '', text, flags=re.DOTALL)
    # 移除其他可能的情况：<think>\n</think> 等
    text = re.sub(r'<think>[\s\n]*</think>', '', text)
    # 移除所有残留的<*>标签
    text = re.sub(r'<.*?>', '', text)
    return text.strip()


def _call_vllm(dialog_text: str, timeout: int = 30) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V30_0810},
        {"role": "user", "content": dialog_text}
    ]
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": False,
        "frequency_penalty": 0.8,
        "presence_penalty": 0.6,
        "enable_thinking": False,
    }
    resp = requests.post(f"{DASHSCOPE_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    result = _clean_think_tags(resp.json()["choices"][0]["message"]["content"])

    return result


def merge_consecutive_utterances(window):
    """
    对滑动窗口内的连续同角色语句进行合并
    """
    if not window:
        return []

    merged = []
    current_role = None
    current_content = ""

    for utterance in window:
        if utterance.startswith("客服："):
            role = "客服"
            content = utterance[len("客服："):].strip()
        elif utterance.startswith("客户："):
            role = "客户"
            content = utterance[len("客户："):].strip()
        else:
            continue

        if role == current_role:
            current_content += " " + content
        else:
            if current_role:
                merged.append(f"{current_role}：{current_content}")
            current_role = role
            current_content = content

    if current_role:
        merged.append(f"{current_role}：{current_content}")

    return merged


@app.route("/generate", methods=["POST"])
def generate():
    start_total = time.time()  # 整个方法开始时间
    data = request.get_json(force=True)

    source = data.get("source", "unknown")
    session_id = data.get("session_id", "unknown")
    dialog_history = data.get("dialog_history", [])
    # 将dialog_history 中的"Client:"替换为 "客户："，"Service:"替换为 "客服："
    dialog_history = [utterance.replace("Client:", "客户：").replace("Service:", "客服：") for utterance in dialog_history]
    # 合并连续同角色语句
    dialog_history = merge_consecutive_utterances(dialog_history)
    dialog_history = dialog_history[-10:]  # 只保留最近n轮对话，减少输入长度，提升响应速度

    # 无需垫话的情况：对话为空、最后一句是客服说的
    if not dialog_history or dialog_history[-1].startswith("客服："):
        return jsonify({
            "code": 200,
            "message": "对话历史为空或最后一句话是客服说的，无需垫话",
            "data": {"source": source, "session_id": session_id, "dianhua": ""}
        })

    # 获取最后一句客服的话
    last_cushion = ""
    for utterance in reversed(dialog_history):
        if utterance.startswith("客服："):
            last_cushion = utterance[len("客服："):].strip()
            # 只保留开头到第一个句号的部分（垫话部分）
            first_period = last_cushion.find("。")
            if first_period != -1:
                last_cushion = last_cushion[:first_period + 1]
            break

    conversation = "\n".join(dialog_history)
    try:
        start_vllm = time.time()  # vllm调用开始时间
        ori_dianhua = _call_vllm(conversation)

        vllm_latency = time.time() - start_vllm  # vllm调用耗时
        total_latency = time.time() - start_total  # 整个方法耗时

        # 如果vllm返回的垫话包含"我帮您查一下"，则替换成随机查询等待场景垫话
        if "查一下" in ori_dianhua:
            dianhua = ori_dianhua.split("，")[0] + "，" + random.choice(pre_sales_cushions_v2)
        else:
            dianhua = ori_dianhua

        # 去重：与该 session 最近一次垫话比较，相同则随机替换
        if len(dianhua) > 3 and dianhua in last_cushion:
            for old, new in DIANHUA_REPLACEMENTS:
                if old in dianhua:
                    dianhua = dianhua.replace(old, new)
                    break
                if new in dianhua:
                    dianhua = dianhua.replace(new, old)
                    break

        # 打印监控日志
        current_time = time.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(time.time() * 1000) % 1000:03d}"
        app.logger.info(f" {current_time}, source：{source}, Session ID：{session_id}, 上文对话内容: {'  '.join(dialog_history)}, 原始模型垫话内容：{ori_dianhua} >>>>> 修改后垫话内容：{dianhua}, generate方法总耗时: {total_latency:.3f}s, vllm调用耗时: {vllm_latency:.3f}s")

        return jsonify({
            "code": 200,
            "message": "success",
            "data": {
                "source": source,
                "session_id": session_id,
                "dianhua": dianhua,
                "latency": round(total_latency, 3)
            }
        })
    except Exception as e:
        app.logger.error("vLLM调用失败: %s", e)
        return jsonify({
            "code": 500,
            "message": str(e),
            "data": {"source": source, "session_id": session_id, "dianhua": ""}
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "code": 200,
        "message": "success",
        "data": {"status": "ok"}
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5011))
    app.run(host="0.0.0.0", port=port)

# 对应 V29优化短垫话、重复垫话的问题