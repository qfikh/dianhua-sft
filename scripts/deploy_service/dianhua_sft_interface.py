import requests
import re
import time
import os
import random
from flask import Flask, request, Response
import json
# from redis_helper import redis_get, redis_set

# 系统提示词（所有版本的 prompt）统一从 prompt.py 引入
from prompt import *

def jsonify(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype="application/json")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:6015") # TODO 注意不同环境下这里要改

# 售前咨询(车型配置、车型对比、价格)三个场景下加长垫话
PRE_SALES_CUSHIONS = ['收到，我马上为您查询详情。', '围绕您咨询的问题，请稍候我核查内容。', '对于您的咨询，稍作等候我核实相关内容。', '辛苦短暂等候，我调取资料梳理相关内容。', '查询资料内容，麻烦您等等。', '核对详情数据，我快速整理下。', '我帮您看一下，请稍等。', '短暂等候一下，我调取后台资料进行查询。', '核对相关数据，需要一点时间。', '我马上帮您核对清楚情况，稍等。', '关于您的咨询，麻烦等下我马上核查内容。', '您咨询的问题，请耐心等候我即刻检索。', '麻烦耐心等候，我调取资料梳理相关详情。', '稍候片刻，我检索业务相关资料信息。', '结合您的提问，劳烦稍等我核对相关资料。', '请稍作等候，我查询本次咨询涉及的资料。', '您稍等，我帮您查一查。', '您请放心稍候，我正在仔细查阅中。', '知悉您的咨询，请稍片刻我核查对应内容。', '在线稍候片刻，我查找业务对应的相关资料。', '看到您的问题，请稍等我检索相关内容。', '您刚提出的疑问，麻烦等片刻我检索资料。', '您稍等，我去后台查一下。', '稍候片刻，我检索业务相关资料内容。', '辛苦等候片刻，我调取咨询对应的资料。', '对应您的发问，麻烦等下我调取详细信息。', '您咨询的问题，请稍候我快速调取内容。', '我正在为您查询请稍候。', '查询参考内容，稍等我核对完毕。', '麻烦稍候一小会，我调取后台资料检索。', '麻烦耐心等候，我查询咨询涉及的内容。', '麻烦稍等片刻，我马上为您查询相关内容。', '短暂等候片刻，我查找业务所需相关信息。', '核对数据细则，您稍微等等。', '请稍候，我即刻为您核实相关信息。', '依据您的提问，稍作等待我帮您核实详情。', '您提出的疑问，请等一会我即刻查询信息。', '依照您的咨询，劳烦稍等我核对全部详情。', '麻烦您稍等，我帮您查一下。', '辛苦稍作等候，我调取资料核对咨询内容。', '请稍等，我来给您仔细对一下资料。', '查找原始数据，麻烦您稍等。', '依照您的提问，稍候片刻我帮您查清楚。', '麻烦稍作等候片刻，我查询相关资料。', '请等一下，我帮您仔细看下。', '请稍等一下，我帮您核实下。', '请耐心等候，我即刻调取资料核对相关信息。', '请在线稍候，我检索您咨询的业务资料。', '针对您的疑问，劳烦稍等我马上查阅资料。', '请您稍等，我来确认一下。', '您稍等，我帮您核实。', '稍等片刻不要挂断，我检索后台相关资料。', '烦请稍等，我进入系统查一下。', '查询资料内容，稍等我整理下。', '辛苦耐心等候一会，我查询业务内容。', '有关您的问题，请稍候我为您核实资料。', '稍等哈，我为您快速找一下参考点。', '针对您的问题，请稍等片刻我去查找答案。', '请稍等，我核实一下。', '您的问题我帮您核实一下。', '查询参考内容，还请您稍等。', '您稍等，我去系统里面查一下。', '请稍等，我这边调一下数据帮您核实。', '请您耐心等待，我帮您核实一下。', '关于您的疑问，请稍候我立刻为您查找。', '耐心在线等候，我梳理本次咨询相关信息。', '您稍候，我马上为您整理出答复。', '麻烦在线等候，我查找您需要的相关信息。', '针对您的咨询，劳烦等候我核对详情资料。', '您刚才的疑问，稍作等候我马上检索核对。', '请稍等，我帮您快速看下。', '烦请您略候，我为您核对相关细节。', '请您稍等一下，我来核实。', '您的问题我记下了，请稍等我帮您查询。', '收到您的疑问，请等片刻我检索详情信息。', '我这边帮您核实一下，请您稍等。', '整理相关内容，辛苦您稍等。', '我这就帮您看看，您稍等一小会儿。', '请等候片刻，我调取业务所需相关资料。', '我查下相关资料，麻烦您稍等。', '承接您的提问，耐心稍等我查找相关内容。', '请等候片刻无需挂断，我梳理资料。', '请您稍等一下，我来帮您确认。', '汇总相关数据，还请您稍等。', '查找相关信息，麻烦您等一下。', '收到您的提问，麻烦稍候我为您核对信息。', '核对参考内容，还请您稍等。', '稍等一小会儿，我检索您咨询的相关资料。', '您咨询的内容，请耐心等候我马上核查。', '调取各类资料，麻烦您等片刻。', '针对您的疑问，稍等片刻我帮您检索核对。', '请稍候片刻，我检索本次咨询所需的资料。', '稍等一下，我帮您把信息查询一下。', '您刚提的问题，请稍候我调取相关信息。', '知悉您的问题，请耐心等候我马上查询。', '查询参考资料，请稍等。', '围绕您的问题，麻烦稍等我调取对应资料。', '稍等片刻，我来帮您查查。', '辛苦短暂在线等候，我调取咨询资料。', '结合您的疑问，稍等一会我核实相关资料。']
PRE_SALES_CUSHIONS_V2 = ['收到，我马上为您查询相关信息。','正在查询相关内容，请您稍等','我帮您看一下，请稍等。','您稍等，我帮您查一查。','您稍等，我去后台查一下。','我正在为您查询，请稍候。','请稍候，我正在为您核实相关信息。','麻烦您稍等，我帮您查一下。','请稍等，我帮您仔细看下。','请稍等一下，我帮您核实下。','请您稍等，我来确认一下。','您稍等，我帮您核实一下。','请稍等，我核实一下。','我帮您核实一下。','我这就帮您看看，请稍等。','请您稍等一下，我来帮您确认。','正在查找相关信息，麻烦您等一下。','请稍等，我来帮您查查。']

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

# ============ 敏感词拦截（AC 自动机，启动时构建一次，请求路径零 IO） ============
import ahocorasick  # pip install pyahocorasick

RISK_CUSHION = "您说的这个。"

# 敏感词库：从外部 txt 文件加载，每行一个词，# 开头为注释；更新后重启服务生效
SENSITIVE_WORDS_FILE = os.getenv(
    "SENSITIVE_WORDS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensitive_words.txt"),
)


def _init_sensitive_word_matcher():
    """加载敏感词库并构建 AC 自动机；文件缺失时 fail-open（打日志但不阻断服务）"""
    words = []
    if os.path.exists(SENSITIVE_WORDS_FILE):
        with open(SENSITIVE_WORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    words.append(w)
    else:
        app.logger.error("敏感词库文件不存在: %s，敏感词拦截未生效(fail-open)", SENSITIVE_WORDS_FILE)

    automaton = ahocorasick.Automaton()
    for w in words:
        automaton.add_word(w, w)
    automaton.make_automaton()
    app.logger.info("敏感词拦截已启用(AC自动机)，词库共 %d 个词", len(words))
    return automaton


_AC_AUTOMATON = _init_sensitive_word_matcher()


def contains_sensitive_word(text: str) -> str:
    """检测文本是否包含敏感词，命中返回命中的词，未命中返回空字符串
       用 AC 自动机 扫一遍文本 ， 命中第一个敏感词就立刻返回
    """
    text = text.lower()
    for _, word in _AC_AUTOMATON.iter(text):
        return word
    return ""

def _clean_think_tags(text: str) -> str:
    """清理空的 <think\> 标签
    如果 <think\></think\> 标签内部没有内容，则移除标签
    """
    # 移除空的 <think\></think\> 标签（包括空白内容）
    text = re.sub(r'<think\>\s*</think\>', '', text, flags=re.DOTALL)
    # 移除其他可能的情况：<think\>\n</think\> 等
    text = re.sub(r'<think\>[\s\n]*</think\>', '', text)
    # 移除所有残留的<*>标签
    text = re.sub(r'<.*?>', '', text)
    return text.strip()


def _call_vllm(dialog_text: str, timeout: int = 30) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V30_0928},
        {"role": "user", "content": dialog_text}
    ]
    payload = {
        "model": "qwen3-4b_sft",
        "messages": messages,
        "max_tokens": 20,  # 售前场景垫话需要 ~17 个 token，改为 20
        "temperature": 0.8, # 适度创造性，避免千篇一律
        "top_p": 0.9,
        "stream": False,
        "frequency_penalty": 0.8,  # 惩罚重复
        "presence_penalty": 0.6,   # 鼓励多样性
    }
    resp = requests.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    result = _clean_think_tags(resp.json()["choices"][0]["message"]["content"])

    # 如果包含逗号，按逗号切分，60%取第一部分，40%取第二部分,应对之前出现的"嗯，了解了。"多次重复的问题
    # if "，" in result:
    #     parts = result.split("，", 1)
    #     if len(parts) == 2:
    #         result = parts[0] if random.random() < 0.6 else parts[1]
    #         result = result.strip() + "。" if not result.strip().endswith("。") else result.strip()

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

    # 敏感词拦截：客户最后一句话命中敏感词，直接返回固定垫话，不再调用模型
    hit_word = contains_sensitive_word(dialog_history[-1][len("客户："):].strip())
    if hit_word:
        total_latency = time.time() - start_total
        app.logger.info(f"source：{source}, Session ID：{session_id}, hit_word:{hit_word}, 命中敏感词拦截, 直接垫固定话术, 耗时: {total_latency:.3f}s")
        return jsonify({
            "code": 200,
            "message": "success",
            "data": {
                "source": source,
                "session_id": session_id,
                "dianhua": RISK_CUSHION,
                "latency": round(total_latency, 3)
            }
        })

    # 获取最后一句客服的话，用于判断垫话是否重复
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
            dianhua = ori_dianhua.split("，")[0] + "，" + random.choice(PRE_SALES_CUSHIONS_V2)
        else:
            dianhua = ori_dianhua
        # 针对越界问题，比如发送短信、发送邮件，的垫话， 垫成：您说的这个。

        commit_actions = ["发","记","转接","联系","专人跟进","帮您预约","帮您安排","马上安排","有意思"]
        for action in commit_actions:
            if action in dianhua:
                dianhua = RISK_CUSHION
                break

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