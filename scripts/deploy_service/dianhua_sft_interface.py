import requests
import re
import time
from datetime import datetime
import os
import random
from flask import Flask, request, Response
import json
from prompt import *

app = Flask(__name__)
import logging
app.logger.setLevel(logging.INFO)

# 业务日志按天落盘：每天自动切换 dianhua_flask_YYYYMMDD.log
# gunicorn 运行日志（access/error）也合并写入同一按天文件
# 日志目录优先取环境变量 LOG_DIR（与启动脚本一致），未设置则用脚本所在目录下的 logs
LOG_DIR = os.getenv("LOG_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


class DailyFileHandler(logging.FileHandler):
    """按天切换日志文件，跨天自动新建当天文件"""
    def __init__(self, log_dir, encoding="utf-8"):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        super().__init__(self._today_path(), encoding=encoding)

    def _today_path(self):
        return os.path.join(self._log_dir, "dianhua_flask_%s.log" % datetime.now().strftime("%Y%m%d"))

    def emit(self, record):
        if self._today_path() != self.baseFilename:  # 已跨天，切换新文件
            self.close()
            self.baseFilename = self._today_path()
            self.stream = self._open()
        super().emit(record)

app.logger.handlers = [DailyFileHandler(LOG_DIR)]
app.logger.propagate = False  # 只写按天文件，避免重复输出到 stderr

# gunicorn 运行日志（access / error）也合并写入同一按天文件
# （master 进程启动阶段的少量日志仍在 dianhua_flask_gunicorn.log，app 加载后即并入按天文件）
for _logger_name in ("gunicorn.access", "gunicorn.error"):
    _gl = logging.getLogger(_logger_name)
    _gl.handlers = [DailyFileHandler(LOG_DIR)]
    _gl.propagate = False

def jsonify(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype="application/json")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:6015") # TODO 注意不同环境下这里要改

# 售前咨询(车型配置、车型对比、价格)三个场景下加长垫话
# PRE_SALES_CUSHIONS = ['收到，我马上为您查询详情。', '围绕您咨询的问题，请稍候我核查内容。', '对于您的咨询，稍作等候我核实相关内容。', '辛苦短暂等候，我调取资料梳理相关内容。', '查询资料内容，麻烦您等等。', '核对详情数据，我快速整理下。', '我帮您看一下，请稍等。', '短暂等候一下，我调取后台资料进行查询。', '核对相关数据，需要一点时间。', '我马上帮您核对清楚情况，稍等。', '关于您的咨询，麻烦等下我马上核查内容。', '您咨询的问题，请耐心等候我即刻检索。', '麻烦耐心等候，我调取资料梳理相关详情。', '稍候片刻，我检索业务相关资料信息。', '结合您的提问，劳烦稍等我核对相关资料。', '请稍作等候，我查询本次咨询涉及的资料。', '您稍等，我帮您查一查。', '您请放心稍候，我正在仔细查阅中。', '知悉您的咨询，请稍片刻我核查对应内容。', '在线稍候片刻，我查找业务对应的相关资料。', '看到您的问题，请稍等我检索相关内容。', '您刚提出的疑问，麻烦等片刻我检索资料。', '您稍等，我去后台查一下。', '稍候片刻，我检索业务相关资料内容。', '辛苦等候片刻，我调取咨询对应的资料。', '对应您的发问，麻烦等下我调取详细信息。', '您咨询的问题，请稍候我快速调取内容。', '我正在为您查询请稍候。', '查询参考内容，稍等我核对完毕。', '麻烦稍候一小会，我调取后台资料检索。', '麻烦耐心等候，我查询咨询涉及的内容。', '麻烦稍等片刻，我马上为您查询相关内容。', '短暂等候片刻，我查找业务所需相关信息。', '核对数据细则，您稍微等等。', '请稍候，我即刻为您核实相关信息。', '依据您的提问，稍作等待我帮您核实详情。', '您提出的疑问，请等一会我即刻查询信息。', '依照您的咨询，劳烦稍等我核对全部详情。', '麻烦您稍等，我帮您查一下。', '辛苦稍作等候，我调取资料核对咨询内容。', '请稍等，我来给您仔细对一下资料。', '查找原始数据，麻烦您稍等。', '依照您的提问，稍候片刻我帮您查清楚。', '麻烦稍作等候片刻，我查询相关资料。', '请等一下，我帮您仔细看下。', '请稍等一下，我帮您核实下。', '请耐心等候，我即刻调取资料核对相关信息。', '请在线稍候，我检索您咨询的业务资料。', '针对您的疑问，劳烦稍等我马上查阅资料。', '请您稍等，我来确认一下。', '您稍等，我帮您核实。', '稍等片刻不要挂断，我检索后台相关资料。', '烦请稍等，我进入系统查一下。', '查询资料内容，稍等我整理下。', '辛苦耐心等候一会，我查询业务内容。', '有关您的问题，请稍候我为您核实资料。', '稍等哈，我为您快速找一下参考点。', '针对您的问题，请稍等片刻我去查找答案。', '请稍等，我核实一下。', '您的问题我帮您核实一下。', '查询参考内容，还请您稍等。', '您稍等，我去系统里面查一下。', '请稍等，我这边调一下数据帮您核实。', '请您耐心等待，我帮您核实一下。', '关于您的疑问，请稍候我立刻为您查找。', '耐心在线等候，我梳理本次咨询相关信息。', '您稍候，我马上为您整理出答复。', '麻烦在线等候，我查找您需要的相关信息。', '针对您的咨询，劳烦等候我核对详情资料。', '您刚才的疑问，稍作等候我马上检索核对。', '请稍等，我帮您快速看下。', '烦请您略候，我为您核对相关细节。', '请您稍等一下，我来核实。', '您的问题我记下了，请稍等我帮您查询。', '收到您的疑问，请等片刻我检索详情信息。', '我这边帮您核实一下，请您稍等。', '整理相关内容，辛苦您稍等。', '我这就帮您看看，您稍等一小会儿。', '请等候片刻，我调取业务所需相关资料。', '我查下相关资料，麻烦您稍等。', '承接您的提问，耐心稍等我查找相关内容。', '请等候片刻无需挂断，我梳理资料。', '请您稍等一下，我来帮您确认。', '汇总相关数据，还请您稍等。', '查找相关信息，麻烦您等一下。', '收到您的提问，麻烦稍候我为您核对信息。', '核对参考内容，还请您稍等。', '稍等一小会儿，我检索您咨询的相关资料。', '您咨询的内容，请耐心等候我马上核查。', '调取各类资料，麻烦您等片刻。', '针对您的疑问，稍等片刻我帮您检索核对。', '请稍候片刻，我检索本次咨询所需的资料。', '稍等一下，我帮您把信息查询一下。', '您刚提的问题，请稍候我调取相关信息。', '知悉您的问题，请耐心等候我马上查询。', '查询参考资料，请稍等。', '围绕您的问题，麻烦稍等我调取对应资料。', '稍等片刻，我来帮您查查。', '辛苦短暂在线等候，我调取咨询资料。', '结合您的疑问，稍等一会我核实相关资料。']
PRE_SALES_CUSHIONS_V2 = ['收到，我马上为您查询相关信息。','正在查询相关内容，请您稍等','我帮您看一下，请稍等。','您稍等，我帮您查一查。','您稍等，我去后台查一下。','我正在为您查询，请稍候。','请稍候，我正在为您核实相关信息。','麻烦您稍等，我帮您查一下。','请稍等，我帮您仔细看下。','请稍等一下，我帮您核实下。','请您稍等，我来确认一下。','您稍等，我帮您核实一下。','请稍等，我核实一下。','我帮您核实一下。','我这就帮您看看，请稍等。','请您稍等一下，我来帮您确认。','正在查找相关信息，麻烦您等一下。','请稍等，我来帮您查查。']

# 黄赌毒敏感话题、代词问题 用的垫话
RISK_CUSHION = ["您说的这个问题。", "您问的这个问题。", "您说的这个情况。", "您提到的这个情况。"]
# 过度承诺原有的垫话"您说的这个。"  “您说的这个”还有“您说的情况”像半句话，且有时候衔接的比较突兀。--> "了解您说的情况。"
# 针对还有 认同、附和或确认客户对问题的负面评价如："确实", "真挺", "真是", "挺闹心", 也修改成这个。
PROMISE_CUSHION = ["了解您说的情况了。","了解您的问题了。", "了解您的情况了。", "您说的情况已了解", "明白您的意思了。"]  # "了解您提到的情况。",
# 述求指令类问题：垫话优化路径： "马上给您发送/转接/预约/安排" --> "您说的这个。" --> "请您稍等。"
COMMAND_CUSHION = ["好的，请您稍等。", "请您稍等一下。", "请您稍等片刻。", "麻烦您稍等一下。"]
# 转人工述求： 跟指令类的垫话很相似
ZHUAN_RENGONG_CUSHION  = ["好的，了解您的问题了。", ]
# 垫话中包含 您先忙 的 --> 嗯嗯，好的。
BUSY_CUSHION = ["嗯嗯，好的。"]


# commit_actions = ["发","记","转接","联系","专人跟进","帮您预约","帮您安排","马上安排","有意思"]
# 过度承诺 / 越权指令类垫话的兜底替换规则表 (触发关键词, 候选垫话)，按顺序匹配，命中即停
CUSHION_FALLBACKS = [
    (("查一下"), PRE_SALES_CUSHIONS_V2),
    (("真人", "人工"), ZHUAN_RENGONG_CUSHION),
    (("您先忙。", "您先忙，"), BUSY_CUSHION),
    (("您说的这个。", "您说的这个，", "您问的这个。", "您问的这个，", "您提到的这个。", "您提到的这个，", "您提的这个。", "您提的这个，"), RISK_CUSHION),  # "您说的这个问题。"
    (("确实", "真挺", "真是", "挺闹心", "记着", "记下", "记一下", "已记录", "已经记录", "您记录", "专人跟进", "有意思"), PROMISE_CUSHION),  # 过度承诺--> 了解您说的这个情况
    (("发给", "发送", "您发", "转接", "接通", "连线", "您连接", "联系", "您预约", "您安排", "马上安排"), COMMAND_CUSHION), # 越权指令--> 请稍等
]


# 垫话去重时的词语替换规则 (old, new)
DIANHUA_REPLACEMENTS = [
    ("嗯，", "嗯嗯，"),
    ("了解了。", "明白了。"),
    ("了解。", "明白。"),
    ("，好，", "，好的，"),
    ("我帮您看一下", "我帮您确认一下")
]


# ============ 敏感词拦截（AC 自动机，启动时构建一次，请求路径零 IO） ============
import ahocorasick  # pip install pyahocorasick
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

def _build_sft_messages(dialog_history: list, system_prompt: str) -> list:
    """将 客服：/客户： 标注的历史还原为与训练一致的 ChatML 多轮消息"""
    messages = [{"role": "system", "content": system_prompt}]
    for line in dialog_history:
        if line.startswith("客户："):
            messages.append({"role": "user", "content": line[len("客户："):].strip()})
        elif line.startswith("客服："):
            messages.append({"role": "assistant", "content": line[len("客服："):].strip()})
    return messages



def _call_vllm(dialog_text: str, timeout: int = 30) -> str:
    # messages = [ # 对话内容拼接成单轮的方式
    #     {"role": "system", "content": SYSTEM_PROMPT_V31_1022},
    #     {"role": "user", "content": dialog_text}
    # ]
    # 多轮对话的方式
    messages = _build_sft_messages(dialog_text.split("\n"), SYSTEM_PROMPT_V32_1110)

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
    request_data = request.get_json(force=True)
    request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    source = request_data.get("source", "unknown")
    session_id = request_data.get("session_id", "unknown")
    origin_dialog_history = request_data.get("dialog_history", [])
    # 完整的对话历史  将dialog_history 中的"Client:"替换为 "客户："，"Service:"替换为 "客服："
    origin_dialog_history = [utterance.replace("Client:", "客户：").replace("Service:", "客服：") for utterance in origin_dialog_history]
    # 合并连续同角色语句
    origin_dialog_history = merge_consecutive_utterances(origin_dialog_history)
    dialog_history = origin_dialog_history[-10:]  # 只保留最近n轮对话，减少输入长度，提升响应速度

    response_data = {
        "source": source,
        "session_id": session_id,
        "request_time": request_time,
        "dianhua": "",
        "latency": 0.0,
    }
    log_data = {
        "source": source,
        "session_id": session_id,
        "request_time": request_time,
        "dialog_history": origin_dialog_history,
        "message": "",
        "model_dianhua": "",
        "final_dianhua": "",
    }

    # Part1、根据对话历史直接规则垫话
    total_latency = time.time() - start_total
    last_conversation = dialog_history[-1].strip() if dialog_history else ""
    # 无需垫话的情况：对话为空、最后一句是客服说的
    if not dialog_history or last_conversation.startswith("客服："):
        response_data["latency"] = total_latency
        log_data["message"] = "对话历史为空或最后一句话是客服说的，无需垫话。"
        app.logger.info(json.dumps(log_data, ensure_ascii=False))
        return jsonify({
            "code": 200,
            "message": "对话历史为空或最后一句话是客服说的，无需垫话。",
            "data": response_data
        })

    # 如果客户要求转人工、真人客服，则直接返回 "好的，了解您的述求了。" 
    if "真人" in last_conversation or "人工" in last_conversation :
        response_data["latency"] = total_latency
        zhuanRenGong = random.choice(ZHUAN_RENGONG_CUSHION)
        response_data["dianhua"] = zhuanRenGong
        log_data["message"] = "用户要求转人工，直接返回固定垫话。"
        log_data["final_dianhua"] = zhuanRenGong
        log_data["total_latency"] = round(total_latency, 4)
        app.logger.info(json.dumps(log_data, ensure_ascii=False))
        return jsonify({
            "code": 200,
            "message": "用户要求转人工，直接返回固定垫话。",
            "data": response_data
        })   

    # 最后一句为 "客户：用户无输入"，则直接返回 "喂，您好" 
    if "客户：用户无输入" in last_conversation :
        response_data["latency"] = total_latency
        response_data["dianhua"] = "您好。" 
        log_data["message"] = "用户无输入，直接返回固定垫话。"
        log_data["final_dianhua"] = "您好。"
        log_data["total_latency"] = round(total_latency, 4)
        app.logger.info(json.dumps(log_data, ensure_ascii=False))
        return jsonify({
            "code": 200,
            "message": "用户无输入，直接返回固定垫话。",
            "data": response_data
        })
    # 最后一句为 "客户：好的。"，则直接返回 "嗯嗯，好的。" 
    if last_conversation == "客户：好的。":
        response_data["latency"] = total_latency
        response_data["dianhua"] = "嗯嗯，好的。" 
        log_data["message"] = "用户回复：好的。返回固定垫话。"
        log_data["final_dianhua"] = "嗯嗯，好的。"
        log_data["total_latency"] = round(total_latency, 4)
        app.logger.info(json.dumps(log_data, ensure_ascii=False))
        return jsonify({
            "code": 200,
            "message": "用户回复：好的。返回固定垫话。",
            "data": response_data
        })

    # 敏感词拦截：客户最后一句话命中敏感词，直接返回固定垫话，不再调用模型
    hit_word = contains_sensitive_word(last_conversation)
    if hit_word:
        total_latency = time.time() - start_total
        response_data["latency"] = total_latency
        response_data["dianhua"] = random.choice(RISK_CUSHION) 
        log_data["message"] = f"命中敏感词 {hit_word}, 直接垫固定话术。"
        log_data["final_dianhua"] = random.choice(RISK_CUSHION)
        log_data["total_latency"] = round(total_latency, 4)
        app.logger.info(json.dumps(log_data, ensure_ascii=False))
        return jsonify({
            "code": 200,
            "message": "success",
            "data": response_data
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
    
    # Part2、prompt 控制 模型生成垫话
    try:    
        start_vllm = time.time()  # vllm调用开始时间
        model_dianhua = _call_vllm(conversation)
        vllm_latency = time.time() - start_vllm  # vllm调用耗时

        # Part3、规则引擎 修改模型生成的垫话
        # 如果模型生成的垫话包含"我帮您查一下"，则替换成随机查询等待场景垫话 
        dianhua = model_dianhua
        # if "查一下" in dianhua: 
        #     dianhua = random.choice(PRE_SALES_CUSHIONS_V2)
        
        # 代词问题、过度承诺、越权指令类 垫话兜底替换：按规则表顺序匹配，命中即停
        for keywords, cushions in CUSHION_FALLBACKS:
            if any(kw in dianhua for kw in keywords):
                dianhua = random.choice(cushions)
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
            # 如果仍然与最近一次垫话相同，随机替换，目前对话轮次较少，重复度较低。
            # if dianhua == last_cushion:
            #     dianhua = random.choice(cushions)
        
        total_latency = time.time() - start_total  # 整个方法耗时
        response_data["latency"] = total_latency
        response_data["dianhua"] = dianhua

        # 打印监控日志
        log_data["message"] = f"模型生成垫话，然后规则引擎修改"
        log_data["model_dianhua"] = model_dianhua
        log_data["final_dianhua"] = dianhua
        log_data["vllm_latency"] = round(vllm_latency, 4)
        log_data["total_latency"] = round(total_latency, 4)
        app.logger.info(json.dumps(log_data, ensure_ascii=False))

        return jsonify({
            "code": 200,
            "message": "success",
            "data": response_data
        })
    except Exception as e:
        app.logger.error("vLLM调用失败: %s", e)
        return jsonify({
            "code": 500,
            "message": str(e),
            "data": response_data
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