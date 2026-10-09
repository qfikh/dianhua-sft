import requests
import re
import time
import os
import random
from flask import Flask, request, Response
import json
from redis_helper import redis_get, redis_set

def jsonify(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype="application/json")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:5010")

SYSTEM_PROMPT = """
以下是汽车客服和客户之间的对话数据，你是一个话术指导老师。现在需要你进行一次"垫话"。
"垫话"的意思是：在回答客户问题之前，先进行一次语言承接，比如"好的。"、"我明白您的意思。"、"嗯。"、"嗯嗯。"、"我明白了。"、"我知道了。"、"晓得了。"。

垫话的基本要求如下：
坐席的垫话核心作用是承接客户话语、体现专业礼貌、平稳过渡到正式解答，区别于日常闲聊，需贴合服务岗位的规范与调性，标准如下：
（1）简洁短句，不拖沓
控制在 1-10 字，不占用沟通时长，不打断对话节奏，避免长篇铺垫。
（2）要灵活多变，根据不同的场景输出不同的垫话，不要只输出一种说法，如大部分都输出"我明白了。"是不可以的。可以输出："嗯。""嗯嗯。"，"嗯好。"，"我明白了。"，"我懂您的意思。"，"了解了。"，"我清楚了。"，"晓得了。"，"我理解您的情况。"，"嗯，我晓得了。"，"明白您的诉求。"，"我知晓了。"，"了解。"，"我理解啦。"，"清楚了。"，"我明白。"，"知晓您的问题。"，"明白您的情况了。"，"我了解。"，"我懂啦。"，"了解您的诉求了。"，"明白。"，"了解您的问题了。"，"我知悉了。"，"嗯，明白。"，"我已了解您的情况。"，"您的问题我知晓了。"，"明白您的意思。"，"我清楚您的问题了。"，"您的意思我了解了。"，"我知晓您的诉求了。"，"您的情况我明白了。"，"我理解您的感受。"，"您的需求我了解了。"等。
（3）可以输出语气词，如"嗯。"、"嗯嗯。"、"嗯好。"等，但不能输出"哦"。
（4）精准承接，不脱节
贴合客户上一句的表述（疑问、否认、迷茫、急躁等），不凭空说垫话。
（5）专业中性，不情绪化
不随意夸张附和、不情绪化吐槽，保持坐席的职业中立性。
（6）轻度共情，适配情绪
客户疑惑、急躁时，垫话带轻微安抚感；客户正常沟通时，保持常规承接即可。
（7）不能带有观点，严禁对客户问题做出 "能"/"不能"/"可以"/"不可以"/"是的"/"不行"/"不对" 等任何明确答复性表述，不直接回应问题本身，保持专业礼貌，输出结果不可以是疑问句，不能出现问号。
（8）垫话必须带标点，每条结尾必须加句号，即使垫话只有一个字，比如"嗯"，也要加句号。
（9）输出垫话严禁重复或复述客户最后一句原话，不得对客户问题做出任何直接答复，仅使用规范专业的承接用语，保持礼貌、中性、无观点。
场景要求：
（1）如果客户只是说了语气词，比如"嗯"、"啊"等，要输出"无"。
（2）如只说 "对对对""好的""知道了"，无需额外承接垫话，要输出"无"。
（3）无有效内容可承接时，要输出"无"。
（4）只有零散词语，没有完整意思表达，不适合垫话，要输出"无"。
（5）如果客户没有说完话，如语句不完整、卡顿、只说一半、还在组织语言，没有形成完整问题或诉求，要输出"无"。/no_think
""".strip()

SYSTEM_PROMPT_V6_0520 ="""你作为车联网客服话术专家，请根据对话为坐席生成一句仅用于语言承接的垫话（不表达观点、不指示动作、非问题回答）。"""


SYSTEM_PROMPT_V8_0602 = """你是汽车客服坐席，需要基于客户最新发言生成一句简短的承接话语（垫话）。

【核心任务】
仅作对话承接，保持对话连贯性，为后续服务动作做自然过渡。

【具体要求】
1. 功能纯粹：仅承接，不表达观点、不指示动作、不回答问题。
2. 内容中性：避免任何实质性内容，不提供信息、建议或解决方案。
3. 风格语气：专业客服口吻，自然流畅。
4. 字数限制：严格控制在1-12个字。

【输出格式】
请直接生成一句简短的垫话，无需任何前缀和解释。/no_think """


SYSTEM_PROMPT_V9_0622_V0 = """
## 角色设定
你是专业汽车客服，仅负责生成对话承接垫话。
## 基础核心任务
根据客户最新一条发言，生成2-15字的过渡垫话，仅用于衔接对话、铺垫后续服务；**绝对不解答客户问题、不输出实质信息与建议**，口吻专业自然拟人。

## 三大售前查询意图标准定义（仅客户提以下三类问题才触发追加话术"我帮您查一下。"，不属于一下三类的坚决不要添加追加话术）
### 1. 车型配置查询
判定标准：单款车型名称、车型参数、车辆基础售价咨询；仅车辆基础信息查询，不含车辆功能操作、门店库存查询
正例：V260售价多少、这款车越野性能如何
反例：C260L空调调节方法、门店有没有EQS现车
### 2. 多车型对比查询
判定标准：同时提及≥2款车型，含对比、哪个更好、两者区别类诉求
正例：C级和E级哪个值得买、CLA与C级配置差异
反例：C级全系配置清单（单车型查询，不属于对比）
### 3. 价格政策咨询
判定标准：官方指导价、购车优惠、贷款金融、置换补贴相关金钱类咨询；不含保养、维修售后费用
正例：帮我查一下V260的价格、有没有优惠活动、可以分期吗？分期利息是多少？、有没有置换补贴？
反例：车辆一次保养收费标准

## 追加话术强制触发规则（需**同时满足3个条件**，垫话末尾拼接：我帮您查一下。）
1. 客户最后一句发言意图为上述车型配置/车型对比/价格政策三类之一，且明确表达了查询诉求（如询问、对比、哪个更好、区别、优惠、价格等），
2. 如果客户的问题严格属于上述三类查询意图中的任意一类，不输出追加话术；仅当客户问题同时满足上述三类查询意图条件时，才在垫话末尾追加"我帮您查一下。"
3. 客服历史对话未针对该问题给出有效答复；仅"稍等、好的"这类过渡垫话，不视为有效答复

## 硬性强制约束
1. 功能单一：只做对话过渡承接，不输出观点、不引导操作、不回答客户疑问
2. 内容中立：全程无任何车型、价格、方案类实质信息输出
3. 追加排他：非指定三类售前查询场景，禁止添加 “我帮您查一下。”；
4. 语气规范：汽车官方客服柔和专业口语，通顺自然
5. 字数硬性限制：基础垫话2~15字，不可超长/过短

## 输出格式规范
仅输出最终完整承接语句（基础垫话+可选追加话术），无任何解释、前缀、分段、思考内容。/no_think """



SYSTEM_PROMPT_V9_0622 = """
## 角色设定
你是专业汽车客服，仅负责生成对话承接垫话。
## 基础核心任务
根据客户最新一条发言，生成2-20字过渡垫话，仅用于衔接对话、铺垫后续服务；**绝对不解答客户问题、不输出实质信息与建议**，口吻专业自然拟人。

## 三大售前查询意图标准定义（仅客户提以下三类问题才触发追加话术"我帮您查一下。"）
### 1. 车型配置查询
判定标准：单款车型名称、车型参数、车辆基础售价咨询；仅车辆基础信息查询，不含车辆功能操作、门店库存查询
正例：V260售价多少、这款车越野性能如何
反例：C260L空调调节方法、门店有没有EQS现车
### 2. 多车型对比查询
判定标准：同时提及≥2款车型，含对比、哪个更好、两者区别类诉求
正例：C级和E级哪个值得买、CLA与C级配置差异
反例：C级全系配置清单（单车型查询，不属于对比）
### 3. 价格政策咨询
判定标准：官方指导价、购车优惠、贷款金融、置换补贴相关金钱类咨询；不含保养、维修售后费用
正例：帮我查一下V260的价格、有没有优惠活动、可以分期吗？分期利息是多少？、有没有置换补贴？
反例：车辆一次保养收费标准

## 追加话术强制触发规则（需**同时满足2个条件**，垫话末尾拼接：我帮您查一下。）
1. 仅当客户最后提出的问题是上述三类查询意图时，且明确表达了查询诉求（如询问、对比、哪个更好、区别、优惠、价格等）才在垫话末尾追加"我帮您查一下。"
2. 客服历史对话未针对该问题给出有效答复。

## 生成垫话的强制约束：
1. 功能单一：只做对话过渡承接，不输出观点、不引导操作、不回答客户疑问
2. 内容中立：全程无任何车型、价格、方案类实质信息输出
3. 语气规范：汽车官方客服柔和专业口语，通顺自然
4. 字数硬性限制：生成的垫话长度约为2~20字，不可超长/过短

## 输出格式规范
仅输出最终完整承接语句（基础垫话+可选追加话术），无任何解释、前缀、分段、思考内容。/no_think """



# 售前咨询(车型配置、车型对比、价格)三个场景下加长垫话
pre_sales_cushions = ['收到，我马上为您查询详情。', '围绕您咨询的问题，请稍候我核查内容。', '对于您的咨询，稍作等候我核实相关内容。', '辛苦短暂等候，我调取资料梳理相关内容。', '查询资料内容，麻烦您等等。', '核对详情数据，我快速整理下。', '我帮您看一下，请稍等。', '短暂等候一下，我调取后台资料进行查询。', '核对相关数据，需要一点时间。', '我马上帮您核对清楚情况，稍等。', '关于您的咨询，麻烦等下我马上核查内容。', '您咨询的问题，请耐心等候我即刻检索。', '麻烦耐心等候，我调取资料梳理相关详情。', '稍候片刻，我检索业务相关资料信息。', '结合您的提问，劳烦稍等我核对相关资料。', '请稍作等候，我查询本次咨询涉及的资料。', '您稍等，我帮您查一查。', '您请放心稍候，我正在仔细查阅中。', '知悉您的咨询，请稍片刻我核查对应内容。', '在线稍候片刻，我查找业务对应的相关资料。', '看到您的问题，请稍等我检索相关内容。', '您刚提出的疑问，麻烦等片刻我检索资料。', '您稍等，我去后台查一下。', '稍候片刻，我检索业务相关资料内容。', '辛苦等候片刻，我调取咨询对应的资料。', '对应您的发问，麻烦等下我调取详细信息。', '您咨询的问题，请稍候我快速调取内容。', '我正在为您查询请稍候。', '查询参考内容，稍等我核对完毕。', '麻烦稍候一小会，我调取后台资料检索。', '麻烦耐心等候，我查询咨询涉及的内容。', '麻烦稍等片刻，我马上为您查询相关内容。', '短暂等候片刻，我查找业务所需相关信息。', '核对数据细则，您稍微等等。', '请稍候，我即刻为您核实相关信息。', '依据您的提问，稍作等待我帮您核实详情。', '您提出的疑问，请等一会我即刻查询信息。', '依照您的咨询，劳烦稍等我核对全部详情。', '麻烦您稍等，我帮您查一下。', '辛苦稍作等候，我调取资料核对咨询内容。', '请稍等，我来给您仔细对一下资料。', '查找原始数据，麻烦您稍等。', '依照您的提问，稍候片刻我帮您查清楚。', '麻烦稍作等候片刻，我查询相关资料。', '请等一下，我帮您仔细看下。', '请稍等一下，我帮您核实下。', '请耐心等候，我即刻调取资料核对相关信息。', '请在线稍候，我检索您咨询的业务资料。', '针对您的疑问，劳烦稍等我马上查阅资料。', '请您稍等，我来确认一下。', '您稍等，我帮您核实。', '稍等片刻不要挂断，我检索后台相关资料。', '烦请稍等，我进入系统查一下。', '查询资料内容，稍等我整理下。', '辛苦耐心等候一会，我查询业务内容。', '有关您的问题，请稍候我为您核实资料。', '稍等哈，我为您快速找一下参考点。', '针对您的问题，请稍等片刻我去查找答案。', '请稍等，我核实一下。', '您的问题我帮您核实一下。', '查询参考内容，还请您稍等。', '您稍等，我去系统里面查一下。', '请稍等，我这边调一下数据帮您核实。', '请您耐心等待，我帮您核实一下。', '关于您的疑问，请稍候我立刻为您查找。', '耐心在线等候，我梳理本次咨询相关信息。', '您稍候，我马上为您整理出答复。', '麻烦在线等候，我查找您需要的相关信息。', '针对您的咨询，劳烦等候我核对详情资料。', '您刚才的疑问，稍作等候我马上检索核对。', '请稍等，我帮您快速看下。', '烦请您略候，我为您核对相关细节。', '请您稍等一下，我来核实。', '您的问题我记下了，请稍等我帮您查询。', '收到您的疑问，请等片刻我检索详情信息。', '我这边帮您核实一下，请您稍等。', '整理相关内容，辛苦您稍等。', '我这就帮您看看，您稍等一小会儿。', '请等候片刻，我调取业务所需相关资料。', '我查下相关资料，麻烦您稍等。', '承接您的提问，耐心稍等我查找相关内容。', '请等候片刻无需挂断，我梳理资料。', '请您稍等一下，我来帮您确认。', '汇总相关数据，还请您稍等。', '查找相关信息，麻烦您等一下。', '收到您的提问，麻烦稍候我为您核对信息。', '核对参考内容，还请您稍等。', '稍等一小会儿，我检索您咨询的相关资料。', '您咨询的内容，请耐心等候我马上核查。', '调取各类资料，麻烦您等片刻。', '针对您的疑问，稍等片刻我帮您检索核对。', '请稍候片刻，我检索本次咨询所需的资料。', '稍等一下，我帮您把信息查询一下。', '您刚提的问题，请稍候我调取相关信息。', '知悉您的问题，请耐心等候我马上查询。', '查询参考资料，请稍等。', '围绕您的问题，麻烦稍等我调取对应资料。', '稍等片刻，我来帮您查查。', '辛苦短暂在线等候，我调取咨询资料。', '结合您的疑问，稍等一会我核实相关资料。']
pre_sales_cushions_v2 = ['收到，我马上为您查询相关信息。','正在查询相关内容，请您稍等','我帮您看一下，请稍等。','您稍等，我帮您查一查。','您稍等，我去后台查一下。','我正在为您查询，请稍候。','请稍候，我正在为您核实相关信息。','麻烦您稍等，我帮您查一下。','请稍等，我帮您仔细看下。','请稍等一下，我帮您核实下。','请您稍等，我来确认一下。','您稍等，我帮您核实一下。','请稍等，我核实一下。','我帮您核实一下。','我这就帮您看看，请稍等。','请您稍等一下，我来帮您确认。','正在查找相关信息，麻烦您等一下。','请稍等，我来帮您查查。']

app = Flask(__name__)
import logging
app.logger.setLevel(logging.INFO)

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
        {"role": "system", "content": SYSTEM_PROMPT_V9_0622},
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

    # 获取最后一句客户的话
    last_customer_utterance = dialog_history[-1][len("客户："):].strip()

    # 客户回复太短时，随机不垫话（一定概率返回空）
    # if len(last_customer_utterance) <= 2 and random.random() < 0.5:
    #     return jsonify({
    #         "code": 200,
    #         "message": "客户回复太短，随机跳过垫话",
    #         "data": {"source": source, "session_id": session_id, "dianhua": ""}
    #     })

    # 如果客户最后一句话是问句，直接回复嗯
    # if last_customer_utterance.endswith("？") or last_customer_utterance.endswith("?"):
    #     return jsonify({
    #         "code": 200,
    #         "message": "客户最后一句话是问句，直接回复嗯。",
    #         "data": {"source": source, "session_id": session_id, "dianhua": "嗯~。"}
    #     })


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
        # 如果原始垫话="嗯。" 替换成 "嗯嗯。"
        if dianhua == "嗯。":
            dianhua = "嗯嗯。"

        # 去重：与该 session 最近一次垫话比较，相同则随机替换
        if session_id and session_id != "unknown":
            redis_key = f"cushion_history:{session_id}"
            last_cushion = redis_get(redis_key)
            if last_cushion and dianhua == last_cushion:
                fallback_pool = ["嗯嗯。", "好的。", "了解。", "明白", "嗯，好的。", "嗯嗯，好的。", "好的，了解。", "嗯，了解。", "好的，明白。", "好的，明白了。", "嗯嗯，明白。"]
                fallback_pool = [c for c in fallback_pool if c != dianhua]
                dianhua = random.choice(fallback_pool)
                app.logger.info(f"  [去重] session_id={session_id}，原始垫话：{last_cushion}与最近一次相同，随机替换为：{dianhua}")
            redis_set(redis_key, dianhua, ex=180)
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

# 对应训练数据集的V27版本