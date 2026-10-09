import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import json
import time
import re
import Levenshtein
from typing import List, Dict
import concurrent.futures
import threading
from scripts.llm_api.qwen import call_with_messages

JUDGE_SAME_PROMPT = """
### 角色设定
你是一名资深的**语义一致性质检专家**，拥有极高的语言敏感度和逻辑判断力。你的任务是基于提供的“对话历史”，严格审核“生成的垫话”是否在**语义（意思）、情感色彩和功能指代**上与“原有的垫话内容”保持一致。

### 任务背景
- **场景**：车联网客服对话。
- **垫话定义**：指客服用于语言承接的短句，通常不表达观点、不指示动作、非问题回答（如：了解了、明白、我理解您的心情等）。
- **判定目标**：判断“生成的垫话”是否可以作为“原有的垫话内容”的**完美语义替代品**。

### 判定标准（严格执行）

请遵循以下逻辑进行分步判断：

**第一步：核心语义核对**
1.  **同义判定**：如果两者表达的核心意思完全相同，仅在语气词（如“嗯”、“啊”、“哦”）或句式繁简上有差异，视为【一致】。
    - *示例：* “了解了” vs “嗯，了解了”；“好的” vs “好”。
2.  **近义词判定**：如果两者使用了不同的近义词，但核心指代对象和语境含义完全对应，视为【一致】。
    - *示例：* “我理解您的心情” vs “我明白您的感受”；“明白” vs “知道”。

**第二步：语义偏差排查（发现以下情况立即判定为【不一致】）**
1.  **指代对象错位**：如果“生成的垫话”改变了原话的指代对象（例如：将针对“心情/感受”的回应，改成了针对“情况/问题/疑问”的回应）。
    - *判定逻辑：* “理解心情” ≠ “了解情况”；“明白疑问” ≠ “明白需求”。
2.  **语义范围偏差**：如果“生成的垫话”过于笼统（如“好的”、“嗯”），而“原有垫话”是具体的承接（如“我理解您的顾虑”），视为【不一致】。
3.  **情感色彩缺失**：如果“原有垫话”包含特定的情感共鸣（如“歉意”、“安抚”），而“生成的垫话”仅为中性陈述，视为【不一致】。
4.  **逻辑矛盾**：如果两者在逻辑上无法互通，视为【不一致】。

**第三步：最终裁决**
综合以上分析，只有当“生成的垫话”在核心含义、指代对象和情感色彩上完全覆盖“原有垫话”时，才能判定为一致。

### 输入数据
请根据以下内容进行判断：
- **【对话历史】**：{conversation_history}
- **【原有垫话内容】**：{original_dianhua}
- **【生成的垫话内容】**：{generated_dianhua}

### 输出格式
请直接输出“**一致**”或“**不一致**”，不要添加任何解释、理由或其他文本。
"""


SYSTEM_PROMPT_V2="""
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
"""

# SYSTEM_PROMPT = """
# 你是一名专业的汽车客服。你的核心任务是根据对话历史，生成一句符合规范的"垫话"，用于承接客户话语、体现专业礼貌，并平稳过渡到正式解答。\n\n**垫话核心规范：**\n1.  **身份与目的**：你是客服，目标是礼貌承接，不直接回答客户问题。\n2.  **格式要求**：必须为1-10个字的短句，且以句号结尾（例如："好的。"、"我明白了。"）。\n3.  **内容禁区**：严禁在垫话中包含任何观点或直接答复（如"能"、"不能"、"是的"），严禁使用疑问句。\n4.  **表达多样性**：需灵活使用多种规范表达（如"嗯。"、"了解。"、"我明白您的意思。"），不可千篇一律。\n5.  **场景判断（关键）**：当客户发言仅为语气词（如"嗯"、"啊"）、简单附和（如"对对对"）、语句不完整或无有效内容时，你应输出"无"。\n\n请严格遵循以上规范生成垫话。
# """

SYSTEM_PROMPT_V4=""


SYSTEM_PROMPT_V5 ="""
你是一个话术指导老师。请根据以下对话，为坐席生成**唯一一句垫话**。
## 垫话定义
垫话是坐席在正式回答客户问题前，用于语言承接的简短过渡用语，仅起承接作用，不表达任何观点，不指示客户任何动作，**不是对问题的回答**。
## 垫话要求
1. 字数1-10字，简洁短句
2. 表达多样化，禁止高度重复
3. 不能复制、复述客户任何原话
4. 不能回答问题（不能说"能/不能/可以/不可以"等）
5. 必须带正确的标点，以句号结尾
6. 每次只输出一句，不加任何额外文字
## 输出规则
1. 根据客户最后一句话判断场景，选择对应的表达：
   - 客户提供信息：用"知晓"/"了解"/"嗯"相关表达
   - 客户表达观点：用"理解"/"明白"相关表达
   - 客户骂人：用"理解您的感受"/"我懂您的心情"相关表达
   - 客户判断：用"知道了"/"收到"相关表达
   - 客户不满：用"我理解您的心情"/"明白您的感受"相关表达
   - 客户指示：用"清楚了"/"好的"相关表达
   - 客户疑问：输出"无"
   - 客户语气词：输出"无"
   - 客户表达残缺：输出"无"
2. 在对应的表达范围内随机选择，确保多样性

"""

SYSTEM_PROMPT_V6_0520 ="""你作为车联网客服话术专家，请根据对话为坐席生成一句仅用于语言承接的垫话（不表达观点、不指示动作、非问题回答）。"""

SYSTEM_PROMPT_V7_0526 ="""你是汽车客服坐席，基于客户最新发言生成一句垫话，仅作承接，不表达观点、不指示动作、不回答问题。/no_think"""

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
4. 字数硬性限制：基础垫话2~20字，不可超长/过短

## 输出格式规范
仅输出最终完整承接语句（基础垫话+可选追加话术），无任何解释、前缀、分段、思考内容。/no_think """



SYSTEM_PROMPT_V29_0807 = """
## 角色设定
你是专业汽车客服，仅负责生成对话承接垫话。
## 基础核心任务
根据客户最新一条发言，生成2-20字过渡垫话，仅用于衔接对话、铺垫后续服务；**绝对不解答客户问题、不输出实质信息与建议**，口吻专业自然拟人。

## 三大售前查询意图标准定义（仅客户提以下三类问题才触发追加话术\"我帮您查一下。\"）
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
1. 仅当客户最后提出的问题是上述三类查询意图时，且明确表达了查询诉求（如询问、对比、哪个更好、区别、优惠、价格等）才在垫话末尾追加\"我帮您查一下。\"
2. 客服历史对话未针对该问题给出有效答复。

## 生成垫话的强制约束：
1. 功能单一：只做对话过渡承接，不输出观点、不引导操作、不回答客户疑问
2. 内容中立：全程无任何车型、价格、方案类实质信息输出
3. 语气规范：汽车官方客服柔和专业口语，通顺自然
4. 字数硬性限制：生成的垫话长度约为2~20字，不可超长/过短

## 输出格式规范
仅输出最终完整承接语句（基础垫话+可选追加话术），无任何解释、前缀、分段、思考内容。/no_think
"""

SYSTEM_PROMPT_V30_0810 = """
## 角色
你是汽车客服热线坐席，负责在客户说完话后、正式回答之前，说一句自然的承接性垫话，让对话衔接流畅。

## 垫话要求
- 职能边界：只说过渡承接的话，不解答客户问题、不给建议、不输出结论性表态（"可以"/"不行"/"能"/"不能"/"是的"等）。
- 语气风格：口语化、自然拟人，像真实客服在电话里随口说的一句过渡话。
- 内容：可以简短复述客户问题要点、确认听到的内容，但不要展开解释或给出答案。

## 输入说明
对话内容是车机语音识别（ASR）转写结果，口语化强、断句多、语气词多，不一定是完整句子。

## 输出格式
只输出一句话，不带任何解释、前缀、思考过程或分段标记。"""



class CushionVLLMClient:
    """vLLM垫话生成客户端"""

    def __init__(self, base_url="http://localhost:5010"):
        self.base_url = base_url
        self.system_prompt = SYSTEM_PROMPT_V30_0810.strip()
        # 添加统计变量
        self.request_count = 0
        self.total_latency = 0.0
        self.latencies = []  # 保存所有请求的耗时
        # 线程锁，保证多线程安全
        self._lock = threading.Lock()

    def _clean_think_tags(self, text: str) -> str:
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

    def generate_cushion(self, conversation_history: str, timeout=30, max_retries=5) -> str:
        """生成单条垫话，调用Flask接口，失败或返回为空时重试"""
        # 判断输入是否为空或无效
        if not conversation_history.strip():
            return "无"

        # 解析对话历史为列表格式（Flask接口接收的格式）
        dialog_history = []
        lines = conversation_history.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("客户：") or line.startswith("客服："):
                dialog_history.append(line)

        if not dialog_history:
            return "无"

        # 构造Flask接口请求体
        payload = {
            "dialog_history": dialog_history
        }

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    timeout=timeout
                )
                latency = time.time() - start_time

                if response.status_code == 200:
                    result = response.json()
                    cushion = result.get("data", {}).get("dianhua", "")

                    # 清理空的 <think\> 标签
                    cushion = self._clean_think_tags(cushion)

                    # 如果 cushion 为空，重试
                    if not cushion:
                        print(f"⚠️ 返回为空，第 {attempt + 1}/{max_retries} 次重试")
                        continue

                    # 统计耗时（线程安全）
                    with self._lock:
                        self.request_count += 1
                        self.total_latency += latency
                        self.latencies.append(latency)
                        avg_latency = self.total_latency / self.request_count

                    print(f"✅ 生成成功 | 延迟: {latency:.3f}s | 平均延迟: {avg_latency:.3f}s | 垫话: {cushion}")
                    return cushion
                else:
                    print(f"❌ 请求失败: {response.status_code} - {response.text} | 第 {attempt + 1}/{max_retries} 次重试")

            except Exception as e:
                print(f"❌ 请求异常: {e} | 第 {attempt + 1}/{max_retries} 次重试")

        print(f"❌ 已达最大重试次数 {max_retries}，返回默认值")
        return "无"

    def get_stats(self):
        """获取统计信息"""
        if self.request_count == 0:
            return "暂无统计数据"

        avg_latency = self.total_latency / self.request_count
        max_latency = max(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0

        return f"请求总数: {self.request_count}, 平均延迟: {avg_latency:.3f}s, 最大延迟: {max_latency:.3f}s, 最小延迟: {min_latency:.3f}s"

    def batch_generate(self, conversations: List[str], batch_size=4, max_workers=4) -> List[str]:
        """批量生成垫话（支持多并发）

        Args:
            conversations: 对话列表
            batch_size: 每批处理的对话数量（用于进度显示）
            max_workers: 最大并发数
        """
        results = []
        total = len(conversations)

        # 使用线程池执行并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_conv = {executor.submit(self.generate_cushion, conv): conv
                            for conv in conversations}

            # 处理完成的任务
            completed = 0
            for future in concurrent.futures.as_completed(future_to_conv):
                conv = future_to_conv[future]
                try:
                    cushion = future.result()
                    results.append({
                        "conversation": conv,
                        "cushion": cushion
                    })
                except Exception as e:
                    print(f"❌ 处理对话失败: {e}")
                    results.append({
                        "conversation": conv,
                        "cushion": "无"
                    })

                completed += 1
                if completed % batch_size == 0:
                    print(f"处理进度: {completed}/{total}")

        print(f"📦 处理完成: {total}/{total}")
        return results

# 实现一个方法，输入是对话历史、原有垫话内容、生成的垫话内容，输出是对比结果：一致/不一致，
# 方案是将对话历史、原有垫话内容、生成的垫话内容输入到一个专门判断语义一致性的prompt中，让模型输出一致/不一致的结果.
# 调用


def test_data_generation(client: CushionVLLMClient, max_workers=4):
    """
    读取test_0414_720_comp_prompt.json文件中的数据，读取每条数据中的 messages 列表，找到system角色的content作为system_prompt，找到user角色的content作为conversation_history，调用batch_generate接口并发生成垫话内容，
    将 生成的垫话内容与原有assistant角色的content进行对比，输出对比结果：一致/不一致。
    最后将 system_prompt、conversation_history、原有垫话内容、生成的垫话内容、对比结果 输出到一个新的 xlsx 文件中 test_0414_720_comp_prompt_output.xlsx 中。

    Args:
        client: CushionVLLMClient实例
        max_workers: 最大并发数，默认为4
    """
    from openpyxl import Workbook
    import os

    # json_path = "../data/0417_dianhua_test_complex_prompt.json"
    # json_path = "../data/0417_dianhua_test_complex_prompt.json"
    # json_path = "../data/0417_dianhua_train_complex_prompt.json"
    #json_path = "../data/datasets/0604_v4_dianhua_test_sft_simple_prompt.json"
    # json_path = "../../data/datasets/v29_0804_dianhua_test_sft_simple_prompt.json"
    json_path = "../../data/datasets/v30_0811_dianhua_test_sft_simple_prompt.json"

    # output_path = "../data/output/v18_0527_v3_qwen3-4b_sft_test_simple_prompt_output.xlsx"
    # output_path = "../data/output/v19_0527_v3_qwen3-4b_sft_test_simple_prompt_output.xlsx"
    # output_path = "../data/output/v19_0527_v3_2_qwen3-4b_sft_test_simple_prompt_output.xlsx"
    #output_path = "../data/output/v19_0527_v3_2_qwen3-4b_sft_zliu_test_simple_prompt_output.xlsx"
    output_path = "../../data/output/v30_0811_qwen3-4b_sft_test_simple_prompt_output-succ-train.xlsx"


    print(f" 输出结果文件: {output_path}")
    print(f" 并发数: {max_workers}")

    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    print(f"📖 共读取 {len(data_list)} 条数据")

    # 准备数据
    conversations_data = []
    original_cushions = []
    system_prompts = []
    conversation_histories = []

    for idx, item in enumerate(data_list):
        # if idx >= 10:
        #     break

        messages = item.get("messages", [])
        system_prompt = ""
        conversation_history = ""
        original_cushion = ""

        # 提取 system prompt
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break

        # 提取最后一个 assistant 作为垫话内容
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                original_cushion = msg["content"]
                break

        # 构建完整的对话历史（排除 system 和最后一个 assistant）
        history_messages = []
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                continue
            # 跳过最后一个 assistant（垫话内容）
            if msg["role"] == "assistant" and i == len(messages) - 1:
                continue
            history_messages.append(msg)

        # 将对话历史格式化为字符串（参照 build_sft_data_v3_0526.py 的格式）
        history_lines = []
        for msg in history_messages:
            if msg["role"] == "user":
                history_lines.append(f"客户：{msg['content']}")
            elif msg["role"] == "assistant":
                history_lines.append(f"客服：{msg['content']}")
        conversation_history = "\n".join(history_lines)

        if conversation_history:
            # 只存储对话历史，用于生成垫话
            conversations_data.append(conversation_history)
            original_cushions.append(original_cushion)
            system_prompts.append(system_prompt)
            conversation_histories.append(conversation_history)

    print(f"准备处理 {len(conversations_data)} 条对话数据")
    

    # 使用batch_generate进行并发处理
    batch_results = client.batch_generate(conversations_data, batch_size=max_workers, max_workers=max_workers)

    # print(f"conversations_data:{conversations_data}, batch_results:{batch_results}")
    # 创建Excel文件
    wb = Workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="system_prompt")
    ws.cell(row=1, column=2, value="conversation_history")
    ws.cell(row=1, column=3, value="原有垫话内容")
    ws.cell(row=1, column=4, value="生成的垫话内容")
    ws.cell(row=1, column=5, value="字符串对比结果")
    ws.cell(row=1, column=6, value="语义一致率")
    ws.cell(row=1, column=7, value="编辑距离相似度")

    
    def normalize_text(text):
        text = re.sub(r'\s+', '', text)
        text = text.replace('，', '').replace('。', '').replace('！', '').replace('？', '')
        text = text.replace(',', '').replace('!', '').replace('?', '')
        return text.lower()

    def calc_similarity(s1, s2):
        """基于编辑距离计算相似度（0~1）"""
        if not s1 and not s2:
            return 1.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - Levenshtein.distance(s1, s2) / max_len

    total_count = 0
    match_count = 0

    # 处理结果并写入Excel
    for idx, (result, original_cushion, system_prompt, conversation_history) in enumerate(zip(batch_results, original_cushions, system_prompts, conversation_histories)):
        generated_cushion = result["cushion"]

        norm_original = normalize_text(original_cushion)
        norm_generated = normalize_text(generated_cushion)
        string_compare_result = "一致" if norm_original == norm_generated else "不一致"

        # 计算编辑距离相似度
        # similarity = calc_similarity(norm_original, norm_generated)
        
        # 有一个为空时直接判定为不一致
        compare_result = "不一致"
        # if not norm_original or not norm_generated:
        #     compare_result = "不一致"
        # else:
        #     # 使用qwen判断是否语义一致
        #     judge_prompt = JUDGE_SAME_PROMPT.format(
        #         conversation_history=conversation_history,
        #         original_dianhua=original_cushion,
        #         generated_dianhua=generated_cushion
        #     )
        #
        #     judge_result = call_with_messages(
        #         model="qwen-plus",
        #         system_prompt=judge_prompt,
        #         user_content="请判断生成的垫话与原有的垫话是否一致。"
        #     )
        #     compare_result = "一致" if "一致" in judge_result and "不一致" not in judge_result else "不一致"
        if compare_result == "一致":
            match_count += 1
        total_count += 1

        row = idx + 2
        ws.cell(row=row, column=1, value=system_prompt)
        ws.cell(row=row, column=2, value=conversation_history)
        ws.cell(row=row, column=3, value=original_cushion)
        ws.cell(row=row, column=4, value=generated_cushion)
        ws.cell(row=row, column=5, value=string_compare_result)
        ws.cell(row=row, column=6, value=compare_result)
        # ws.cell(row=row, column=7, value=round(similarity, 4))

        print(f"[{idx+1}/{len(conversations_data)}] 原有=[{original_cushion}] | 生成=[{generated_cushion}]  | {compare_result}")

    wb.save(output_path)

    accuracy = (match_count / total_count * 100) if total_count > 0 else 0
    print(f"\n处理完成！共处理 {total_count} 条数据")
    print(f"一致率: {accuracy:.2f}% ({match_count}/{total_count})")
    print(f"结果已保存至: {output_path}")

    # 打印统计信息
    print(f"\n请求统计信息: {client.get_stats()}")


# 使用示例
if __name__ == "__main__":
    # v4版本相比v3 版本主要的改动是，由原来的直接请求vllm服务，改为请求flask接口
    # 初始化客户端
    # client = CushionVLLMClient("http://192.168.86.3:5013")
    client = CushionVLLMClient("http://192.168.86.3:5011")

    # 并发数配置
    max_workers = 1  # 可以根据需要调整并发数

    # 测试集验证效果
    test_data_generation(client, max_workers=max_workers)