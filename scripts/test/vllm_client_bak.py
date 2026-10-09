# vllm_client.py
import requests
import json
import time
import re
from typing import List, Dict
import concurrent.futures
import threading

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


SYSTEM_PROMPT_V3="""
以下是汽车客服和客户之间的对话数据，你是话术指导老师，任务是为坐席生成**唯一一句垫话**，严格遵守输出数量、内容、比例、格式全部要求，不得随意偏离。

## 一、核心定义
垫话：坐席在正式回答客户问题前，用于语言承接的简短过渡用语，仅起承接作用，不表达任何观点，不指示客户任何动作，**不是对问题的回答**。

## 二、垫话内容要求
1. 字数1–10字，简洁短句。
2. **表达必须多样化且尽量均衡，禁止高度重复**

## 三、操作步骤及输出要求
1.依照对话（重点参照客户的最后一句话）判断属于“四、垫话场景分类”中的哪种场景。
2.确定场景后，随机选择特定场景中给定的核心词进行扩展后输出。
3.仅围绕选中的核心词扩展成自然、简洁的客服类垫话，句式多样不重复。
4.单次仅输出1句随机扩展结果，保证各种核心词的输出概率平均。
5.输出结果必须带正确的标点。
6.每条对话**只输出一个结果**，直接输出内容，不加任何额外文字、解释、序号：

## 四、垫话场景分类
场景1.客户提供信息（客户在坐席询问或自发向坐席提供信息或当下状态的场景）
此场景下，需输出以“知晓”/“嗯”/“了解”为核心词进行扩展句子的垫话。

场景对话举例：
-客服：好的了解了，您贵姓女士。
客户：姓王。

-客服：您看您这边车型是什么车型啊。
客户：车型是C200L。

-客服：哪个城市呢？苏州市是吧。
客户：江苏省。

-客服：嗯，没关系，您这边是麻烦您跟我说一下联系的哪家店呀？我这边查一下店内信息。
客户：嗯，岳阳。

-客户：我快点啊，我喝酒，我这是喝酒喝多了。

-客户：就刚刚我现在要我现在要准备开车，但是这个车就。 发动不了了呀。

-客服：我需要现在需要收集一下您的这个信息，我这边给您反馈到后台，帮您核实一下看看具体是什么问题。
客户：嗯，还有还有那个胎压也不显示。

-客服：您现在还在店里吗先生？
客户：在。

场景2.客户表达观点（客户对某事/物进行积极/消极/中立的评价或阐述）
此场景下，需输出以“理解”/“明白”为核心词进行扩展句子的垫话。

场景对话举例：
-客户：这边这个销售就那种一副态度很高的一样。看不起人，感觉就是买不起车，穷光蛋过来。 你们这边奔驰的店。

场景3.客户骂人（客户使用脏话进行辱骂）
此场景下，需输出以“理解您的感受”/“我懂您的心情”为核心词进行扩展句子的垫话。

场景对话举例：
-客户：奔驰服务态度这样吗，我操。

-客户：对他妈非常态度非常恶劣，我操。 这提那批的，我操，让我打这个打那个，我现在已经打烦了，有点。

场景4.判断（客户对坐席的问题或已知情况或建议进行是/否判断）
此场景下，需输出以“知道了”/“收到”为核心词进行扩展句子的垫话。

场景对话举例：
-客服：好张先生是这样，如果他也绑定您的这个的话，是登录您的0709结尾手机号码验证码登不上去的对吗。
客户：对，应该是吧。

-客服：哦，了解您是湖州之星这个安吉分公司。
客户：对对对对对。

-客服：呃，来锁就是锁车的时候也是用APP锁的。
客户：是的是的。

-客服：如果您车辆丢失的话，建议您报警的。
客户：不行。


场景5.不满（客户最后一句话对某事/物表达出负面情绪）
此场景下，需输出以“我理解您的心情”/“明白您的感受”为核心词进行扩展句子的垫话。

场景对话举例：
-客户：如果我今天之内再看不到你们有任何合理的解决方案。 我我我我我不是恐吓你们，我们也会同步向你们的竞争对手，比如说宝马啊，那个奥迪，去公开你们这个4S店做这种龌龊的行为。 好吧，我。喂。 已经给足机会你们了。

-客户：这不忽悠人呢吗？给我让我买个套餐，完了忽悠我呗。

-客户：嗯，踢皮球啊，没给我操作，我还得还。 连连续再跑一遍吗。

场景6.客户作出指示（客户最后一句话对坐席作出要求/指令，或客户表示自己下一步的行动）
此场景下，需输出以“清楚了”/“好的”为核心词进行扩展句子的垫话。

场景对话举例：
-客户：那你那个什么呀？那你让他给我回电话吧。

-客户：您这边可不可以帮我尽快的跟我的4S店联系一下，让他们跟我。 处理一下售后的问题。

-客户：我给您看一下啊稍等马上一分钟。

-客户：所以，我是想去把这个问题解决。然后，我今天出差了，车就在4S店，大概一周的时间，厂家能不能安排一个技术人员？

-客户：那10分钟之后不给我回复，我再给你打电话。

场景7.疑问（客户最后一句话提出问题）
此场景下，需直接输出“无”。

场景对话举例：
-客户：有查到吗？

-客户：我想问一下，你们这边的员工现在都是什么人？都招什么？"

-客户：哦要重新登录对吗。

-客户：那就是APP的问题是吧？

场景8.语气词（客户最后一句话仅使用单纯语气词：“嗯”、“啊”、“哦”、“呃”等；）
此场景下，需直接输出“无”。

场景对话举例：
-客服：好的，那正在为您预约，请稍后。
客户：嗯。

场景9.残缺（客户最后一句话严重不完整、表达断续，无有效信息；）
此场景下，需直接输出“无”。

场景对话举例：
-客服：上午好呢，上海奔驰客服中心，我是张硕，请问有什么可以帮您，您好。
客户：喂，你好，我想问一下就是。


## 五、标注标准（红线禁止项，出现即严重错误）
1. **严禁复制、复述客户任何原话**
绝对不能输出客户的问句、陈述句。
2. **严禁一次输出多条结果**
每条对话**只输出一个结果**：禁止多句、多条、列表输出。
3. **严禁无标点**
除“无”以外，禁止输出无标点符号的结果。
4. **专业中性，不发表观点**
不出现“是的”“不行”“不对”等判断词。
5.**贴合客户情绪，轻度承接，不做解答**
"""

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

class CushionVLLMClient:
    """vLLM垫话生成客户端"""

    def __init__(self, base_url="http://localhost:5010"):
        self.base_url = base_url
        self.system_prompt = SYSTEM_PROMPT_V3.strip()
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
        return text.strip()

    def generate_cushion(self, system_prompt: str, conversation_history: str, timeout=30) -> str:
        """生成单条垫话"""
        # 判断输入是否为空或无效
        if not conversation_history.strip():
            return "无"
        # 替换对话中的角色标签，确保输入格式统一
        conversation_history = conversation_history.replace("Client:", "客户：").replace("Service:", "客服：")

        # 构造消息 - 在 system prompt 末尾添加 /no_think 关闭思考模式
        messages = [
            {"role": "system", "content": system_prompt + "\n/no_think"},
            {"role": "user", "content": conversation_history}
        ]

        payload = {
            "model": "qwen3-4b_sft",
            "messages": messages,
            "max_tokens": 10,  # 垫话很短，50足够
            "temperature": 0.8, # 适度创造性，避免千篇一律
            "top_p": 0.9,
            "stream": False,
            "frequency_penalty": 0.8,  # 惩罚重复
            "presence_penalty": 0.6,   # 鼓励多样性
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=timeout
            )
            latency = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                cushion = result["choices"][0]["message"]["content"]

                # 清理空的 <think\> 标签
                cushion = self._clean_think_tags(cushion)

                # 提取token统计
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # 统计耗时（线程安全）
                with self._lock:
                    self.request_count += 1
                    self.total_latency += latency
                    self.latencies.append(latency)
                    avg_latency = self.total_latency / self.request_count

                print(f"✅ 生成成功 | 延迟: {latency:.3f}s | 平均延迟: {avg_latency:.3f}s | prompt: {prompt_tokens} | 垫话: {completion_tokens}")
                return cushion
            else:
                print(f"❌ 请求失败: {response.status_code} - {response.text}")
                return "无"

        except Exception as e:
            print(f"❌ 请求异常: {e}")
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
                    print(f"📦 处理进度: {completed}/{total}")

        print(f"📦 处理完成: {total}/{total}")
        return results



def eval_cushion_generation(client: CushionVLLMClient, excel_path: str = "../data/0410翠翠测试数据.xlsx"):
    """
    读取Excel文件中的声道(说话的人)、文本(说话内容)，在每次客户说话时都去调用垫话接口，
    每次请求最多4句历史对话，组装成 conversation_history 传入 generate_cushion 接口，
    生成垫话内容，输出到 C 列。

    Args:
        client: CushionVLLMClient实例
        excel_path: Excel文件路径，默认为 ../data/0410翠翠测试数据.xlsx
    """
    from openpyxl import load_workbook
    import os

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        print(f"文件不存在: {excel_path}")
        return

    # 加载Excel文件
    print(f"📖 正在读取文件: {excel_path}")
    wb = load_workbook(excel_path)
    ws = wb.active

    # 获取总行数
    total_rows = ws.max_row
    print(f"总行数: {total_rows}")

    # 用于存储对话历史的滑动窗口
    history_window = []  # 存储最近4句对话
    max_history = 4

    # 统计
    cushion_count = 0

    # 从第2行开始遍历（第1行是表头）
    for row_idx in range(2, total_rows + 1):
        speaker = ws.cell(row=row_idx, column=1).value  # A列：声道
        text = ws.cell(row=row_idx, column=2).value      # B列：文本

        if not speaker or not text:
            continue

        speaker = str(speaker).strip()
        text = str(text).strip()

        # 构建当前对话行
        if speaker == "customer":
            # 客户说话，先加入历史
            history_window.append(f"客户：{text}")

            # 保持滑动窗口大小
            while len(history_window) > max_history:
                history_window.pop(0)

            # 组装对话历史，并在末尾添加客服标签提示生成垫话
            conversation_history = "\n".join(history_window) + "\n客服："

            # 调用垫话接口
            cushion = client.generate_cushion(conversation_history)

            # 写入C列
            ws.cell(row=row_idx, column=3, value=cushion)
            cushion_count += 1

            print(f"行{row_idx}: 客户说话 -> 生成垫话: {cushion}")

        elif speaker == "agent":
            # 客服说话，加入历史
            history_window.append(f"客服：{text}")

            # 保持滑动窗口大小
            while len(history_window) > max_history:
                history_window.pop(0)

    # 保存文件
    output_path = excel_path.replace(".xlsx", "_output_0413.xlsx")
    wb.save(output_path)
    print(f"\n✅ 处理完成！共生成 {cushion_count} 条垫话")
    print(f"📁 结果已保存至: {output_path}")



def test_train_data_generation(client: CushionVLLMClient):
    """
    读取0410垫话标注合并_4800.xlsx文件中的C列文本(历史对话)，调用generate_cushion接口生成垫话内容，将生成的垫话内容插入到第F列，
    并判断原有D列的内容与 新生成的F列内容是否一致，在G列输出对比结果：一致/不一致。
    """
    from openpyxl import load_workbook
    import os

    excel_path = "../data/0410垫话标注合并_4800.xlsx"

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        print(f"文件不存在: {excel_path}")
        return

    # 加载Excel文件
    print(f"📖 正在读取文件: {excel_path}")
    wb = load_workbook(excel_path)
    ws = wb.active

    # 获取总行数
    total_rows = ws.max_row
    print(f"总行数: {total_rows}")

    # 统计
    total_count = 0
    match_count = 0

    # 写入表头
    ws.cell(row=1, column=6, value="模型生成垫话")
    ws.cell(row=1, column=7, value="对比结果")

    # 从第2行开始遍历（第1行是表头）
    for row_idx in range(2, total_rows + 1):
        history_text = ws.cell(row=row_idx, column=2).value  # C列：历史对话
        original_result = ws.cell(row=row_idx, column=4).value  # D列：标注结果

        # if row_idx ==10:
        #     break

        if not history_text:
            continue

        history_text = str(history_text).replace("Client:", "客户：").replace("Service:", "客服：").strip() + "\n客服："


        # 调用垫话接口
        generated_cushion = client.generate_cushion(history_text)

        # 写入F列
        ws.cell(row=row_idx, column=6, value=generated_cushion)

        # 对比D列和F列内容
        original_result_str = str(original_result).strip() if original_result else ""

        # 对比时去除空格和标点差异，进行宽松匹配
        def normalize_text(text):
            """标准化文本用于对比"""
            import re
            # 去除空白字符
            text = re.sub(r'\s+', '', text)
            # 统一标点符号
            text = text.replace('，', '').replace('。', '').replace('！', '').replace('？', '')
            text = text.replace(',', '').replace('!', '').replace('?', '')
            return text.lower()

        normalized_original = normalize_text(original_result_str)
        normalized_generated = normalize_text(generated_cushion)

        # 判断是否一致
        if normalized_original == normalized_generated:
            compare_result = "一致"
            match_count += 1
        else:
            compare_result = "不一致"

        # 写入G列
        ws.cell(row=row_idx, column=7, value=compare_result)
        total_count += 1

        print(f"行{row_idx}: 原标注=[{original_result_str}] | 生成=[{generated_cushion}] | {compare_result}")

    # 保存文件
    output_path = excel_path.replace(".xlsx", "_0413_sft_comp_prompt_output.xlsx")
    wb.save(output_path)

    # 输出统计
    accuracy = (match_count / total_count * 100) if total_count > 0 else 0
    print(f"\n✅ 处理完成！共处理 {total_count} 条数据")
    print(f"📊 一致率: {accuracy:.2f}% ({match_count}/{total_count})")
    print(f"📁 结果已保存至: {output_path}")



def test_data_generation(client: CushionVLLMClient, max_workers=4):
    """
    读取test_0414_720_comp_prompt.json文件中的数据，读取每条数据中的 messages 列表，找到system角色的content作为system_prompt，找到user角色的content作为conversation_history，调用generate_cushion接口生成垫话内容，
    将 生成的垫话内容与原有assistant角色的content进行对比，输出对比结果：一致/不一致。
    最后将 system_prompt、conversation_history、原有垫话内容、生成的垫话内容、对比结果 输出到一个新的 xlsx 文件中 test_0414_720_comp_prompt_output.xlsx 中。

    Args:
        client: CushionVLLMClient实例
        max_workers: 最大并发数，默认为4
    """
    from openpyxl import Workbook
    import os

    # json_path = "../data/0417_dianhua_test_complex_prompt.json"
    json_path = "../data/0417_dianhua_test_complex_prompt.json"
    # json_path = "../data/0417_dianhua_train_complex_prompt.json"
    # json_path = "../data/0420_dianhua_test_simple_prompt.json"


    output_path = "../data/v12_0428_dianhua_test_complex_prompt_output.xlsx"
    #output_path = "../data/0417_dianhua_train_complex_prompt_2000_output.xlsx"
    #output_path = "../data/0417_dianhua_train_complex_prompt_output.xlsx"
    print(f" 输出结果文件: {output_path}")
    print(f" 并发数: {max_workers}")

    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    print(f"📖 共读取 {len(data_list)} 条数据")

    wb = Workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="system_prompt")
    ws.cell(row=1, column=2, value="conversation_history")
    ws.cell(row=1, column=3, value="原有垫话内容")
    ws.cell(row=1, column=4, value="生成的垫话内容")
    ws.cell(row=1, column=5, value="对比结果")

    def normalize_text(text):
        text = re.sub(r'\s+', '', text)
        text = text.replace('，', '').replace('。', '').replace('！', '').replace('？', '')
        text = text.replace(',', '').replace('!', '').replace('?', '')
        return text.lower()

    total_count = 0
    match_count = 0

    for idx, item in enumerate(data_list):
        messages = item.get("messages", [])
        if idx >= 2000:
            break

        system_prompt = ""
        conversation_history = ""
        original_cushion = ""

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                conversation_history = msg["content"]
            elif msg["role"] == "assistant":
                original_cushion = msg["content"]

        if not conversation_history:
            continue

        generated_cushion = client.generate_cushion(system_prompt, conversation_history)

        norm_original = normalize_text(original_cushion)
        norm_generated = normalize_text(generated_cushion)
        compare_result = "一致" if norm_original == norm_generated else "不一致"
        if compare_result == "一致":
            match_count += 1
        total_count += 1

        row = idx + 2
        ws.cell(row=row, column=1, value=system_prompt)
        ws.cell(row=row, column=2, value=conversation_history)
        ws.cell(row=row, column=3, value=original_cushion)
        ws.cell(row=row, column=4, value=generated_cushion)
        ws.cell(row=row, column=5, value=compare_result)

        print(f"[{idx+1}/{len(data_list)}] 原有=[{original_cushion}] | 生成=[{generated_cushion}] | {compare_result}")
    

    wb.save(output_path)

    accuracy = (match_count / total_count * 100) if total_count > 0 else 0
    print(f"\n✅ 处理完成！共处理 {total_count} 条数据")
    print(f"📊 一致率: {accuracy:.2f}% ({match_count}/{total_count})")
    print(f"📁 结果已保存至: {output_path}")

    # 打印统计信息
    print(f"\n📈 请求统计信息: {client.get_stats()}")

    # 打印统计信息
    print(f"\n📈 请求统计信息: {client.get_stats()}")


# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    client = CushionVLLMClient("http://192.168.86.3:5020")

    # 并发数配置
    max_workers = 8  # 可以根据需要调整并发数

    # 测试训练数据生成：读取Excel文件中的历史对话，生成垫话，并与原有标注结果进行对比，输出一致/不一致结果
    # test_train_data_generation(client)

    # 测试集验证效果
    test_data_generation(client, max_workers=max_workers)
    # 评估模式：处理Excel文件
    # eval_cushion_generation(client, max_workers=max_workers)

    # 测试模式：单条生成
    # test_conv = """客服：您好，我是一汽大众的官方客服，看到您近期有在app上关注过我们的车，是吗？\n客户：诶，是的 \n客服：嗯~请问您是在关注我们的哪款车型呢？\n客户：最新的车""".strip()
    # test_conv =  """客服：上午好，欢迎致电梅赛德斯奔驰互联服务中心，我叫陈海燕，请问有什么可以帮您？\n客户：你们是泉州还是哪里的？"""
    # test_conv += "\n客服："  # 在对话末尾添加客服标签，提示模型生成垫话
    # cushion = client.generate_cushion(test_conv)
    # print(f"\n测试对话:\n{test_conv}")
    # print(f">>>>生成垫话: {cushion}")
    
    
