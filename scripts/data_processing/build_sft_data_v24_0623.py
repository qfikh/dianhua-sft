import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import requests
import re
SYSTEM_PROMPT_V1="""简单的prompt"""
SYSTEM_PROMPT_V2="""
以下是汽车客服和客户之间的对话数据，你是一个话术指导老师。现在需要你进行一次“垫话”。
“垫话”的意思是：在回答客户问题之前，先进行一次语言承接，比如“好的。”、“我明白您的意思。”、“嗯。”、“嗯嗯。”、“我明白了。”、“我知道了。”、“晓得了。”。

垫话的基本要求如下：
坐席的垫话核心作用是承接客户话语、体现专业礼貌、平稳过渡到正式解答，区别于日常闲聊，需贴合服务岗位的规范与调性，标准如下：
（1）简洁短句，不拖沓
控制在 1-10 字，不占用沟通时长，不打断对话节奏，避免长篇铺垫。
（2）要灵活多变，根据不同的场景输出不同的垫话，不要只输出一种说法，如大部分都输出“我明白了。”是不可以的。可以输出：“嗯。”“嗯嗯。”，“嗯好。”，“我明白了。”，“我懂您的意思。”，“了解了。”，“我清楚了。”，“晓得了。”，“我理解您的情况。”，“嗯，我晓得了。”，“明白您的诉求。”，“我知晓了。”，“了解。”，“我理解啦。”，“清楚了。”，“我明白。”，“知晓您的问题。”，“明白您的情况了。”，“我了解。”，“我懂啦。”，“了解您的诉求了。”，“明白。”，“了解您的问题了。”，“我知悉了。”，“嗯，明白。”，“我已了解您的情况。”，“您的问题我知晓了。”，“明白您的意思。”，“我清楚您的问题了。”，“您的意思我了解了。”，“我知晓您的诉求了。”，“您的情况我明白了。”，“我理解您的感受。”，“您的需求我了解了。”等。
（3）可以输出语气词，如“嗯。”、“嗯嗯。”、“嗯好。”等，但不能输出“哦”。
（4）精准承接，不脱节
贴合客户上一句的表述（疑问、否认、迷茫、急躁等），不凭空说垫话。
（5）专业中性，不情绪化
不随意夸张附和、不情绪化吐槽，保持坐席的职业中立性。
（6）轻度共情，适配情绪
客户疑惑、急躁时，垫话带轻微安抚感；客户正常沟通时，保持常规承接即可。
（7）不能带有观点，严禁对客户问题做出 “能”/“不能”/“可以”/“不可以”/“是的”/“不行”/“不对” 等任何明确答复性表述，不直接回应问题本身，保持专业礼貌，输出结果不可以是疑问句，不能出现问号。
（8）垫话必须带标点，每条结尾必须加句号，即使垫话只有一个字，比如“嗯”，也要加句号。
（9）输出垫话严禁重复或复述客户最后一句原话，不得对客户问题做出任何直接答复，仅使用规范专业的承接用语，保持礼貌、中性、无观点。
场景要求：
（1）如果客户只是说了语气词，比如“嗯”、“啊”等，要输出“无”。
（2）如只说 “对对对”“好的”“知道了”，无需额外承接垫话，要输出“无”。
（3）无有效内容可承接时，要输出“无”。
（4）只有零散词语，没有完整意思表达，不适合垫话，要输出“无”。
（5）如果客户没有说完话，如语句不完整、卡顿、只说一半、还在组织语言，没有形成完整问题或诉求，要输出“无”。/no_think
"""


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

SYSTEM_PROMPT_V6_0520 ="""你作为车联网客服话术专家，请根据对话为坐席生成一句仅用于语言承接的垫话（不表达观点、不指示动作、非问题回答）。"""

SYSTEM_PROMPT_V7_0526 ="""你是汽车客服坐席，基于客户最新发言生成一句垫话，仅作承接，不表达观点、不指示动作、不回答问题。/no_think"""

SYSTEM_PROMPT_V8_0602 = """你是汽车客服坐席，需要基于客户最新发言生成一句简短的承接话语（垫话）。

【核心任务】
仅作对话承接，保持对话连贯性，为后续服务动作做自然过渡。

【具体要求】
1. 功能纯粹：仅承接，不表达观点、不指示动作、不回答问题。
2. 内容中性：避免任何实质性内容，不提供信息、建议或解决方案。
3. 风格语气：专业客服口吻，自然流畅。
4. 字数限制：严格控制在2-15个字。

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
4. 字数硬性限制：生成的垫话长度约为2~20字，不可超长/过短

## 输出格式规范
仅输出最终完整承接语句（基础垫话+可选追加话术），无任何解释、前缀、分段、思考内容。/no_think """


def parse_history_to_messages(history):
    """将历史对话解析为多轮 messages 格式，确保 user 开头且无连续相同角色"""
    messages = []
    lines = history.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("客户："):
            messages.append({"role": "user", "content": line[3:].strip()})
        elif line.startswith("客服："):
            messages.append({"role": "assistant", "content": line[3:].strip()})

    # 1. 跳过开头连续的 assistant
    while messages and messages[0]["role"] != "user":
        messages = messages[1:]

    # 2. 合并连续相同角色（防止意外）
    merged = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += " " + m["content"]
        else:
            merged.append(dict(m))

    return merged


def clean_thinking_tags(text):
    """清理 <think></think> 标签，防止模型学习思考模式"""
    import re
    # 移除 <think>...</think> 及其内容
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return text.strip()


def classify_dianhua(dianhua):
    """将垫话分类，用于分层抽样"""
    dianhua = str(dianhua).strip()
    if len(dianhua) <= 4:
        return "短垫话"
    elif len(dianhua) <= 7:
        return "中垫话"
    elif len(dianhua) <= 10:
        return "中长垫话"
    else:
        return "长垫话"



def get_query_wait_cushions():
    """读取查询等待场景垫话数据

    从 Excel 文件中读取 "查询等待场景垫话" 列数据，返回列表。
    """
    import pandas as pd

    input_path = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618查询等待场景垫话100条.xlsx"

    df = pd.read_excel(input_path)

    if "查询等待场景垫话" not in df.columns:
        print(f"列名 '查询等待场景垫话' 不存在于输入文件中，当前列名: {df.columns.tolist()}")
        return []

    cushions = df["查询等待场景垫话"].dropna().tolist()
    print(f"共读取 {len(cushions)} 条查询等待场景垫话,cushions:{cushions}")

    return cushions





def generate_pre_sales_cushions():
    """为售前数据生成通用垫话+"我帮您查一下"，串行调用 Flask 接口"""
    import pandas as pd
    import random
    # 查询 等待场景垫话100条.xlsx 文件中的 "查询等待场景垫话" 列数据，将其放到一个列表中。
    query_wait_cushions = get_query_wait_cushions()

    input_path = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output-filtered.xlsx"
    output_path = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0625售前数据-交付数据-清洗后4220-output-filtered-dianhua.xlsx"
    api_url = "http://192.168.86.3:5011/generate"


    print(f"输入文件: {input_path}")
    print(f"API地址: {api_url}")

    # 读取 Excel
    df = pd.read_excel(input_path)

    if "文本" not in df.columns:
        print(f"列名 '文本' 不存在于输入文件中，请检查文件格式")
        return

    # 读取"文本"列的对话数据
    df_col = df["文本"].tolist()
    total = len(df_col)
    print(f"共读取 {total} 条数据")

    # 串行调用接口
    cushions = []
    for idx, dialog_text in enumerate(df_col):
        # if idx <= 6000:
        #     continue
        # 使用正则表达式统一替换各种格式的角色前缀
        dialog_text = re.sub(r'[Ss]ervice[：:]', '客服：', dialog_text)
        dialog_text = re.sub(r'[Cc]lient[：:]', '客户：', dialog_text)
        dialog_text = dialog_text.strip()
        # 将对话文本按行分割，转换为 dialog_history 列表
        dialog_history = [line.strip() for line in dialog_text.split("\n") if line.strip()]

        # print(f"\n正在处理第 {idx+1} 条数据 | 对话历史: {dialog_history}")
        payload = {
            "source": "batch",
            "session_id": "batch",
            "dialog_history": dialog_history
        }

        try:
            response = requests.post(api_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                # print(f"第 {idx+1} 条请求成功: {result}")
                cushion = result.get("data", {}).get("dianhua", "")
            else:
                print(f"第 {idx+1} 条请求失败: {response.status_code}")
                cushion = ""
        except Exception as e:
            print(f"第 {idx+1} 条请求异常: {e}")
            cushion = ""

        # 将垫话的句号替换成逗号，然后随机拼接查询等待场景垫话
        if cushion:
            cushion = cushion.rstrip("。") + "，" + "我帮您查一下。"
        print(f"第 {idx+1} 条请求完成 | 垫话: {cushion}")


        cushions.append(cushion)

        if (idx + 1) % 100 == 0:
            print(f"处理进度: {idx + 1}/{total}")

    # 补齐 cushions 到与 DataFrame 相同的长度
    cushions.extend([""] * (len(df) - len(cushions)))

    # 将垫话内容写入 Excel
    df["模型的结果"] = cushions

    # 规范化"文本"列中的角色前缀
    df["文本"] = df["文本"].apply(lambda x: re.sub(r'[Ss]ervice[：:]', 'Service:', str(x)) if pd.notna(x) else x)
    df["文本"] = df["文本"].apply(lambda x: re.sub(r'[Cc]lient[：:]', 'Client:', str(x)) if pd.notna(x) else x)

    # 保存到 Excel
    df.to_excel(output_path, index=False)

    print(f"\n处理完成！共处理 {total} 条数据")
    print(f"结果已保存至: {output_path}")



def build_sft_data(excel_file_path):
    # 读取 Excel 文件
    df = pd.read_excel(excel_file_path, sheet_name='标注结果')

    # 提取 B 列（历史对话）和 D 列（垫话内容）
    # B 列是第1列（0-indexed），D 列是第3列
    history_dialogues = df.iloc[:, 1]  # B 列
    dianhua_list = df.iloc[:, 3]    # D 列
    modify_dianhua_list = df.iloc[:, 6]  # G 列，修改后的垫话内容（如果有修改的话）

    print(f"共读取 {len(df)} 条数据")
    # 构建数据列表
    max_len = 0
    data_list = []
    labels = []  # 用于分层抽样的标签
    for history, dianhua, modify_dianhua in zip(history_dialogues, dianhua_list, modify_dianhua_list):
        # 跳过空行
        if pd.isna(history) or pd.isna(dianhua):
            continue
        # 只保留history对话的最后10行
        history = "\n".join(history.split("\n")[-10:])

        # 将history对话中 Client 替换为 客户， Service 替换为 客服
        history = history.replace("Client:", "客户：").replace("Service:", "客服：")
        history = history.replace("Client：", "客户：").replace("Service：", "客服：")


        # 解析历史对话为多轮 messages
        history_messages = parse_history_to_messages(history)

        max_len = max(max_len, len(history))

        # 构建完整的 messages：system + 多轮对话 + 最终垫话
        messages = [{"role": "system", "content": SYSTEM_PROMPT_V9_0622.strip()}]
        messages.extend(history_messages)
        
        #如果有修改过的垫话内容，优先使用修改过的内容，否则使用原来的内容
        dianhua_clean = str(dianhua) if pd.isna(modify_dianhua) else str(modify_dianhua)
        messages.append({"role": "assistant", "content": dianhua_clean})

        data = {"messages": messages}
        data_list.append(data)
        labels.append(classify_dianhua(dianhua))

    print("最长对话长度（字符数）：", max_len)
    print("垫话类型分布：", pd.Series(labels).value_counts().to_dict())

    # 按 8:1:1 分层抽样划分训练集、验证集、测试集
    train_data, temp_data, _, temp_labels = train_test_split(
        data_list, labels, test_size=0.2, random_state=42, stratify=labels
    )
    eval_data, test_data, _, _ = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    output_dir = os.path.dirname(excel_file_path)

    # 输出训练集
    train_path = os.path.join(output_dir, '../datasets/v27_0625_dianhua_train_sft_simple_prompt.json')
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    print(f"训练集（{len(train_data)}条）已保存到 {train_path}")

    # 输出验证集
    eval_path = os.path.join(output_dir, '../datasets/v27_0625_dianhua_eval_sft_simple_prompt.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)
    print(f"验证集（{len(eval_data)}条）已保存到 {eval_path}")

    # 输出测试集
    test_path = os.path.join(output_dir, '../datasets/v27_0625_dianhua_test_sft_simple_prompt.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"测试集（{len(test_data)}条）已保存到 {test_path}")



# 使用示例
if __name__ == "__main__":
    # get_query_wait_cushions()

    # 为售前咨询三个场景生成通用垫话
    # generate_pre_sales_cushions()
    
    # v6 相比v5 主要改动：替换嗯 到多个字；针对售前数据拉长垫话内容。
    # excel_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0623垫话标注-平衡分布-交付数据-17355_分类结果.xlsx"  # 实际文件名
    # excel_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0624垫话标注-替换嗯-交付数据-18354_分类结果.xlsx"  # 实际文件名
    excel_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0625垫话标注-替换嗯-交付数据-18352_分类结果.xlsx"  # 优化了号牌结束语感谢语、售前三场景的垫话

    # 处理成标准的

    build_sft_data(excel_file)

# 第一步：为售前咨询三个场景生成通用垫话 + 固定“我帮您查一下。”
# 第二步：scripts/modify_origin_data.py  # 对原始标注数据进行修改，结合客户的最后一句话进行分类，判断是否包含结束语、感谢语、号牌等信息，并打上对应的标签。
# 第三步：scripts/build_sft_data_v6_0618.py # 增加分层抽样的功能，确保训练集、验证集、测试集中垫话类型的分布相似，且每个类型都有足够的样本。并且在构建数据时，优先使用修改过的垫话内容（如果有修改的话），否则使用原来的内容。最后输出的数据格式是符合 v8 版本 prompt 的格式。


