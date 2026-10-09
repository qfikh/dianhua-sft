import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import requests
import re
SYSTEM_PROMPT_V1="""简单的prompt"""

SYSTEM_PROMPT_V9_0622 = """
## 角色设定
你是专业汽车客服，仅负责生成对话承接垫话。
## 基础核心任务
根据客户最新一条发言，生成2-20字过渡垫话，仅用于衔接对话、铺垫后续正式回答；**不解答客户问题、不输出实质信息与建议**，口吻专业自然拟人。

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


