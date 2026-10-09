
DialogLabelPrompt = """
## 【任务目标】
请根据给定的对话片段（包含客服和客户的完整对话轮次），基于对话上下文逻辑，判断该片段是否符合以下四类标签之一：**结束语**、**感谢语**、**号牌**。若符合，返回对应标签；若不符合任何标签，返回"无标签"。分类时需严格遵循优先级顺序：**结束语 > 感谢语 > 号牌**

### 标签定义与判定规则（结合上下文）
#### 1. 结束语
- 客户的最后一句话明确表示结束对话的意图，且上下文显示对话已自然终止（例如问题已解决、需求已完成）。
- 典型结束语包括："再见"、"挂了吧"、"拜拜"、"没事了，结束吧"等。
- 必须满足上下文条件：对话前文已显示问题闭环（例如客服已解决问题，或客户确认需求已满足）。
- 排除：若客户使用礼貌用语（如"谢谢"）但未明确表达结束，且对话可能继续，则不归为此类。
- 结束语判定必须严格基于客户最后一句话的内容，不能因为前文出现过结束语关键词就判定为结束语。如果最后一句是新问题或新诉求，即使前文出现过'再见'，也不应判定为结束语。

#### 2. 感谢语
- 必须是客户最后一句话，且核心表意是表达谢意，常见内容包含：谢谢、多谢、辛苦了、麻烦你了、辛苦啦、多谢帮忙等致谢语句。
- 上下文适配场景：业务沟通、问题解答、协助处理完成后客户表达感激；可搭配简单语气助词、标点。
- 排除：
  1. 致谢语句同时带有结束对话意图的，按高优先级判定为**结束语**；
  2. 感谢内容出现在对话中间，并非客户最后一句话；
  3. 仅客套铺垫、无实际致谢含义的句子不纳入。

#### 3. 号牌
- 判定条件：客户最后一句话必须包含完整或部分手机号、车牌号、车架号信息，（示例：手机号138*1234、手机号是138、3256、车牌是京A12345、或者大写数字时比如一三八五）；
- 重要：判定必须严格基于客户最后一句话的内容，不能因为前文对话中出现过数字/号码就判定为号牌。如果最后一句是语气词、普通文字或不含数字/字母的短句，即使前文有号码信息，也不应判定为号牌。
- 排除：客户最后一句只是提到了"号码"、"车牌"、"车架号"，但没有提供具体数字/字母时，不应该归为号牌这一类。
- 排除：省份简称（如浙、京、沪）后跟的是地名、店名、人名等非车牌号格式的内容，不应归为号牌。车牌号必须符合"省份简称+字母+数字/字母"的格式（如浙A12345）。
- 排除：客户只是提到"手机号码"、"车牌"等词汇，但没有提供具体的数字或字母组合，即使是在讨论号码相关话题，也不应判定为号牌。例如"手机号码没用了"、"号码换了"等表述，只是在描述情况，不是在提供号码信息。

### 优先级与冲突处理规则
- 若对话同时符合多个标签，按优先级顺序确定最终标签： 感谢语 > 结束语 > 号牌。例如，若一句同时兼具结束+感谢，优先判定为感谢语。
- 若对话不符合任何标签定义，返回"无标签"。
"""


import pandas as pd
from qwen import call_with_messages


def truncate_lines(text, n=10):
    """保留最后n行"""
    if pd.isna(text) or not str(text).strip():
        return text
    return "\n".join(str(text).strip().split("\n")[-n:])


def classify_conversation(conversation):
    """调用 qwen 对对话进行分类"""
    prompt = DialogLabelPrompt + "\n\n## 对话内容\n" + conversation + "\n\n请直接输出标签名称，不要输出任何解释。"
    result = call_with_messages(
        model="qwen-plus",
        system_prompt=prompt,
        user_content="请分类。"
    )
    return result.strip()


def save_to_excel(df, classifications, modified_cushions, output_file, sheet_name, conversation_col):
    """保存当前处理结果到 Excel"""
    # 补充空值，使长度与 DataFrame 行数匹配
    temp_classifications = classifications + [""] * (len(df) - len(classifications))
    temp_cushions = modified_cushions + [""] * (len(df) - len(modified_cushions))

    # 创建 DataFrame 副本用于保存
    df_save = df.copy()
    df_save[conversation_col] = df_save[conversation_col].apply(truncate_lines)
    df_save["分类"] = temp_classifications
    df_save["修改后的垫话"] = temp_cushions

    # 保存到文件
    df_save.to_excel(output_file, sheet_name=sheet_name, index=False)
    print(f"已保存 {len(classifications)} 条处理结果到: {output_file}")


def classify_and_modify_cushions():
    """对对话数据进行分类并生成垫话：号牌/结束语/感谢语/无标签

    功能说明：
    1. 读取 Excel 文件中的对话数据
    2. 遍历每行对话，提取客户最后一句话
    3. 根据客户最后一句话的内容进行预过滤
    4. 调用 LLM 对符合条件的对话进行分类（号牌/结束语/感谢语）
    5. 根据分类结果生成对应的垫话
    6. 每处理1000条保存一次进度，最终保存完整结果

    输出：
    - 分类结果列：号牌/结束语/感谢语/无标签
    - 修改后的垫话列：嗯嗯。/好的。/不客气。/空
    """
    excel_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0616垫话标注-替换嗯-交付数据-15362_分类结果.xlsx" #有少量分的不太准
    # 输出文件路径
    output_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0625垫话标注-替换嗯-交付数据-15362_分类结果.xlsx"
    sheet_name = "标注结果"

    # 读取 Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"共读取 {len(df)} 条数据")
    print(f"列名: {df.columns.tolist()}")

    # 获取对话内容列（B列，索引1）
    conversation_col = df.columns[1]
    
    # 新建"分类"列和"修改后的垫话"列
    classifications = []
    modified_cushions = []

    for idx, row in df.iterrows():
        # if idx >= 10:
        #     break
        try:
            conversation = row[conversation_col]
            # 跳过空行
            if pd.isna(conversation) or not str(conversation).strip():
                classifications.append("")
                continue
            conversation = str(conversation).strip()
            # 只保留最后6句对话，节省 token
            lines = conversation.split("\n")
            conversation = "\n".join(lines[-6:])

            # 获取最后一句话
            last_line = conversation.strip().split("\n")[-1] if conversation.strip() else ""
            # 提取客户最后一句话（去掉"客户："前缀）
            if last_line.startswith("客户："):
                last_customer_line = last_line[3:].strip()
            else:
                last_customer_line = last_line.strip()

            # 预过滤：如果客户最后一句话既不包含数字、也不包含告别词、也不包含感谢词，则跳过分类
            if (not any(c.isdigit() or c in "一二三四五六七八九十零" for c in last_customer_line)
                and not ("拜" in last_customer_line or "再见" in last_customer_line or "先这样" in last_customer_line or "就这样" in last_customer_line or "没有" in last_customer_line or "没了" in last_customer_line)
                and not ("谢" in last_customer_line or "辛苦" in last_customer_line)):
                modified_cushions.append("")
                classifications.append("无标签")

            else:
                # 调用分类
                result = classify_conversation(conversation)
                print(f"第 {idx+1} 行分类结果: {result}")
                classifications.append(result)

                # 根据分类结果修改模型结果列的垫话内容
                if result == "号牌" and any(c.isdigit() or c in "零一二三四五六七八九" for c in last_customer_line):
                    modified_cushions.append("嗯嗯。")
                elif result == "结束语" and ("拜" in last_customer_line or "再见" in last_customer_line or "先这样" in last_customer_line or "就这样" in last_customer_line or "没有" in last_customer_line or "没了" in last_customer_line):
                    modified_cushions.append("好的。")
                elif result == "感谢语" and ("谢" in last_customer_line or "辛苦" in last_customer_line):
                    modified_cushions.append("不客气。")
                else:
                    modified_cushions.append("")

        except Exception as e:
            print(f"第 {idx+1} 行分类失败: {e}")
            classifications.append("分类失败")

        # 每处理1000条保存一次（带进度后缀，不覆盖）
        # if (idx + 1) % 1000 == 0:
        #     print(f"已处理 {idx+1}/{len(df)} 条")
        #     progress_file = output_file.replace(".xlsx", f"_{idx+1}.xlsx")
        #     save_to_excel(df, classifications, modified_cushions, progress_file, sheet_name, conversation_col)

    # 最终保存完整结果
    save_to_excel(df, classifications, modified_cushions, output_file, sheet_name, conversation_col)
    print(f"\n处理完成！结果已保存至: {output_file}")

    # 统计分类结果
    from collections import Counter
    counter = Counter(classifications)
    print("\n分类统计:")
    for item, count in counter.most_common():
        print(f"  {item}: {count}")




def update_data():
    """更新数据：对F列为空的行进行分类处理"""
    excel_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0616垫话标注-替换嗯-交付数据-19582_分类结果.xlsx"
    sheet_name = "标注结果"

    # 读取 Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"共读取 {len(df)} 条数据")
    print(f"列名: {df.columns.tolist()}")

    # 获取对话内容列（B列，索引1）
    conversation_col = df.columns[1]
    # 分类列（F列，索引5）
    classify_col = df.columns[5]
    # 垫话列（G列，索引6）
    cushion_col = df.columns[6]

    # 输出文件路径
    output_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0616垫话标注-替换嗯-交付数据-19582_分类结果_test.xlsx"

    # 统计待处理数量
    to_process = df[df[classify_col].isna() | (df[classify_col] == "")].shape[0]
    print(f"待处理行数: {to_process}")

    processed_count = 0

    for idx, row in df.iterrows():
        # if idx >= 2000:
        #     break
        # 如果 F 列已有值，跳过
        if pd.notna(row[classify_col]) and str(row[classify_col]).strip():
            continue
        try:
            conversation = row[conversation_col]
            # 跳过空行
            if pd.isna(conversation) or not str(conversation).strip():
                df.at[idx, classify_col] = ""
                continue
            conversation = str(conversation).strip()
            # 只保留最后6句对话，节省 token
            lines = conversation.split("\n")
            conversation = "\n".join(lines[-6:])
            # 获取最后一句话
            last_line = conversation.strip().split("\n")[-1] if conversation.strip() else ""
            # 提取客户最后一句话（去掉"客户："前缀）
            if last_line.startswith("客户："):
                last_customer_line = last_line[3:].strip()
            else:
                last_customer_line = last_line.strip()

            # 预过滤：如果客户最后一句话既不包含数字、也不包含告别词、也不包含感谢词，则跳过分类
            if (not any(c.isdigit() or c in "一二三四五六七八九十零" for c in last_customer_line)
                and not ("拜" in last_customer_line or "再见" in last_customer_line or "这样" in last_customer_line or "没有" in last_customer_line or "没了" in last_customer_line)
                and not ("谢" in last_customer_line or "辛苦" in last_customer_line)):
                df.at[idx, cushion_col] = ""
                df.at[idx, classify_col] = "无标签"

            else:
                # 调用分类
                result = classify_conversation(conversation)
                print(f"第 {idx+1} 行分类结果: {result}")
                df.at[idx, classify_col] = result

                # 根据分类结果修改模型结果列的垫话内容
                if result == "号牌" and any(c.isdigit() or c in "一二三四五六七八九十零" for c in last_customer_line):
                    df.at[idx, cushion_col] = "嗯嗯。"
                elif result == "结束语" and ("拜" in last_customer_line or "再见" in last_customer_line or "这样" in last_customer_line or "没有" in last_customer_line or "没了" in last_customer_line):
                    df.at[idx, cushion_col] = "好的。"
                elif result == "感谢语" and ("谢" in last_customer_line or "辛苦" in last_customer_line):
                    df.at[idx, cushion_col] = "不客气。"
                else:
                    df.at[idx, cushion_col] = ""

            processed_count += 1

        except Exception as e:
            print(f"第 {idx+1} 行分类失败: {e}")
            df.at[idx, classify_col] = "分类失败"

        # 每处理1000条保存一次
        if processed_count > 0 and processed_count % 1000 == 0:
            print(f"已处理 {processed_count}/{to_process} 条")
            progress_file = output_file.replace(".xlsx", f"_{processed_count}.xlsx")
            df.to_excel(progress_file, sheet_name=sheet_name, index=False)
            print(f"已保存进度到: {progress_file}")

    # 最终保存完整结果
    df.to_excel(output_file, sheet_name=sheet_name, index=False)
    print(f"\n处理完成！共处理 {processed_count} 条，结果已保存至: {output_file}")

    # 统计分类结果
    from collections import Counter
    counter = Counter(df[classify_col].dropna().tolist())
    print("\n分类统计:")
    for item, count in counter.most_common():
        print(f"  {item}: {count}")


if __name__ == "__main__":
    #对对话数据进行分类并生成垫话：号牌/结束语/感谢语/无标签
    classify_and_modify_cushions()
    

    # 对F列为空的行进行分类处理, 比如对于新加的数据没有F列的分类的
    # update_data()

