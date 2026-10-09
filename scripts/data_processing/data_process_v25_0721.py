import openpyxl
import random
from pathlib import Path
# from scripts.llm_optimize import query_classify


DATA_DIR = Path(__file__).parent.parent / "data" / "output"
EXCEL_FILE = DATA_DIR / "0522_V1617_不同结果对比.xlsx"
TARGET_SHEET = "Sheet"
SHUFFLE_SHEET = "shuffle"
SAMPLE_COUNT = 100

JUDGE_CUSHION_PROMPT = """
### 角色设定
你是一名资深的**客服话术质检专家**，你的任务是筛选出人工标注中**明显有问题**的垫话，放行基本合格的垫话。

### 任务背景
- **场景**：车联网客服对话。
- **垫话定义**：客服在正式回答前的承接用语，如"了解了"、"明白"、"我理解您的心情"等。

### 判定原则：宁可放过，不可错杀
- 只要垫话**没有明显问题**，就判定为【合理】
- 只有出现**明显硬伤**时，才判定为【不合理】

**以下情况判定为【不合理】（明显硬伤）：**
1. **逻辑不通**：垫话与客户发言完全无关，答非所问
2. **不自然**：明显机器腔、生硬拼接、不符合人类说话习惯
3. **错误回答**：垫话直接回答了客户问题（如"可以"、"不行"、"能"、"不能"）
4. **严重冗长**：垫话超过20字，完全不像承接用语
5. **场景严重错位**：客户在骂人，垫话却说"好的"；客户在陈述，垫话却反问

**以下情况仍判定为【合理】（允许的差异）：**
- 语气词差异（"嗯" vs "嗯嗯" vs "好的"）
- 表达繁简（"明白" vs "我明白了"）
- 同义替换（"了解" vs "知晓"）
- 字数在1-15字范围内的自然表达

### 输入数据
- **【对话历史】**：{conversation_history}
- **【客服的垫话】**：{cushion}

### 输出格式
- 如果合理，请直接输出"**合理**"
- 如果不合理，请输出"**不合理|原因**"，原因用一句话概括，如"不合理|垫话直接回答了客户问题"
"""



SCENE_CLASSIFY_PROMPT = """你是一个汽车领域客服专家，擅长将对话片段进行场景分类。请判断以下对话是否符合以下三种场景之一，且必须满足条件：对话结尾处用户提出了该类问题，且客服尚未有效回复。

## 三种场景定义

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

## 判断条件
1. 必须是对话结尾处用户的意图
2. 客服尚未有效回复该问题（如只是说"好的，稍等"这种垫话不算有效回复）

## 对话内容
{dialog}

## 输出要求
请只输出以下内容之一（不要输出其他任何文字）：
- 如果符合场景1（车型配置）：输出"车型配置"
- 如果符合场景2（车型对比）：输出"车型对比"
- 如果符合场景3（价格政策）：输出"价格政策"
- 如果不符合任何场景：输出"不符合"
"""




def shuffle_sample():
    """从 Sheet 读取数据，随机抽取 100 条，写入 shuffle 标签页"""
    wb = openpyxl.load_workbook(EXCEL_FILE)
    ws = wb[TARGET_SHEET]

    # 读取表头
    headers = [cell.value for cell in ws[1]]

    # 读取所有数据行（跳过表头）
    all_rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        all_rows.append(row)

    print(f"共读取 {len(all_rows)} 条数据")

    # 随机抽取
    sampled = random.sample(all_rows, min(SAMPLE_COUNT, len(all_rows)))
    print(f"随机抽取 {len(sampled)} 条数据")

    # 如果 shuffle 标签页已存在则删除
    if SHUFFLE_SHEET in wb.sheetnames:
        del wb[SHUFFLE_SHEET]

    # 创建 shuffle 标签页并写入数据
    ws_shuffle = wb.create_sheet(SHUFFLE_SHEET)
    ws_shuffle.append(headers)
    for row in sampled:
        ws_shuffle.append(list(row))

    wb.save(EXCEL_FILE)
    print(f"已写入 {SHUFFLE_SHEET} 标签页")




def judge_cushion_quality():
    """判断垫话质量

    读取 origin_data 目录下 0520垫话标注-交付数据-15362.xlsx 文件 "标注结果" sheet 页中的
    文本（B列）和标注结果（D列）两列的内容，文本是客服和客户的对话历史，标注结果是客服的垫话。
    调用 qwen-plus 来判断客服的垫话是否合理，将结果输出到 F 列。
    """
    from qwen import call_with_messages

    excel_file = Path(__file__).parent.parent / "data" / "origin_data" / "0520垫话标注-交付数据-15362.xlsx"
    sheet_name = "标注结果"

    wb = openpyxl.load_workbook(excel_file)
    ws = wb[sheet_name]

    # 读取所有行
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    print(f"共读取 {len(rows)} 条数据")

    # 写入表头
    ws.cell(row=1, column=6, value="垫话质量判定")
    ws.cell(row=1, column=7, value="不合理原因")

    count = {"合理": 0, "不合理": 0, "跳过": 0}

    for idx, row in enumerate(rows, start=2):
        conversation_history = row[1]  # B列：对话历史
        cushion = row[3]  # D列：标注结果（垫话）

        if idx > 2000:  # 先处理前 200 条，测试效果
            break

        # 跳过空行
        if not conversation_history or not cushion:
            ws.cell(row=idx, column=6, value="")
            count["跳过"] += 1
            continue

        conversation_history = str(conversation_history).strip()
        cushion = str(cushion).strip()

        # 只保留最后10句对话，节省 token
        lines = conversation_history.split("\n")
        conversation_history = "\n".join(lines[-10:])

        # 有一个为空时直接判定为不合理
        if not conversation_history or not cushion:
            ws.cell(row=idx, column=6, value="不合理")
            count["不合理"] += 1
            continue

        # 调用 qwen-plus 判断垫话是否合理
        judge_prompt = JUDGE_CUSHION_PROMPT.format(
            conversation_history=conversation_history,
            cushion=cushion
        )
        judge_result = call_with_messages(
            model="qwen-plus",
            system_prompt=judge_prompt,
            user_content="请判断客服的垫话是否合理。"
        )

        # 解析结果
        if "不合理" in judge_result:
            result = "不合理"
            # 提取原因（格式：不合理|原因）
            if "|" in judge_result:
                reason = judge_result.split("|", 1)[1].strip()
            else:
                reason = ""
            ws.cell(row=idx, column=7, value=reason)
        else:
            result = "合理"
            ws.cell(row=idx, column=7, value="")

        ws.cell(row=idx, column=6, value=result)
        count[result] += 1

        if idx % 100 == 0:
            print(f"已处理 {idx - 1}/{len(rows)} 条")

    wb.save(excel_file)
    print(f"\n处理完成！")
    print(f"合理: {count['合理']}, 不合理: {count['不合理']}, 跳过: {count['跳过']}")
    print(f"结果已保存至: {excel_file}")
    # 分析2千条样本后，明显不合理的20条，其中大部分是 我理解您的问题了、我了解您的疑惑了这类，我们认为是正常。所以整体数据质量是可以的。



def extract_dialog_from_logs():
    """从线上服务日志中提取对话片段

    从 data/data1/0616_dialog_history.log 中提取上文对话内容，
    保存到 data/data1/0616_extracted_dialog.xlsx 中，列名为"文本"。
    """
    import re
    import pandas as pd
    from collections import Counter

    # log_file = Path(__file__).parent.parent / "data" / "data1" / "0616_dialog_history.log"
    # output_file = Path(__file__).parent.parent / "data" / "data1" / "0616_extracted_dialog.xlsx"
    # 0721 第四版垫话优化
    log_file = "/Users/xujianjun/workspace/dianhua-sft/data/ren_ji_dialog_process/dialog_history_0602-0721.log"
    output_file = "/Users/xujianjun/workspace/dianhua-sft/data/ren_ji_dialog_process/extract_dialog_history_0721.xlsx"

    # 正则匹配：提取"上文对话内容:"后面的内容
    # 格式1: 以", 垫话内容"或", 修改后垫话内容"或", 原始模型垫话内容"结尾
    # 格式2: 以", generate方法总耗时"结尾
    # 格式3: 以", >>>>>"结尾
    pattern = re.compile(r'上文对话内容: (.*?)(?:[，,] (?:原始模型|修改后)?垫话内容[：:]|, generate方法总耗时|, >>>>>)')

    dialogs = []
    dianhuas= []
    skipped = 0
    with open(log_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            dianhua = ""
            # if i >= 20000:
            #     break
            if "客服" not in line:
                skipped += 1
                # print("无客服字眼，line:",line)
                continue
            match = pattern.search(line)
            if match:
                dialog = match.group(1).strip()
                if not dialog:
                    print("空对话内容 line:", line[:100])
                if dialog:  # 跳过空内容
                    # 按"客户："和"客服："进行换行，并去除角色后面的空格
                    dialog = re.sub(r'\s*(客户：|客服：)\s*', r'\n\1', dialog).strip()
                    # 按换行切分，去掉包含"用户诉求"的行
                    lines = [l.split("用户诉求")[0] if "用户诉求" in l else l for l in dialog.split('\n')]
                    # 过滤掉只有角色标签但没有实际内容的行（如"客户："或"客服："后无内容）
                    lines = [l for l in lines if not re.match(r'^(客户：|客服：)\s*$', l.strip())]
                    # 对话片段不超过10行，超过10行的做截断，只保留最后10行
                    if len(lines) > 10:
                        lines = lines[-10:]
                    if len(lines) < 2:
                        # print(" 对话内容太少 line:", line[:100])
                        skipped += 1
                        continue

                    dialog = '\n'.join(lines).strip()
                    # 过滤条件：包含"一起大众"且对话行数≤5，跳过整通对话
                    if "一汽大众" in dialog and len(lines) <= 5:
                        skipped += 1
                        continue
                    if "垫话" in dialog:
                        print("异常对话内容,不包含 垫话 dialog: ",dialog)
                    dialogs.append(dialog)
                    # 提取垫话内容
                    if "修改后垫话内容" in line:
                        dianhua = line.split('修改后垫话内容：')[1]
                    elif "垫话内容" in line:
                        dianhua = line.split('垫话内容:')[1]
                    else:
                        print("异常垫话内容：",line)
                        dianhua = ""
                    # 先清理日志标记（在按句号分割前处理）
                    for sep in ["上文对话内容", "generate方法总耗时", "vllm调用耗时", "总耗时:"]:
                        if sep in dianhua:
                            dianhua = dianhua.split(sep)[0]
                    # 按句号或换行截取垫话主体
                    for sep in ["。", "\n"]:
                        if sep in dianhua:
                            dianhua = dianhua.split(sep)[0]
                            break
                    dianhua = dianhua.strip()
                    # 如果dianhua 以逗号结尾，就去掉逗号
                    if dianhua.endswith('，') or dianhua.endswith(','):
                        dianhua = dianhua[:-1].strip()
                    if dianhua:
                        dianhua += "。"
                    dianhuas.append(dianhua)
            else:
                print("异常对话内容：不包含上文对话内容 line: ",line)
                
    # 对dialog相同，dianhua也相同的只保留一个，确保dialog和dianhua对应关系正确
    seen = set()
    unique_dialogs = []
    unique_dianhuas = []
    for dialog, dianhua in zip(dialogs, dianhuas):
        key = (dialog, dianhua)
        if key not in seen:
            seen.add(key)
            unique_dialogs.append(dialog)
            unique_dianhuas.append(dianhua)
    dialogs = unique_dialogs
    dianhuas = unique_dianhuas

    # 统计对话行数分布
    line_count_stats = Counter()
    for dialog in dialogs:
        line_count = len([l for l in dialog.split('\n') if l.strip()])
        line_count_stats[line_count] += 1
    print("对话行数分布:")
    for count in sorted(line_count_stats.keys()):
        print(f"  {count}行: {line_count_stats[count]} 条")

    # 保存到 xlsx
    df = pd.DataFrame({'对话片段': dialogs, '来源' : '人机', '初版垫话' : dianhuas})
    df.to_excel(output_file, index=False)

    print(f"共提取 {len(dialogs)} 条对话")
    print(f"跳过 {skipped} 条无'客服'的行")
    print(f"结果已保存至: {output_file}")



def filter_dialogs_by_scene():
    """分析标注的三类售前场景数据，并对每条对话进行场景分类

    从 CSV 文件读取对话，判断是否符合三种场景之一（车型配置、车型对比、价格政策），
    输出到 Excel 文件，符合的对话在右边一列标注"需要查询"。
    """
    from qwen import call_with_messages
    import pandas as pd

    input_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output.xlsx"
    output_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output-filtered.xlsx"

    # 读取 CSV
    df = pd.read_excel(input_file)
    print(f"共读取 {len(df)} 条对话")

    # 分类结果
    results = []
    count = 0

    for idx, row in df.iterrows():
        dialog = str(row['文本']).strip()
        # if idx > 200:  # 先处理前 200 条，测试效果
        #     break

        if not dialog or dialog == 'nan':
            results.append('')
            continue

        # 只取最后10句对话（节省 token）
        lines = dialog.split('\n')
        # 过滤掉<=3行的对话，避免过短的对话导致误判
        if len(lines) <= 3:
            results.append('不符合')
            continue
        # 如果最后一句以Service结尾，说明客服已经回复了，不符合查询场景
        if lines[-1].startswith('客服') or lines[-1].startswith('Service'):
            results.append('不符合')
            continue
        recent_dialog = '\n'.join(lines[-10:])

        # 调用 API 判断场景
        prompt = SCENE_CLASSIFY_PROMPT.format(dialog=recent_dialog)
        try:
            result = call_with_messages(
                model="qwen-plus",
                system_prompt=prompt,
                user_content="请判断对话场景。"
            ).strip()
            print(f"第 {idx+1} 条对话场景判断结果: {result}")

            if result in ["车型配置", "价格政策", "车型对比"]:
                results.append(result)
                count += 1
            else:
                results.append('不属于售前查询场景')
        except Exception as e:
            print(f"第 {idx+1} 条判断失败: {e}")
            results.append('')

        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 条")

    # 补齐 results 列表长度（因为 break 提前退出，results 长度可能小于 df 行数）
    results.extend([''] * (len(df) - len(results)))

    # 添加结果列
    df['场景筛选'] = results

    # 保存到 Excel
    df.to_excel(output_file, index=False)
    print(f"\n处理完成！")
    print(f"符合场景的对话: {count} 条")
    print(f"结果已保存至: {output_file}")


def sample_dialogs_by_scene():
    """按场景随机抽样对话

    从已分类的 Excel 文件中：
    - 车型对比：随机抽取 200 条
    - 车型配置：随机抽取 400 条
    - 价格政策：随机抽取 400 条
    保存到新的 Excel 文件，供后续分析使用。
    """
    import pandas as pd

    input_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output-filtered.xlsx"
    # output_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output-filtered-sampled1000.xlsx"
    # output_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output-filtered-sampled2000.xlsx"
    output_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output-filtered-sampled2992.xlsx"

    # 读取已分类的数据
    df = pd.read_excel(input_file)
    print(f"共读取 {len(df)} 条数据")

    # 抽样配置：场景 -> 抽取数量
    #sample_config = {"车型对比": 200, "车型配置": 400, "价格政策": 400}
    # sample_config = {"车型对比": 200, "车型配置": 900, "价格政策": 900}
    sample_config = {"车型对比": 2000, "车型配置": 2000, "价格政策": 2000} # 不采样全保留
    
    sampled_dfs = []
    stats = {}

    for scene, count in sample_config.items():
        # 筛选该场景的数据
        scene_df = df[df['场景筛选'] == scene]
        actual_count = len(scene_df)

        if actual_count == 0:
            print(f"警告：{scene} 场景没有数据")
            continue
        
        # 随机抽样（如果数据量不足，取全部）
        if actual_count >= count:
            sampled = scene_df.sample(n=count, random_state=42)
            stats[scene] = f"抽取 {count}/{actual_count} 条"
        else:
            sampled = scene_df
            stats[scene] = f"数据不足，取全部 {actual_count} 条"

        sampled_dfs.append(sampled)

    # 合并所有抽样结果
    if sampled_dfs:
        result_df = pd.concat(sampled_dfs, ignore_index=True)
        result_df.to_excel(output_file, index=False)

        print(f"\n抽样统计：")
        for scene, stat in stats.items():
            print(f"  {scene}: {stat}")
        print(f"\n总计抽取 {len(result_df)} 条数据")
        print(f"结果已保存至: {output_file}")
    else:
        print("没有找到符合条件的场景数据")


def filter_zhuanjie_dialogs():
    """ 筛选并过滤掉 "转接"或"转人工"的对话 """
    import pandas as pd

    input_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0625垫话标注-替换嗯-交付数据-18352_分类结果.xlsx"
    output_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0625垫话标注-替换嗯-交付数据-18352_分类结果_优化.xlsx"

    df = pd.read_excel(input_file)
    print(f"共读取 {len(df)} 条数据")

    # 筛选包含"转接"或"转人工"的对话
    filtered_df = df[df['文本'].str.contains('转接|转人工', na=False)]

    filtered_df.to_excel(output_file, index=False)
    print(f"\n筛选出 {len(filtered_df)} 条转接场景对话")
    print(f"结果已保存至: {output_file}")


def merge_renren_renji_dialogs():
    """ 合并 人人、人机对话数据 和 生成的售前场景数据 """
    import pandas as pd

    # 处理人机对话数据
    renji_file = "/Users/xujianjun/workspace/dianhua-sft/data/ren_ji_dialog_process/extract_dialog_history_0721.xlsx"
    renji_df = pd.read_excel(renji_file)
    print(f"共读取 {len(renji_df)} 条人机对话数据")


    # 读取 人人 和 生成的售前场景数据
    renren_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0722垫话标注-替换嗯-交付数据-18352_分类结果.xlsx"
    renren_df = pd.read_excel(renren_file)
    print(f"共读取 {len(renren_df)} 条人人和生成的售前场景数据")
    # 删除第一、三、五列的内容
    renren_df.drop(columns=[renren_df.columns[0], renren_df.columns[2], renren_df.columns[4]], inplace=True)
    # 将第1列的列名由文本改为 对话片段，
    renren_df.rename(columns={renren_df.columns[0]: '对话片段'}, inplace=True)
    # 在第二列新插入一列为来源，默认值为 人机
    renren_df.insert(1, '来源', '人人')
    # 将第3列的列名由模型结果改为 初版垫话
    renren_df.rename(columns={renren_df.columns[2]: '初版垫话'}, inplace=True)
    
    # 最终合并后的列名：对话片段、来源、初版垫话、分类、修改后的垫话 
    merged_df = pd.concat([renji_df, renren_df], ignore_index=True)
    # 将NaN替换为空字符串
    merged_df = merged_df.fillna('')

    # 针对场景分类列取值为 车型配置、价格政策、车型对比的 将该行的 来源列 标记为"生成" 
    merged_df.loc[merged_df['场景分类'].isin(["车型配置", "价格政策", "车型对比"]), '来源'] = '生成'

    # 处理异常数据：过滤掉对话过短、角色标注颠倒、末句是客服说的的数据
    drop_indices = []
    for idx, text in enumerate(merged_df['对话片段']):
        text = str(text).strip()
        lines = text.split("\n")
        if len(lines) < 2:
            drop_indices.append(idx)
            continue
        # 如果文本包含"转接"且 Client 行包含客服中心介绍，说明角色标注颠倒
        if "转接" in text and any(("Client" in line) and ("梅赛德斯奔驰客服中心" in line or "奔驰客服中心" in line) for line in lines):
            drop_indices.append(idx)
            continue
        # 最后一句话是客服说的（service/客服/Service），过滤掉
        last_line = lines[-1].strip()
        if last_line.startswith("客服") or last_line.startswith("Service") or last_line.startswith("service"):
            drop_indices.append(idx)
            continue
        # 客户最后一句话是"用户无输入"或只有角色标签无实际内容，过滤掉
        if "：" in last_line:
            last_content = last_line.split("：", 1)[-1].strip()
        elif ":" in last_line:
            last_content = last_line.split(":", 1)[-1].strip()
        else:
            last_content = last_line.strip()
        if "用户无输入" in last_line or not last_content:
            drop_indices.append(idx)
            continue
    if drop_indices:
        merged_df = merged_df.drop(index=drop_indices).reset_index(drop=True)
        print(f"过滤掉 {len(drop_indices)} 条异常数据（对话过短/角色颠倒/末句为客服/用户无输入）")
           
    
    # 保存到 Excel
    output_file = "/Users/xujianjun/workspace/dianhua-sft/data/origin_data/0722人人-人机-生成-整体待标注数据.xlsx"
    merged_df.to_excel(output_file, index=False)
    print(f"\n合并完成！")
    print(f"共合并 {len(merged_df)} 条数据")
    print(f"结果已保存至: {output_file}")










if __name__ == "__main__":
    # shuffle_sample()
    # judge_cushion_quality()

# 第一步 线上人机对话数据处理
    # 0616 从线上服务日志dianhua_flask_20260610.log 中提取出对话片段
    # extract_dialog_from_logs()

# 第二步 处理 生成的售前场景数据
    # 筛选符合特定场景的对话
    # filter_dialogs_by_scene()

    # 对生成的 售前场景数据 按场景随机抽样对话
    # sample_dialogs_by_scene()

# 第三步 合并人人、人机、生成的数据
    merge_renren_renji_dialogs()

# 第四步 再调用 query_classify.py 中的 main() 函数，给每条数据添加上意图分类 
    # query_classify.main()
    