import asyncio
import pandas as pd
from tqdm import tqdm


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


async def classify_scene_async(session, dialog, idx, semaphore):
    """异步调用 API 进行场景分类"""
    async with semaphore:
        url = "http://192.168.86.3/v1/chat/completions"
        headers = {
            "Authorization": "Bearer gpustack_bfcec12a323d24b5_4aa50f5b0f5bb33059ad6d458a3d4a6c",
            "Content-Type": "application/json"
        }

        prompt = SCENE_CLASSIFY_PROMPT.format(dialog=dialog)
        payload = {
            "model": "qwen3-30b-a3b-instruct-2507",
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请判断对话场景。"}
            ]
        }

        try:
            async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    return idx, content
                else:
                    return idx, f"请求失败: {response.status}"
        except Exception as e:
            return idx, f"异常: {str(e)}"


async def classify_dialogs_batch_async(dialogs, max_concurrent=50):
    """高并发批量分类对话场景

    Args:
        dialogs: 对话列表
        max_concurrent: 最大并发数，默认50

    Returns:
        results: 分类结果列表
    """
    import aiohttp

    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(dialogs)

    async with aiohttp.ClientSession() as session:
        tasks = [
            classify_scene_async(session, dialog, idx, semaphore)
            for idx, dialog in enumerate(dialogs)
        ]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="分类进度"):
            idx, result = await future
            results[idx] = result

    return results


def classify_dialogs_concurrent():
    """高并发调用 API 进行对话场景分类

    读取 Excel 文件中的对话数据，使用异步并发调用 API 进行场景分类，
    将结果保存到新的 Excel 文件。
    """
    from collections import Counter

    input_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output.xlsx"
    output_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0618售前数据-交付数据-清洗后4220-output-filtered.xlsx"

    # 读取数据
    df = pd.read_excel(input_file)
    print(f"共读取 {len(df)} 条数据")

    # 提取对话列表
    dialogs = []
    for _, row in df.iterrows():
        dialog = str(row['文本']).strip()
        if not dialog or dialog == 'nan':
            dialogs.append("")
        else:
            # 只取最后10句对话
            lines = dialog.split('\n')
            if len(lines) <= 3:
                dialogs.append("")
            else:
                recent_dialog = '\n'.join(lines[-10:])
                dialogs.append(recent_dialog)

    # 过滤空对话
    valid_indices = [i for i, d in enumerate(dialogs) if d]
    valid_dialogs = [dialogs[i] for i in valid_indices]
    print(f"有效对话: {len(valid_dialogs)} 条")

    # 高并发调用 API
    print(f"开始并发调用 API，最大并发数: 50")
    results = asyncio.run(classify_dialogs_batch_async(valid_dialogs, max_concurrent=50))

    # 构建完整结果列表
    full_results = [''] * len(df)
    for i, result in zip(valid_indices, results):
        if result in ["车型配置", "车型对比", "价格政策"]:
            full_results[i] = result
        else:
            full_results[i] = ''

    # 添加结果列
    df['场景筛选'] = full_results

    # 统计结果
    counter = Counter(full_results)
    print(f"\n分类统计:")
    for item, count in counter.most_common():
        if item:
            print(f"  {item}: {count}")

    # 保存结果
    df.to_excel(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")


if __name__ == "__main__":
    classify_dialogs_concurrent()
