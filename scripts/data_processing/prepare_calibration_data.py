# 准备量化时用到的校准数据
import json
import random

def build_calibration_text(sample):
    """构建符合量化要求的完整校准文本

    格式：与训练时完全一致的输入格式
    即：System Prompt + 对话历史 + 客服：（待生成位置）
    """
    messages = sample['messages']

    # 构建模型输入：包含除最后一个assistant回复外的所有内容
    parts = []

    for i, msg in enumerate(messages):
        if msg['role'] == 'system':
            # 系统提示词（完整保留）
            parts.append(msg['content'])
        elif msg['role'] == 'user':
            # 用户消息（完整保留，不添加前缀）
            parts.append(msg['content'])
        elif msg['role'] == 'assistant':
            # 只保留最后一个assistant消息的前缀（待生成位置）
            if i == len(messages) - 1:
                # 检查是否已有客服前缀
                if msg['content'].startswith('客服：'):
                    parts.append('客服：')

    return '\n'.join(parts)

def prepare_from_finetune_data():
    """从微调数据集中抽取256条样本作为校准数据"""
    # 设置随机种子确保可重复性
    random.seed(42)
    # 读取训练数据
    with open('../data/0420_dianhua_train_simple_prompt.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤：确保每个样本都有完整的对话结构
    valid_data = []
    for sample in data:
        messages = sample['messages']

        # 必须有system prompt
        has_system = any(m['role'] == 'system' for m in messages)
        # 必须有至少一轮对话（user + assistant）
        dialog_turns = len([m for m in messages if m['role'] in ['user', 'assistant']])
        # 最后必须是assistant回复
        ends_with_assistant = messages[-1]['role'] == 'assistant' if messages else False

        if has_system and dialog_turns >= 2 and ends_with_assistant:
            valid_data.append(sample)

    print(f"原始数据：{len(data)} 条，有效数据：{len(valid_data)} 条")

    # 随机采样256条（如果数据不足则取全部）
    sample_size = min(256, len(valid_data))
    calibration_samples = random.sample(valid_data, sample_size)

    # 保存为文本格式（每行一条完整样本）
    # 由于文本可能包含换行符，我们使用 JSONL 格式保存

    with open('../data/0421_dianhua_calibration_data.jsonl', 'w', encoding='utf-8') as f:
        for sample in calibration_samples:
            calibration_text = build_calibration_text(sample)

            # 确保文本有效
            if calibration_text and len(calibration_text) > 50:
                # 保存为JSONL格式，每行一个JSON对象
                json.dump({"text": calibration_text}, f, ensure_ascii=False)
                f.write('\n')

    # 同时保存纯文本版本（用于不支持JSONL的工具）
    with open('../data/0421_dianhua_calibration_data.txt', 'w', encoding='utf-8') as f:
        for sample in calibration_samples:
            calibration_text = build_calibration_text(sample)

            # 确保文本有效
            if calibration_text and len(calibration_text) > 50:
                # 将文本中的换行符替换为特殊标记
                text_single_line = calibration_text
                f.write(text_single_line + '\n\n')  

    print(f"成功生成校准数据：{sample_size} 条样本")
    print(f"保存路径：../data/0421_dianhua_calibration_data.txt")

    # 验证生成的校准数据（JSONL格式）
    print("\n验证校准数据格式...")
    valid_samples = 0
    total_samples = 0

    with open('../data/0421_dianhua_calibration_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                total_samples += 1
                try:
                    data = json.loads(line)
                    text = data.get('text', '')

                    # 检查是否包含系统提示词
                    has_system = '垫话定义' in text or '垫话要求' in text
                    # 检查是否以"客服："结尾
                    ends_with_assistant = text.rstrip().endswith('客服：')

                    if has_system and ends_with_assistant:
                        valid_samples += 1
                except json.JSONDecodeError:
                    print(f"JSON解析失败：{line[:50]}...")

    print(f"格式验证完成：{valid_samples}/{total_samples} 条样本格式正确")

    # 显示统计信息
    total_chars = sum(len(build_calibration_text(s)) for s in calibration_samples)
    avg_chars = total_chars / len(calibration_samples) if calibration_samples else 0
    print(f"平均字符数：{avg_chars:.1f}")

    # 显示校准数据的tokenizer编码长度分布
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("../output_v5/merged_model_qwen3-4b_0409_ck100")

        token_lengths = []
        for sample in calibration_samples[:10]:  # 检查前10个样本
            text = build_calibration_text(sample)
            tokens = tokenizer.encode(text)
            token_lengths.append(len(tokens))

        if token_lengths:
            avg_tokens = sum(token_lengths) / len(token_lengths)
            max_tokens = max(token_lengths)
            min_tokens = min(token_lengths)
            print(f"\nTokenizer统计（前10个样本）：")
            print(f"平均token数：{avg_tokens:.1f}")
            print(f"最大token数：{max_tokens}")
            print(f"最小token数：{min_tokens}")

    except Exception as e:
        print(f"\nTokenizer统计失败：{e}")

    # 显示一个完整样本示例
    if calibration_samples:
        print("\n完整样本示例（JSONL格式）：")
        sample_text = build_calibration_text(calibration_samples[0])
        print(json.dumps({"text": sample_text}, ensure_ascii=False, indent=2))

        print("\n" + "="*60)
        print("\n样本统计信息：")
        print(f"- 平均字符数：{avg_chars:.1f}")
        print(f"- 样本总数：{sample_size}")
        print(f"- 格式正确率：{valid_samples/total_samples*100:.1f}%")
        print("\n校准数据已保存为：")
        print("- JSONL格式：../data/0421_dianhua_calibration_data.jsonl")
        print("- 单行文本格式：../data/0421_dianhua_calibration_data.txt")

if __name__ == "__main__":
    prepare_from_finetune_data()  # 优先使用场景数据