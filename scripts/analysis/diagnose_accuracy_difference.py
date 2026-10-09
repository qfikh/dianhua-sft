import pandas as pd
import jieba
from collections import Counter

def tokenize_with_punctuation(text):
    """分词并保留标点符号"""
    if pd.isna(text) or text == '':
        return []
    words = list(jieba.cut(str(text).strip()))
    result = []
    for word in words:
        word = word.strip()
        if word:
            result.append(word)
    return result

def analyze_position_accuracy_detail(original_tokens, generated_tokens):
    """详细分析位置准确率"""
    print(f"原始分词: {original_tokens}")
    print(f"生成分词: {generated_tokens}")

    min_length = min(len(original_tokens), len(generated_tokens))
    max_length = max(len(original_tokens), len(generated_tokens))

    print(f"\n长度对比:")
    print(f"原始长度: {len(original_tokens)}")
    print(f"生成长度: {len(generated_tokens)}")
    print(f"长度比: {len(generated_tokens)/len(original_tokens) if len(original_tokens) > 0 else 0:.2f}")

    # 逐位置分析
    print(f"\n逐位置对比:")
    position_matches = 0
    for i in range(max_length):
        orig_token = original_tokens[i] if i < len(original_tokens) else "[PAD]"
        gen_token = generated_tokens[i] if i < len(generated_tokens) else "[PAD]"
        match = "✓" if i < min_length and original_tokens[i] == generated_tokens[i] else "✗"
        if i < min_length and original_tokens[i] == generated_tokens[i]:
            position_matches += 1
        print(f"位置 {i:2d}: {orig_token:8} vs {gen_token:8} {match}")

    position_accuracy = position_matches / len(original_tokens) if len(original_tokens) > 0 else 0.0
    print(f"\n位置准确率: {position_matches}/{len(original_tokens)} = {position_accuracy:.4f}")

    # 分析不匹配的原因
    print(f"\n不匹配分析:")

    # 1. 词序问题
    if len(original_tokens) > 1 and len(generated_tokens) > 1:
        print("1. 词序分析:")
        common_tokens = set(original_tokens) & set(generated_tokens)
        print(f"   公共词: {list(common_tokens)}")
        print(f"   原始独有: {list(set(original_tokens) - set(generated_tokens))}")
        print(f"   生成独有: {list(set(generated_tokens) - set(original_tokens))}")

    # 2. 分词差异
    print("\n2. 分词差异分析:")
    print(f"   原始分词方式: {original_tokens}")
    print(f"   可能的其他分词方式:")
    # 展示其他可能的分词结果
    if len(original_tokens) > 0:
        # 合并相邻词展示不同分词可能
        for i in range(len(original_tokens)-1):
            merged = original_tokens[:i] + [original_tokens[i] + original_tokens[i+1]] + original_tokens[i+2:]
            print(f"     {merged}")

    return position_accuracy

def analyze_excel_file_detailed(file_path, sample_size=10):
    """详细分析Excel文件"""
    print(f"正在分析文件: {file_path}")
    print("="*80)

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 获取C列和D列
    original_col_idx = ord('C') - ord('A')
    generated_col_idx = ord('D') - ord('A')

    original_data = df.iloc[:, original_col_idx]
    generated_data = df.iloc[:, generated_col_idx]

    # 统计信息
    total_rows = len(df)
    position_accuracies = []

    print(f"总行数: {total_rows}")
    print(f"采样分析前 {sample_size} 行数据:")
    print("-"*80)

    # 分析前sample_size行
    for i in range(min(sample_size, total_rows)):
        original = original_data.iloc[i]
        generated = generated_data.iloc[i]

        print(f"\n第 {i+1} 行:")
        print(f"原始: {original}")
        print(f"生成: {generated}")

        # 分词
        original_tokens = tokenize_with_punctuation(original)
        generated_tokens = tokenize_with_punctuation(generated)

        # 详细分析
        acc = analyze_position_accuracy_detail(original_tokens, generated_tokens)
        position_accuracies.append(acc)

        print("\n" + "-"*80)

    # 整体统计
    avg_position_accuracy = sum(position_accuracies) / len(position_accuracies)
    print(f"\n采样数据平均位置准确率: {avg_position_accuracy:.4f}")

    # 分析准确率分布
    print(f"\n准确率分布:")
    acc_counter = Counter([round(acc, 2) for acc in position_accuracies])
    for acc, count in sorted(acc_counter.items()):
        print(f"  准确率 {acc:.2f}: {count} 次")

    # 分析低准确率的原因
    low_acc_count = sum(1 for acc in position_accuracies if acc < 0.5)
    print(f"\n低准确率(<0.5)比例: {low_acc_count}/{len(position_accuracies)} = {low_acc_count/len(position_accuracies):.2%}")

    return position_accuracies

def compare_with_model_acc():
    """分析与模型acc差异的原因"""
    print("\n" + "="*80)
    print("与模型eval_token_acc=0.92的差异分析:")
    print("="*80)

    reasons = [
        "1. 评估方式不同:",
        "   - 我们的位置准确率: 严格按位置对齐比较",
        "   - 模型的eval_token_acc: 可能是预测概率或更宽松的匹配",
        "",
        "2. 分词差异:",
        "   - 我们使用jieba分词",
        "   - 模型可能使用不同的tokenizer（如BERT tokenizer）",
        "",
        "3. 数据预处理差异:",
        "   - Excel数据可能包含特殊字符、格式问题",
        "   - 模型训练时可能进行了清洗和标准化",
        "",
        "4. 序列长度处理:",
        "   - 我们按最短序列长度计算",
        "   - 模型可能使用padding/truncation策略",
        "",
        "5. 匹配策略:",
        "   - 我们要求完全匹配（包括标点）",
        "   - 模型可能忽略标点或大小写",
        "",
        "6. 评估阶段差异:",
        "   - 0.92可能是训练时的最佳验证准确率",
        "   - 0.56是当前测试集的实际表现"
    ]

    for reason in reasons:
        print(reason)

# 主函数
if __name__ == "__main__":
    # Excel文件路径
    excel_file = '/Users/xujianjun/workspace/dianhua-sft/data/0420_dianhua_test_simple_prompt_output.xlsx'

    # 详细分析
    accuracies = analyze_excel_file_detailed(excel_file, sample_size=20)

    # 差异分析
    compare_with_model_acc()

    print(f"\n建议:")
    print(f"1. 检查模型使用的tokenizer和预处理流程")
    print(f"2. 对比模型训练时的验证集和当前测试集的分词结果")
    print(f"3. 考虑使用更宽松的匹配策略（如忽略标点）")
    print(f"4. 分析低准确率样本，查看具体错误模式")