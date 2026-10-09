import jieba

def tokenize_with_punctuation(text):
    """分词并保留标点符号"""
    if not text or text == '':
        return []

    # 使用jieba进行分词
    words = list(jieba.cut(str(text).strip()))

    # 处理结果，保留标点符号
    result = []
    for word in words:
        word = word.strip()
        if word:  # 去除空字符串
            result.append(word)

    return result

def calculate_accuracy_metrics(original_tokens, generated_tokens):
    """计算准确率指标"""
    print(f"原始分词结果: {original_tokens}")
    print(f"生成分词结果: {generated_tokens}")

    # 转换为集合进行计算
    original_set = set(original_tokens)
    generated_set = set(generated_tokens)

    print(f"原始词集合: {original_set}")
    print(f"生成词集合: {generated_set}")

    # 计算交集
    intersection = original_set.intersection(generated_set)
    print(f"交集: {intersection}")

    # 计算精确率、召回率和F1分数
    precision = len(intersection) / len(generated_set) if generated_set else 0.0
    recall = len(intersection) / len(original_set) if original_set else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n精确率 (Precision): {len(intersection)} / {len(generated_set)} = {precision:.4f}")
    print(f"召回率 (Recall): {len(intersection)} / {len(original_set)} = {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")

    # 计算完全匹配率
    exact_match = 1.0 if original_tokens == generated_tokens else 0.0
    print(f"完全匹配: {'是' if exact_match == 1.0 else '否'}")

    # 使用动态规划计算最长公共子序列
    m, n = len(original_tokens), len(generated_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if original_tokens[i-1] == generated_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]
    total_tokens = max(len(original_tokens), len(generated_tokens))
    token_match_rate = lcs_length / total_tokens if total_tokens > 0 else 0.0

    print(f"最长公共子序列长度: {lcs_length}")
    print(f"总词数: {total_tokens}")
    print(f"Token匹配率: {token_match_rate:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'exact_match': exact_match,
        'token_match_rate': token_match_rate
    }

# 测试例子
original = "嗯，好的。"
generated = "好的，明白。"

print("=== 分析例子 ===")
print(f"原始文本: '{original}'")
print(f"生成文本: '{generated}'")
print()

# 分词
original_tokens = tokenize_with_punctuation(original)
generated_tokens = tokenize_with_punctuation(generated)

print("\n=== 计算准确率指标 ===")
metrics = calculate_accuracy_metrics(original_tokens, generated_tokens)

print("\n=== 最终结果汇总 ===")
print(f"精确率: {metrics['precision']:.4f}")
print(f"召回率: {metrics['recall']:.4f}")
print(f"F1分数: {metrics['f1_score']:.4f}")
print(f"完全匹配率: {metrics['exact_match']:.4f}")
print(f"Token匹配率: {metrics['token_match_rate']:.4f}")