import json
from tqdm import tqdm
from collections import defaultdict
from swift.llm import get_template, get_model_tokenizer

# 加载 template
_, tokenizer = get_model_tokenizer(
    '/data/xujianjun/models/Qwen/Qwen3-4B-Instruct-2507',
    load_model=False
)
template = get_template('qwen3', tokenizer, loss_scale='last_round')
template.set_mode('train')  # 训练模式
# 如果你用 loss_scale=last_round，这里也要设置

MAX_LENGTH = 2048

stats = defaultdict(int)
bad_samples = []

with open('0526_v3_dianhua_test_sft_simple_prompt.json') as f:
    data = json.load(f)

for i, sample in enumerate(tqdm(data)):
    try:
        encoded = template.encode(sample)
        input_ids = encoded['input_ids']
        labels = encoded['labels']

        # 检查 1：长度是否会被截断
        if len(input_ids) > MAX_LENGTH:
            stats['too_long'] += 1
            # 检查 target 是否被截掉
            tail = labels[-min(50, len(labels)):]
            if all(x == -100 for x in tail):
                stats['target_truncated'] += 1
                bad_samples.append((i, 'target_truncated', len(input_ids)))

        # 检查 2：labels 是否有非 -100 位置（即是否有训练信号）
        trainable = [x for x in labels if x != -100]
        if len(trainable) == 0:
            stats['no_trainable_tokens'] += 1
            bad_samples.append((i, 'no_trainable', None))
        elif len(trainable) > 12:
            # 垫话很短，正常应该 < 30 token
            # 如果训练的 token 太多，说明 loss_scale 可能没生效
            stats['too_many_trainable'] += 1
            if stats['too_many_trainable'] <= 3:  # 只记录前 3 个样例
                decoded = tokenizer.decode([x for x in labels if x != -100])
                print(f"样本 {i} 训练内容过长: {decoded[:200]}")

        # 检查 3：第一条 message 是不是 assistant
        if sample['messages'][1]['role'] == 'assistant':
            stats['starts_with_assistant'] += 1

        # 检查 4：最后第二条是不是 user（target 前必须是 user）
        if sample['messages'][-2]['role'] != 'user':
            stats['target_not_after_user'] += 1
            bad_samples.append((i, 'target_not_after_user', None))

        # 记录长度分布
        stats[f'len_bucket_{len(input_ids)//500*500}'] += 1

    except Exception as e:
        stats['encode_error'] += 1
        bad_samples.append((i, f'error: {str(e)[:100]}', None))

# 打印统计
print(f"\n总样本数: {len(data)}")
print(f"异常样本数: {len(bad_samples)}")
print("\n各类异常统计:")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")

print("\n前 10 个异常样本:")
for s in bad_samples[:10]:
    print(f"  {s}")

# 抽 1 条正常样本，把 encode 结果完整打印出来，人工 review
print("\n=== 正常样本 encode 示例 ===")
encoded = template.encode(data[0])
print("完整 prompt:")
print(tokenizer.decode(encoded['input_ids']))
print("\n训练目标 (labels 非 -100 部分):")
trainable_ids = [x for x in encoded['labels'] if x != -100]
print(tokenizer.decode(trainable_ids))