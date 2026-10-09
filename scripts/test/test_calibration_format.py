#!/usr/bin/env python3
"""测试校准数据格式是否正确"""

import json

def build_calibration_text(sample):
    """构建符合量化要求的完整校准文本"""
    messages = sample['messages']

    # 构建完整的输入文本
    parts = []

    for msg in messages:
        if msg['role'] == 'system':
            parts.append(msg['content'])
        elif msg['role'] == 'user':
            # 添加用户输入
            content = msg['content'].strip()
            if not content.startswith('客户：'):
                content = f"客户：{content}"
            parts.append(content)
        elif msg['role'] == 'assistant':
            # 检查是否是最后一个消息（待生成位置）
            is_last = msg == messages[-1]
            if is_last:
                # 最后一个assistant消息，添加待生成标记
                parts.append("客服：")
            # 其他的assistant消息（如果有）不需要包含在校准数据中

    return "\n".join(parts)

def test_calibration_format():
    """测试校准数据格式"""
    # 读取测试数据
    with open('../data/0420_dianhua_test_simple_prompt.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"测试数据总数：{len(data)}")

    # 测试前3个样本
    for i, sample in enumerate(data[:3]):
        print(f"\n=== 样本 {i+1} ===")

        # 构建校准文本
        calib_text = build_calibration_text(sample)

        # 检查格式
        has_system = '垫话定义' in calib_text or '垫话要求' in calib_text
        ends_with_assistant = calib_text.rstrip().endswith('客服：')

        print(f"长度：{len(calib_text)} 字符")
        print(f"包含系统提示词：{has_system}")
        print(f"以'客服：'结尾：{ends_with_assistant}")

        if has_system and ends_with_assistant:
            print("✓ 格式正确")
        else:
            print("✗ 格式错误")

        # 显示部分内容
        print("\n校准文本预览：")
        print(calib_text[:200] + "..." if len(calib_text) > 200 else calib_text)

if __name__ == "__main__":
    test_calibration_format()