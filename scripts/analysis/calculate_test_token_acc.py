import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import json
import os
import re
import argparse

class TestTokenAccuracyCalculator:
    """计算测试集上的token级别准确率，与验证集eval_token_acc保持一致逻辑

    关键理解：
    1. eval_token_acc是在模型训练时计算的，使用teacher forcing模式
    2. 计算的是模型预测下一个token的准确率
    3. 只计算目标序列部分的准确率，不包括输入序列
    4. 通常使用交叉熵损失函数，计算每个位置的预测概率
    """

    def __init__(self, model_path: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        """
        初始化计算器

        Args:
            model_path: 模型路径（可以是合并后的模型或原始模型）
            device: 运行设备
        """
        self.device = device
        print(f"正在加载模型: {model_path}")

        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

        # 设置pad token（如果没有的话）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"模型加载完成，设备: {self.device}")

    def tokenize_for_model(self, text: str, max_length: int = 512, add_special_tokens: bool = True) -> List[int]:
        """
        使用模型的tokenizer进行分词，与训练时保持一致

        Args:
            text: 输入文本
            max_length: 最大长度
            add_special_tokens: 是否添加特殊token（如<bos>, <eos>）
        """
        # 这里应该与模型训练时的tokenization逻辑保持一致
        # 通常包括添加特殊token、截断、padding等
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
            padding=False,  # 先不padding，后续根据需要处理
            return_tensors=None
        )
        return encoded['input_ids']

    def calculate_token_accuracy_single(self, input_ids: List[int], target_ids: List[int]) -> Dict[str, float]:
        """
        处理单个样本的token准确率计算，使用更合理的逻辑
        """
        try:
            # 确保两个序列长度一致
            max_length = max(len(input_ids), len(target_ids))

            # 对较短的序列进行padding
            if len(input_ids) < max_length:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_length - len(input_ids))
            if len(target_ids) < max_length:
                target_ids = target_ids + [self.tokenizer.pad_token_id] * (max_length - len(target_ids))

            # 转换为tensor
            input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            target_ids_tensor = torch.tensor(target_ids).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 前向传播获取logits
                outputs = self.model(input_ids_tensor)
                logits = outputs.logits

                # 获取预测结果
                predictions = torch.argmax(logits, dim=-1)

                # 计算准确率
                correct_predictions = (predictions == target_ids_tensor).float()

                # 创建mask，忽略padding token
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                mask = (target_ids_tensor != pad_token_id).float()

                num_valid_tokens = mask.sum().item()

                if num_valid_tokens > 0:
                    token_accuracy = (correct_predictions * mask).sum().item() / num_valid_tokens
                else:
                    token_accuracy = 0.0

                return {
                    'token_accuracy': token_accuracy,
                    'num_valid_tokens': num_valid_tokens,
                    'sequence_length': max_length
                }

        except Exception as e:
            print(f"计算单样本准确率出错: {e}")
            return {
                'token_accuracy': 0.0,
                'num_valid_tokens': 0,
                'sequence_length': 0
            }

    def calculate_token_accuracy_with_teacher_forcing(self, input_text: str, target_text: str) -> Dict[str, float]:
        """
        使用teacher forcing模式计算token准确率（与训练时的eval_token_acc完全一致）

        关键逻辑：
        1. 将输入文本和目标文本拼接，格式为：input + target + eos_token
        2. 创建输入序列（去掉最后一个token）和目标序列（去掉第一个token）
        3. 模型预测每个位置的下一个token
        4. 只计算目标文本部分的预测准确率
        """
        try:
            # Tokenization
            input_ids = self.tokenize_for_model(input_text, add_special_tokens=True)
            target_ids = self.tokenize_for_model(target_text, add_special_tokens=True)

            # 移除target中的特殊token，避免重复
            if target_ids and target_ids[0] == self.tokenizer.bos_token_id:
                target_ids = target_ids[1:]
            if target_ids and target_ids[-1] == self.tokenizer.eos_token_id:
                target_ids = target_ids[:-1]

            # 构建完整的序列：input + target + eos
            combined_ids = input_ids + target_ids + [self.tokenizer.eos_token_id]

            # 创建输入和标签（shifted by 1）
            input_seq = combined_ids[:-1]  # 输入：除了最后一个token
            label_seq = combined_ids[1:]   # 标签：除了第一个token

            # 转换为tensor
            input_tensor = torch.tensor([input_seq]).to(self.device)
            label_tensor = torch.tensor([label_seq]).to(self.device)

            with torch.no_grad():
                # 前向传播获取logits
                outputs = self.model(input_tensor)
                logits = outputs.logits  # shape: [1, seq_len, vocab_size]

                # 获取预测结果
                predictions = torch.argmax(logits, dim=-1)

                # 计算每个位置的预测是否正确
                correct = (predictions == label_tensor).float()

                # 只计算target部分的准确率
                target_start_idx = len(input_ids)  # target开始的索引
                target_end_idx = len(input_ids) + len(target_ids)

                # 提取target部分的预测结果
                target_correct = correct[:, target_start_idx:target_end_idx]

                # 计算target部分的准确率
                target_accuracy = target_correct.mean().item()

                # 计算每个token的概率（可选，用于调试）
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, label_tensor.unsqueeze(-1)).squeeze(-1)
                target_log_probs = token_log_probs[:, target_start_idx:target_end_idx]
                avg_log_prob = target_log_probs.mean().item()

                # 计算困惑度
                target_perplexity = np.exp(-avg_log_prob)

                return {
                    'token_accuracy': target_accuracy,
                    'avg_log_prob': avg_log_prob,
                    'target_perplexity': target_perplexity,
                    'num_target_tokens': len(target_ids),
                    'target_start_idx': target_start_idx,
                    'target_end_idx': target_end_idx,
                    'total_seq_length': len(combined_ids),
                    'method': 'teacher_forcing',
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                    'predictions': predictions[:, target_start_idx:target_end_idx].squeeze().cpu().tolist(),
                    'labels': label_tensor[:, target_start_idx:target_end_idx].squeeze().cpu().tolist()
                }

        except Exception as e:
            print(f"Teacher forcing模式计算出错: {e}")
            import traceback
            traceback.print_exc()
            # 降级到简单模式
            return self.calculate_token_accuracy_simple(input_text, target_text)

    def analyze_prediction_errors(self, result: Dict) -> Dict:
        """
        分析预测错误，提供详细的错误分析
        """
        if 'predictions' not in result or 'labels' not in result:
            return {}

        predictions = result['predictions']
        labels = result['labels']

        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(labels, list):
            labels = [labels]

        # 确保长度一致
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]

        # 分析每个位置的错误
        error_positions = []
        error_types = []

        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if pred != label:
                error_positions.append(i)
                pred_token = self.tokenizer.convert_ids_to_tokens([pred])[0] if pred < len(self.tokenizer) else f"[UNK:{pred}]"
                label_token = self.tokenizer.convert_ids_to_tokens([label])[0] if label < len(self.tokenizer) else f"[UNK:{label}]"

                error_types.append({
                    'position': i,
                    'predicted': pred_token,
                    'actual': label_token,
                    'predicted_id': pred,
                    'actual_id': label
                })

        # 统计错误类型
        error_stats = {}
        for error in error_types:
            pred_token = error['predicted']
            actual_token = error['actual']
            error_key = f"{actual_token} -> {pred_token}"
            error_stats[error_key] = error_stats.get(error_key, 0) + 1

        return {
            'total_errors': len(error_positions),
            'error_rate': len(error_positions) / len(predictions) if predictions else 0,
            'error_types': error_types[:10],  # 只保留前10个错误
            'error_statistics': dict(sorted(error_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            'error_positions': error_positions[:10]
        }

    def calculate_token_accuracy_simple(self, input_text: str, target_text: str) -> Dict[str, float]:
        """
        简化的token准确率计算（降级方案）
        """
        try:
            # Tokenization
            input_ids = self.tokenize_for_model(input_text)
            target_ids = self.tokenize_for_model(target_text)

            # 确保两个序列长度一致
            max_length = max(len(input_ids), len(target_ids))

            # 对较短的序列进行padding
            if len(input_ids) < max_length:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_length - len(input_ids))
            if len(target_ids) < max_length:
                target_ids = target_ids + [self.tokenizer.pad_token_id] * (max_length - len(target_ids))

            # 转换为tensor
            input_ids_tensor = torch.tensor([input_ids]).to(self.device)
            target_ids_tensor = torch.tensor([target_ids]).to(self.device)

            with torch.no_grad():
                # 前向传播获取logits
                outputs = self.model(input_ids_tensor)
                logits = outputs.logits

                # 获取预测结果
                predictions = torch.argmax(logits, dim=-1)

                # 计算准确率
                correct_predictions = (predictions == target_ids_tensor).float()

                # 创建mask，忽略padding token
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                mask = (target_ids_tensor != pad_token_id).float()

                num_valid_tokens = mask.sum().item()

                if num_valid_tokens > 0:
                    token_accuracy = (correct_predictions * mask).sum().item() / num_valid_tokens
                else:
                    token_accuracy = 0.0

                return {
                    'token_accuracy': token_accuracy,
                    'num_valid_tokens': num_valid_tokens,
                    'sequence_length': max_length,
                    'method': 'simple'
                }

        except Exception as e:
            print(f"简单模式计算出错: {e}")
            return {
                'token_accuracy': 0.0,
                'num_valid_tokens': 0,
                'sequence_length': 0,
                'method': 'simple'
            }

    def compare_with_training_metrics(self, results: Dict, training_eval_acc: float = 0.92):
        """
        与训练时的验证集指标进行对比分析
        """
        test_acc = results['summary']['avg_token_accuracy']

        analysis = {
            'training_eval_acc': training_eval_acc,
            'test_token_acc': test_acc,
            'difference': test_acc - training_eval_acc,
            'relative_change': (test_acc - training_eval_acc) / training_eval_acc * 100 if training_eval_acc > 0 else 0,
            'interpretation': ''
        }

        # 根据差异大小给出解释
        diff = abs(test_acc - training_eval_acc)
        if diff < 0.05:
            analysis['interpretation'] = "测试集表现与验证集非常接近，模型泛化能力良好"
        elif diff < 0.1:
            analysis['interpretation'] = "测试集表现与验证集略有差异，但在可接受范围内"
        elif diff < 0.2:
            analysis['interpretation'] = "测试集表现与验证集有明显差异，可能存在过拟合或数据分布差异"
        else:
            analysis['interpretation'] = "测试集表现与验证集差异很大，需要深入分析数据分布和模型性能"

        # 添加具体建议
        if test_acc < training_eval_acc:
            analysis['suggestion'] = "测试集准确率低于验证集，建议：\n" \
                                   "1. 检查测试集数据质量\n" \
                                   "2. 分析是否存在数据分布偏移\n" \
                                   "3. 考虑增加正则化或数据增强"
        else:
            analysis['suggestion'] = "测试集准确率高于验证集，这是正常现象，说明模型泛化能力良好"

        return analysis

    def process_excel_file(self, excel_path: str, input_col: str = 'C', target_col: str = 'D',
                          max_samples: int = None, method: str = 'teacher_forcing') -> Dict:
        """
        处理Excel文件，计算token准确率

        Args:
            excel_path: Excel文件路径
            input_col: 输入文本列（默认为A列）
            target_col: 目标文本列（默认为B列）
            max_samples: 最大样本数（None表示处理所有样本）

        Returns:
            分析结果字典
        """
        print(f"正在处理Excel文件: {excel_path}")

        # 读取Excel文件
        df = pd.read_excel(excel_path)

        # 获取列索引
        input_col_idx = ord(input_col.upper()) - ord('A')
        target_col_idx = ord(target_col.upper()) - ord('A')

        # 获取数据
        input_data = df.iloc[:, input_col_idx]
        target_data = df.iloc[:, target_col_idx]

        # 限制样本数（如果指定）
        if max_samples is not None:
            input_data = input_data.head(max_samples)
            target_data = target_data.head(max_samples)

        results = []
        all_token_accuracies = []

        print(f"开始计算token准确率，共 {len(input_data)} 个样本...")

        # 逐行处理
        for idx, (input_text, target_text) in enumerate(tqdm(zip(input_data, target_data), total=len(input_data))):
            # 跳过空值
            if pd.isna(input_text) or pd.isna(target_text):
                continue

            # Tokenization
            input_ids = self.tokenize_for_model(str(input_text))
            target_ids = self.tokenize_for_model(str(target_text))

            # 检查序列长度，如果差异过大可能需要特殊处理
            if abs(len(input_ids) - len(target_ids)) > 10:
                print(f"警告：第{idx+1}行输入输出长度差异过大（{len(input_ids)} vs {len(target_ids)}）")

            # 调试信息：前5个样本显示详细信息
            if idx < 5:
                print(f"\n样本 {idx+1} 调试信息:")
                print(f"输入文本: {input_text}")
                print(f"目标文本: {target_text}")
                print(f"输入token数: {len(input_ids)}")
                print(f"目标token数: {len(target_ids)}")
                print(f"输入tokens: {self.tokenizer.convert_ids_to_tokens(input_ids)[:10]}...")
                print(f"目标tokens: {self.tokenizer.convert_ids_to_tokens(target_ids)[:10]}...")

            # 根据选择的方法计算准确率
            if method == 'teacher_forcing':
                accuracy_result = self.calculate_token_accuracy_with_teacher_forcing(str(input_text), str(target_text))
            else:
                accuracy_result = self.calculate_token_accuracy_simple(str(input_text), str(target_text))

            # 分析预测错误（仅在使用teacher forcing时）
            error_analysis = {}
            if method == 'teacher_forcing' and 'predictions' in accuracy_result:
                error_analysis = self.analyze_prediction_errors(accuracy_result)

            # 保存结果
            result = {
                'row_index': idx + 1,
                'input_text': input_text,
                'target_text': target_text,
                'input_tokens': self.tokenizer.convert_ids_to_tokens(accuracy_result.get('input_ids', [])),
                'target_tokens': self.tokenizer.convert_ids_to_tokens(accuracy_result.get('target_ids', [])),
                'token_accuracy': accuracy_result['token_accuracy'],
                'num_valid_tokens': accuracy_result['num_valid_tokens'],
                'method': accuracy_result.get('method', 'unknown')
            }

            # 添加错误分析结果
            if error_analysis:
                result['error_analysis'] = error_analysis
            results.append(result)
            all_token_accuracies.append(accuracy_result['token_accuracy'])

        # 计算总体统计
        summary = {
            'total_samples': len(results),
            'avg_token_accuracy': np.mean(all_token_accuracies),
            'std_token_accuracy': np.std(all_token_accuracies),
            'min_token_accuracy': np.min(all_token_accuracies),
            'max_token_accuracy': np.max(all_token_accuracies),
            'median_token_accuracy': np.median(all_token_accuracies)
        }

        return {
            'summary': summary,
            'detailed_results': results,
            'all_token_accuracies': all_token_accuracies
        }

    def save_results(self, results: Dict, output_path: str):
        """保存分析结果到Excel文件"""
        # 创建汇总表
        summary_data = {
            '指标': [
                '总样本数', '平均Token准确率', 'Token准确率标准差',
                '最小Token准确率', '最大Token准确率', '中位数Token准确率'
            ],
            '数值': [
                results['summary']['total_samples'],
                f"{results['summary']['avg_token_accuracy']:.4f}",
                f"{results['summary']['std_token_accuracy']:.4f}",
                f"{results['summary']['min_token_accuracy']:.4f}",
                f"{results['summary']['max_token_accuracy']:.4f}",
                f"{results['summary']['median_token_accuracy']:.4f}"
            ]
        }

        # 创建详细结果表
        detailed_data = {
            '样本编号': [],
            '输入文本': [],
            '目标文本': [],
            '输入分词': [],
            '目标分词': [],
            'Token准确率': [],
            '有效Token数': [],
            '计算方法': []
        }

        for result in results['detailed_results']:
            detailed_data['样本编号'].append(result['row_index'])
            detailed_data['输入文本'].append(result['input_text'])
            detailed_data['目标文本'].append(result['target_text'])
            detailed_data['输入分词'].append(' | '.join(result['input_tokens']))
            detailed_data['目标分词'].append(' | '.join(result['target_tokens']))
            detailed_data['Token准确率'].append(f"{result['token_accuracy']:.4f}")
            detailed_data['有效Token数'].append(result['num_valid_tokens'])
            detailed_data['计算方法'].append(result.get('method', 'unknown'))

        # 写入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 汇总表
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)

            # 详细结果表
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细结果', index=False)

            # 添加错误分析表（如果有）
            error_data = []
            for result in results['detailed_results']:
                if 'error_analysis' in result and result['error_analysis']:
                    error_info = result['error_analysis']
                    error_data.append({
                        '样本编号': result['row_index'],
                        'Token准确率': result['token_accuracy'],
                        '错误数': error_info.get('total_errors', 0),
                        '错误率': error_info.get('error_rate', 0),
                        '主要错误类型': list(error_info.get('error_statistics', {}).keys())[:3] if error_info.get('error_statistics') else []
                    })

            if error_data:
                error_df = pd.DataFrame(error_data)
                error_df.to_excel(writer, sheet_name='错误分析', index=False)

        print(f"分析结果已保存至: {output_path}")
        """保存分析结果到Excel文件"""
        # 创建汇总表
        summary_data = {
            '指标': [
                '总样本数', '平均Token准确率', 'Token准确率标准差',
                '最小Token准确率', '最大Token准确率', '中位数Token准确率'
            ],
            '数值': [
                results['summary']['total_samples'],
                f"{results['summary']['avg_token_accuracy']:.4f}",
                f"{results['summary']['std_token_accuracy']:.4f}",
                f"{results['summary']['min_token_accuracy']:.4f}",
                f"{results['summary']['max_token_accuracy']:.4f}",
                f"{results['summary']['median_token_accuracy']:.4f}"
            ]
        }

        # 创建详细结果表
        detailed_data = {
            '样本编号': [],
            '输入文本': [],
            '目标文本': [],
            '输入分词': [],
            '目标分词': [],
            'Token准确率': [],
            '有效Token数': []
        }

        for result in results['detailed_results']:
            detailed_data['样本编号'].append(result['row_index'])
            detailed_data['输入文本'].append(result['input_text'])
            detailed_data['目标文本'].append(result['target_text'])
            detailed_data['输入分词'].append(' | '.join(result['input_tokens']))
            detailed_data['目标分词'].append(' | '.join(result['target_tokens']))
            detailed_data['Token准确率'].append(f"{result['token_accuracy']:.4f}")
            detailed_data['有效Token数'].append(result['num_valid_tokens'])

        # 写入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 汇总表
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)

            # 详细结果表
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细结果', index=False)

        print(f"分析结果已保存至: {output_path}")


def main(gpu_id: int = 0, method: str = 'teacher_forcing', training_eval_acc: float = 0.92):
    """
    主函数

    Args:
        gpu_id: 要使用的GPU编号（0, 1, 2, 3...）
    """
    """主函数"""

    # 设置使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"使用GPU: {gpu_id}")

    # 模型路径 - 使用命令行参数或默认值
    model_path = args.model or "/data/xujianjun/workspace/dianhua_sft/output_v5/merged_model_qwen3-4b_0409_ck100"

    # 测试数据路径 - 使用命令行参数或默认值
    test_data_path = args.data or "/Users/xujianjun/workspace/dianhua-sft/data/0420_dianhua_test_simple_prompt_output.xlsx"

    # 指定列 - 使用命令行参数
    input_col = args.input_col  # 输入文本列
    target_col = args.target_col  # 目标文本列

    # 输出报告路径
    output_report = "/Users/xujianjun/workspace/dianhua-sft/data/test_token_accuracy_report.xlsx"

    print("开始计算测试集的Token准确率...")
    print("="*60)
    print(f"使用GPU: {gpu_id}")
    print(f"计算方法: {method}")
    print(f"验证集eval_token_acc: {training_eval_acc}")
    print(f"模型路径: {model_path}")
    print(f"测试数据: {test_data_path}")
    print(f"输入列: {input_col}, 目标列: {target_col}")
    print("="*60)

    try:
        # 创建计算器
        calculator = TestTokenAccuracyCalculator(model_path)

        # 处理测试数据
        results = calculator.process_excel_file(
            test_data_path,
            input_col=input_col,
            target_col=target_col,
            max_samples=None,  # 处理所有样本，可以设置为较小的数字进行测试
            method=method  # 传入计算方法
        )

        # 打印结果
        print("\n=== 测试集Token准确率分析结果 ===")
        print(f"总样本数: {results['summary']['total_samples']}")
        print(f"平均Token准确率: {results['summary']['avg_token_accuracy']:.4f}")
        print(f"Token准确率标准差: {results['summary']['std_token_accuracy']:.4f}")
        print(f"最小Token准确率: {results['summary']['min_token_accuracy']:.4f}")
        print(f"最大Token准确率: {results['summary']['max_token_accuracy']:.4f}")
        print(f"中位数Token准确率: {results['summary']['median_token_accuracy']:.4f}")

        # 与验证集对比
        print(f"\n=== 与验证集对比 ===")
        comparison = calculator.compare_with_training_metrics(results, training_eval_acc)
        print(f"验证集eval_token_acc: {comparison['training_eval_acc']}")
        print(f"测试集test_token_acc: {comparison['test_token_acc']:.4f}")
        print(f"差异: {comparison['difference']:.4f}")
        print(f"相对变化: {comparison['relative_change']:.2f}%")
        print(f"分析: {comparison['interpretation']}")
        print(f"建议: {comparison['suggestion']}")

        # 保存详细报告
        calculator.save_results(results, output_report)

        # 显示一些低准确率的例子
        print(f"\n=== 低准确率示例 ===")
        low_acc_samples = [r for r in results['detailed_results'] if r['token_accuracy'] < 0.8]
        for i, sample in enumerate(low_acc_samples[:5]):
            print(f"\n低准确率样本 {i+1} (准确率={sample['token_accuracy']:.4f}):")
            print(f"输入: {sample['input_text']}")
            print(f"目标: {sample['target_text']}")
            print(f"输入分词: {' | '.join(sample['input_tokens'][:20])}")  # 只显示前20个token
            print(f"目标分词: {' | '.join(sample['target_tokens'][:20])}")

            # 显示错误分析（如果有）
            if 'error_analysis' in sample and sample['error_analysis']:
                error_info = sample['error_analysis']
                print(f"错误分析: 共{error_info['total_errors']}个错误")
                if error_info['error_statistics']:
                    print("主要错误类型:")
                    for error_type, count in list(error_info['error_statistics'].items())[:3]:
                        print(f"  {error_type}: {count}次")

            # 显示错误分析（如果有）
            if 'error_analysis' in sample and sample['error_analysis']:
                error_info = sample['error_analysis']
                print(f"错误分析: 共{error_info['total_errors']}个错误")
                if error_info['error_statistics']:
                    print("主要错误类型:")
                    for error_type, count in list(error_info['error_statistics'].items())[:3]:
                        print(f"  {error_type}: {count}次")

    except Exception as e:
        print(f"计算过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    # 添加命令行参数
    parser = argparse.ArgumentParser(description='计算测试集Token准确率')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号（默认0）')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--data', type=str, default=None, help='Excel数据路径')
    parser.add_argument('--input-col', type=str, default='C', help='输入文本列（默认C）')
    parser.add_argument('--target-col', type=str, default='D', help='目标文本列（默认D）')
    parser.add_argument('--method', type=str, default='teacher_forcing', choices=['simple', 'teacher_forcing'],
                       help='计算方法（默认teacher_forcing）')
    parser.add_argument('--eval-acc', type=float, default=0.92, help='训练时验证集的eval_token_acc（默认0.92）')

    args = parser.parse_args()

    # 使用命令行参数或默认值
    main(gpu_id=args.gpu, method=args.method, training_eval_acc=args.eval_acc)