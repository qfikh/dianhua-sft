import pandas as pd
import jieba
import re
from typing import List, Tuple, Dict
import numpy as np


class DianhuaDataAnalyzer:
    """电话数据分词准确率分析器"""

    def __init__(self):
        """初始化分析器"""
        # 加载自定义词典（如果有的话）
        # jieba.load_userdict("custom_dict.txt")
        pass

    def tokenize_with_punctuation(self, text: str) -> List[str]:
        """
        使用jieba分词，保留句尾标点符号

        Args:
            text: 输入文本

        Returns:
            分词结果列表，保留标点符号
        """
        if pd.isna(text) or text == '':
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

    def calculate_token_accuracy(self, original_tokens: List[str], generated_tokens: List[str]) -> Dict[str, float]:
        """
        计算两个分词结果的准确率，包含位置对齐的评估

        Args:
            original_tokens: 原始文本分词结果
            generated_tokens: 生成文本分词结果

        Returns:
            包含各种准确率指标的字典
        """
        if not original_tokens and not generated_tokens:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'exact_match': 1.0,
                'token_match_rate': 1.0,
                'position_accuracy': 1.0,
                'position_accuracy_with_penalty': 1.0
            }

        if not original_tokens or not generated_tokens:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'exact_match': 0.0,
                'token_match_rate': 0.0,
                'position_accuracy': 0.0,
                'position_accuracy_with_penalty': 0.0
            }

        # 1. 基于集合的词袋模型指标（不考虑顺序）
        original_set = set(original_tokens)
        generated_set = set(generated_tokens)
        intersection = original_set.intersection(generated_set)

        precision = len(intersection) / len(generated_set) if generated_set else 0.0
        recall = len(intersection) / len(original_set) if original_set else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 2. 完全匹配率
        exact_match = 1.0 if original_tokens == generated_tokens else 0.0

        # 3. 序列匹配率（使用最长公共子序列）
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

        # 4. 位置准确率（按位置对齐比较）
        position_matches = 0
        min_length = min(len(original_tokens), len(generated_tokens))

        # 逐位置比较token
        for i in range(min_length):
            if original_tokens[i] == generated_tokens[i]:
                position_matches += 1

        # 基础位置准确率
        position_accuracy = position_matches / len(original_tokens) if len(original_tokens) > 0 else 0.0

        # 带长度惩罚的位置准确率（鼓励生成合适长度的文本）
        length_ratio = len(generated_tokens) / len(original_tokens) if len(original_tokens) > 0 else 0.0
        if 0.5 <= length_ratio <= 2.0:  # 在合理长度范围内
            length_penalty = 1.0
        else:
            length_penalty = max(0.1, 1.0 - abs(1.0 - length_ratio))  # 长度偏离越大，惩罚越大

        position_accuracy_with_penalty = position_accuracy * length_penalty

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'exact_match': exact_match,
            'token_match_rate': token_match_rate,
            'position_accuracy': position_accuracy,
            'position_accuracy_with_penalty': position_accuracy_with_penalty,
            'common_tokens': len(intersection),
            'original_token_count': len(original_tokens),
            'generated_token_count': len(generated_tokens),
            'position_matches': position_matches
        }

    def analyze_excel_file(self, file_path: str, original_col: str = 'C', generated_col: str = 'D') -> Dict:
        """
        分析Excel文件中两列数据的分词准确率

        Args:
            file_path: Excel文件路径
            original_col: 原始文本列（默认为C列）
            generated_col: 生成文本列（默认为D列）

        Returns:
            分析结果字典
        """
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)

            # 获取列索引（A=0, B=1, C=2, D=3...）
            original_col_idx = ord(original_col.upper()) - ord('A')
            generated_col_idx = ord(generated_col.upper()) - ord('A')

            # 获取列数据
            original_data = df.iloc[:, original_col_idx]
            generated_data = df.iloc[:, generated_col_idx]

            results = []
            total_accuracy = {
                'precision': [],
                'recall': [],
                'f1_score': [],
                'exact_match': [],
                'token_match_rate': [],
                'position_accuracy': [],
                'position_accuracy_with_penalty': []
            }

            # 逐行分析
            for idx, (original, generated) in enumerate(zip(original_data, generated_data)):
                # 分词
                original_tokens = self.tokenize_with_punctuation(original)
                generated_tokens = self.tokenize_with_punctuation(generated)

                # 计算准确率
                accuracy = self.calculate_token_accuracy(original_tokens, generated_tokens)

                # 保存结果
                row_result = {
                    'row_index': idx + 1,
                    'original_text': original,
                    'generated_text': generated,
                    'original_tokens': original_tokens,
                    'generated_tokens': generated_tokens,
                    'accuracy_metrics': accuracy
                }
                results.append(row_result)

                # 累加准确率指标
                for key in total_accuracy:
                    total_accuracy[key].append(accuracy[key])

            # 计算总体统计
            summary = {
                'total_rows': len(results),
                'avg_precision': np.mean(total_accuracy['precision']),
                'avg_recall': np.mean(total_accuracy['recall']),
                'avg_f1_score': np.mean(total_accuracy['f1_score']),
                'exact_match_rate': np.mean(total_accuracy['exact_match']),
                'avg_token_match_rate': np.mean(total_accuracy['token_match_rate']),
                'avg_position_accuracy': np.mean(total_accuracy['position_accuracy']),
                'avg_position_accuracy_with_penalty': np.mean(total_accuracy['position_accuracy_with_penalty']),
                'precision_std': np.std(total_accuracy['precision']),
                'recall_std': np.std(total_accuracy['recall']),
                'f1_score_std': np.std(total_accuracy['f1_score']),
                'position_accuracy_std': np.std(total_accuracy['position_accuracy'])
            }

            return {
                'summary': summary,
                'detailed_results': results
            }

        except Exception as e:
            raise Exception(f"分析Excel文件时出错: {str(e)}")

    def save_analysis_report(self, analysis_result: Dict, output_path: str):
        """
        保存分析报告到Excel文件

        Args:
            analysis_result: 分析结果
            output_path: 输出文件路径
        """
        # 创建汇总表
        summary_data = {
            '指标': ['总行数', '平均精确率', '平均召回率', '平均F1分数',
                    '完全匹配率', '平均Token匹配率', '平均位置准确率',
                    '平均位置准确率(带惩罚)', '精确率标准差', '召回率标准差',
                    'F1分数标准差', '位置准确率标准差'],
            '数值': [
                analysis_result['summary']['total_rows'],
                f"{analysis_result['summary']['avg_precision']:.4f}",
                f"{analysis_result['summary']['avg_recall']:.4f}",
                f"{analysis_result['summary']['avg_f1_score']:.4f}",
                f"{analysis_result['summary']['exact_match_rate']:.4f}",
                f"{analysis_result['summary']['avg_token_match_rate']:.4f}",
                f"{analysis_result['summary']['avg_position_accuracy']:.4f}",
                f"{analysis_result['summary']['avg_position_accuracy_with_penalty']:.4f}",
                f"{analysis_result['summary']['precision_std']:.4f}",
                f"{analysis_result['summary']['recall_std']:.4f}",
                f"{analysis_result['summary']['f1_score_std']:.4f}",
                f"{analysis_result['summary']['position_accuracy_std']:.4f}"
            ]
        }

        # 创建详细结果表
        detailed_data = {
            '行号': [],
            '原始文本': [],
            '生成文本': [],
            '原始分词': [],
            '生成分词': [],
            '精确率': [],
            '召回率': [],
            'F1分数': [],
            '完全匹配': [],
            'Token匹配率': [],
            '位置准确率': [],
            '位置准确率(带惩罚)': [],
            '公共词数': [],
            '原始词数': [],
            '生成词数': [],
            '位置匹配数': []
        }

        for result in analysis_result['detailed_results']:
            detailed_data['行号'].append(result['row_index'])
            detailed_data['原始文本'].append(result['original_text'])
            detailed_data['生成文本'].append(result['generated_text'])
            detailed_data['原始分词'].append(' | '.join(result['original_tokens']))
            detailed_data['生成分词'].append(' | '.join(result['generated_tokens']))
            detailed_data['精确率'].append(f"{result['accuracy_metrics']['precision']:.4f}")
            detailed_data['召回率'].append(f"{result['accuracy_metrics']['recall']:.4f}")
            detailed_data['F1分数'].append(f"{result['accuracy_metrics']['f1_score']:.4f}")
            detailed_data['完全匹配'].append('是' if result['accuracy_metrics']['exact_match'] == 1.0 else '否')
            detailed_data['Token匹配率'].append(f"{result['accuracy_metrics']['token_match_rate']:.4f}")
            detailed_data['位置准确率'].append(f"{result['accuracy_metrics']['position_accuracy']:.4f}")
            detailed_data['位置准确率(带惩罚)'].append(f"{result['accuracy_metrics']['position_accuracy_with_penalty']:.4f}")
            detailed_data['公共词数'].append(result['accuracy_metrics']['common_tokens'])
            detailed_data['原始词数'].append(result['accuracy_metrics']['original_token_count'])
            detailed_data['生成词数'].append(result['accuracy_metrics']['generated_token_count'])
            detailed_data['位置匹配数'].append(result['accuracy_metrics'].get('position_matches', 0))

        # 创建Excel写入对象
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 写入汇总表
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)

            # 写入详细结果表
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细结果', index=False)

        print(f"分析报告已保存至: {output_path}")


def main():
    """主函数"""
    # 创建分析器实例
    analyzer = DianhuaDataAnalyzer()

    # Excel文件路径
    excel_file = '/Users/xujianjun/workspace/dianhua-sft/data/0420_dianhua_test_simple_prompt_output.xlsx'

    # 输出报告路径
    output_report = '/Users/xujianjun/workspace/dianhua-sft/data/token_accuracy_report.xlsx'

    print("开始分析Excel文件的分词准确率...")

    # 执行分析
    try:
        result = analyzer.analyze_excel_file(excel_file)

        # 打印汇总结果
        print("\n=== 分词准确率分析汇总 ===")
        print(f"总行数: {result['summary']['total_rows']}")
        print(f"平均精确率: {result['summary']['avg_precision']:.4f}")
        print(f"平均召回率: {result['summary']['avg_recall']:.4f}")
        print(f"平均F1分数: {result['summary']['avg_f1_score']:.4f}")
        print(f"完全匹配率: {result['summary']['exact_match_rate']:.4f}")
        print(f"平均Token匹配率: {result['summary']['avg_token_match_rate']:.4f}")
        print(f"平均位置准确率: {result['summary']['avg_position_accuracy']:.4f}")
        print(f"平均位置准确率(带惩罚): {result['summary']['avg_position_accuracy_with_penalty']:.4f}")
        print(f"精确率标准差: {result['summary']['precision_std']:.4f}")
        print(f"召回率标准差: {result['summary']['recall_std']:.4f}")
        print(f"F1分数标准差: {result['summary']['f1_score_std']:.4f}")
        print(f"位置准确率标准差: {result['summary']['position_accuracy_std']:.4f}")

        # 保存详细报告
        analyzer.save_analysis_report(result, output_report)

        # 显示前5行的详细分析结果
        print("\n=== 前5行详细分析结果 ===")
        for i in range(min(5, len(result['detailed_results']))):
            row = result['detailed_results'][i]
            print(f"\n行 {row['row_index']}:")
            print(f"原始文本: {row['original_text']}")
            print(f"生成文本: {row['generated_text']}")
            print(f"原始分词: {' | '.join(row['original_tokens'])}")
            print(f"生成分词: {' | '.join(row['generated_tokens'])}")
            print(f"精确率: {row['accuracy_metrics']['precision']:.4f}")
            print(f"召回率: {row['accuracy_metrics']['recall']:.4f}")
            print(f"F1分数: {row['accuracy_metrics']['f1_score']:.4f}")
            print(f"位置准确率: {row['accuracy_metrics']['position_accuracy']:.4f}")
            print(f"位置准确率(带惩罚): {row['accuracy_metrics']['position_accuracy_with_penalty']:.4f}")
            print("-" * 50)

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()