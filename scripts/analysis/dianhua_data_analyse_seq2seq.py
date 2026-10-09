import pandas as pd
import jieba
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter


class DianhuaSeq2SeqAnalyzer:
    """专门用于序列生成任务的电话数据分析器"""

    def __init__(self):
        """初始化分析器"""
        pass

    def tokenize_with_punctuation(self, text: str) -> List[str]:
        """
        使用jieba分词，保留句尾标点符号
        """
        if pd.isna(text) or text == '':
            return []

        words = list(jieba.cut(str(text).strip()))
        result = []
        for word in words:
            word = word.strip()
            if word:
                result.append(word)
        return result

    def calculate_seq2seq_metrics(self, reference_tokens: List[str], hypothesis_tokens: List[str]) -> Dict[str, float]:
        """
        计算序列生成任务专用的评估指标

        Args:
            reference_tokens: 参考文本（ground truth）的分词结果
            hypothesis_tokens: 生成文本（模型输出）的分词结果

        Returns:
            序列生成评估指标字典
        """
        metrics = {}

        # 1. 词级精确率（按位置对齐）
        position_matches = 0
        min_length = min(len(reference_tokens), len(hypothesis_tokens))
        max_length = max(len(reference_tokens), len(hypothesis_tokens))

        for i in range(min_length):
            if reference_tokens[i] == hypothesis_tokens[i]:
                position_matches += 1

        # 位置精确率（考虑长度惩罚）
        if len(reference_tokens) > 0:
            position_accuracy = position_matches / len(reference_tokens)
            length_penalty = np.exp(1 - max_length / len(reference_tokens))
            position_accuracy_with_penalty = position_accuracy * length_penalty
        else:
            position_accuracy = 0.0
            position_accuracy_with_penalty = 0.0

        # 2. BLEU分数（n-gram精确率）
        def calculate_ngram_precision(ref_ngrams, hyp_ngrams, n):
            """计算n-gram精确率"""
            if not hyp_ngrams:
                return 0.0

            ref_counter = Counter(ref_ngrams)
            hyp_counter = Counter(hyp_ngrams)

            matches = 0
            for ngram, count in hyp_counter.items():
                matches += min(count, ref_counter.get(ngram, 0))

            return matches / len(hyp_ngrams) if hyp_ngrams else 0.0

        # 计算1-4 gram的BLEU分数
        bleu_scores = []
        for n in range(1, min(5, len(reference_tokens) + 1, len(hypothesis_tokens) + 1)):
            ref_ngrams = [tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1)]
            hyp_ngrams = [tuple(hypothesis_tokens[i:i+n]) for i in range(len(hypothesis_tokens) - n + 1)]

            precision = calculate_ngram_precision(ref_ngrams, hyp_ngrams, n)
            bleu_scores.append(precision)

        # 几何平均BLEU分数
        if bleu_scores:
            bleu_score = 1.0
            for score in bleu_scores:
                bleu_score *= score
            bleu_score = bleu_score ** (1/len(bleu_scores))
        else:
            bleu_score = 0.0

        # 3. 编辑距离（Levenshtein距离）
        def levenshtein_distance(s1, s2):
            """计算两个序列之间的编辑距离"""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

            return dp[m][n]

        edit_distance = levenshtein_distance(reference_tokens, hypothesis_tokens)
        max_distance = max(len(reference_tokens), len(hypothesis_tokens))
        normalized_edit_distance = edit_distance / max_distance if max_distance > 0 else 0.0
        token_accuracy = 1 - normalized_edit_distance

        # 4. 重复率检测
        ref_unique_tokens = set(reference_tokens)
        hyp_unique_tokens = set(hypothesis_tokens)

        # 计算重复token的比例
        ref_counter = Counter(reference_tokens)
        hyp_counter = Counter(hypothesis_tokens)

        ref_repeated = sum(1 for count in ref_counter.values() if count > 1)
        hyp_repeated = sum(1 for count in hyp_counter.values() if count > 1)

        # 5. 覆盖率（生成文本包含了多少参考文本的内容）
        coverage = len(ref_unique_tokens.intersection(hyp_unique_tokens)) / len(ref_unique_tokens) if ref_unique_tokens else 0.0

        # 6. 新增率（生成文本新增了多少不在参考中的内容）
        novel_tokens = hyp_unique_tokens - ref_unique_tokens
        novelty = len(novel_tokens) / len(hyp_unique_tokens) if hyp_unique_tokens else 0.0

        # 7. 长度比例
        length_ratio = len(hypothesis_tokens) / len(reference_tokens) if len(reference_tokens) > 0 else 0.0

        metrics.update({
            # 位置级指标
            'position_accuracy': position_accuracy,
            'position_accuracy_with_penalty': position_accuracy_with_penalty,

            # n-gram级指标
            'bleu_score': bleu_score,
            'bleu_1gram': bleu_scores[0] if len(bleu_scores) > 0 else 0.0,
            'bleu_2gram': bleu_scores[1] if len(bleu_scores) > 1 else 0.0,
            'bleu_3gram': bleu_scores[2] if len(bleu_scores) > 2 else 0.0,
            'bleu_4gram': bleu_scores[3] if len(bleu_scores) > 3 else 0.0,

            # 编辑距离指标
            'edit_distance': edit_distance,
            'normalized_edit_distance': normalized_edit_distance,
            'token_accuracy': token_accuracy,

            # 内容质量指标
            'coverage': coverage,
            'novelty': novelty,
            'length_ratio': length_ratio,

            # 重复性指标
            'ref_repeated_tokens': ref_repeated,
            'hyp_repeated_tokens': hyp_repeated,

            # 基础统计
            'reference_length': len(reference_tokens),
            'hypothesis_length': len(hypothesis_tokens),
            'common_tokens': len(ref_unique_tokens.intersection(hyp_unique_tokens))
        })

        return metrics

    def analyze_excel_file(self, file_path: str, reference_col: str = 'C', hypothesis_col: str = 'D') -> Dict:
        """
        分析Excel文件中两列数据的序列生成质量

        Args:
            file_path: Excel文件路径
            reference_col: 参考文本列（默认为C列）
            hypothesis_col: 生成文本列（默认为D列）

        Returns:
            分析结果字典
        """
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)

            # 获取列索引
            reference_col_idx = ord(reference_col.upper()) - ord('A')
            hypothesis_col_idx = ord(hypothesis_col.upper()) - ord('A')

            # 获取列数据
            reference_data = df.iloc[:, reference_col_idx]
            hypothesis_data = df.iloc[:, hypothesis_col_idx]

            results = []
            all_metrics = {}

            # 初始化指标收集器
            metric_names = ['position_accuracy', 'bleu_score', 'token_accuracy',
                          'coverage', 'novelty', 'length_ratio', 'normalized_edit_distance']
            for name in metric_names:
                all_metrics[name] = []

            # 逐行分析
            for idx, (ref_text, hyp_text) in enumerate(zip(reference_data, hypothesis_data)):
                # 分词
                ref_tokens = self.tokenize_with_punctuation(ref_text)
                hyp_tokens = self.tokenize_with_punctuation(hyp_text)

                # 计算序列生成指标
                metrics = self.calculate_seq2seq_metrics(ref_tokens, hyp_tokens)

                # 保存结果
                row_result = {
                    'row_index': idx + 1,
                    'reference_text': ref_text,
                    'hypothesis_text': hyp_text,
                    'reference_tokens': ref_tokens,
                    'hypothesis_tokens': hyp_tokens,
                    'metrics': metrics
                }
                results.append(row_result)

                # 收集指标
                for name in metric_names:
                    all_metrics[name].append(metrics[name])

            # 计算总体统计
            summary = {
                'total_rows': len(results),
                'avg_position_accuracy': np.mean(all_metrics['position_accuracy']),
                'avg_bleu_score': np.mean(all_metrics['bleu_score']),
                'avg_token_accuracy': np.mean(all_metrics['token_accuracy']),
                'avg_coverage': np.mean(all_metrics['coverage']),
                'avg_novelty': np.mean(all_metrics['novelty']),
                'avg_length_ratio': np.mean(all_metrics['length_ratio']),
                'avg_normalized_edit_distance': np.mean(all_metrics['normalized_edit_distance']),
                'std_position_accuracy': np.std(all_metrics['position_accuracy']),
                'std_bleu_score': np.std(all_metrics['bleu_score']),
                'std_token_accuracy': np.std(all_metrics['token_accuracy'])
            }

            return {
                'summary': summary,
                'detailed_results': results,
                'all_metrics': all_metrics
            }

        except Exception as e:
            raise Exception(f"分析Excel文件时出错: {str(e)}")

    def save_analysis_report(self, analysis_result: Dict, output_path: str):
        """
        保存序列生成分析报告到Excel文件

        Args:
            analysis_result: 分析结果
            output_path: 输出文件路径
        """
        # 创建汇总表
        summary_data = {
            '指标': [
                '总行数', '平均位置准确率', '平均BLEU分数', '平均Token准确率',
                '平均覆盖率', '平均新增率', '平均长度比例', '平均归一化编辑距离',
                '位置准确率标准差', 'BLEU分数标准差', 'Token准确率标准差'
            ],
            '数值': [
                analysis_result['summary']['total_rows'],
                f"{analysis_result['summary']['avg_position_accuracy']:.4f}",
                f"{analysis_result['summary']['avg_bleu_score']:.4f}",
                f"{analysis_result['summary']['avg_token_accuracy']:.4f}",
                f"{analysis_result['summary']['avg_coverage']:.4f}",
                f"{analysis_result['summary']['avg_novelty']:.4f}",
                f"{analysis_result['summary']['avg_length_ratio']:.4f}",
                f"{analysis_result['summary']['avg_normalized_edit_distance']:.4f}",
                f"{analysis_result['summary']['std_position_accuracy']:.4f}",
                f"{analysis_result['summary']['std_bleu_score']:.4f}",
                f"{analysis_result['summary']['std_token_accuracy']:.4f}"
            ]
        }

        # 创建详细结果表
        detailed_data = {
            '行号': [],
            '参考文本': [],
            '生成文本': [],
            '参考分词': [],
            '生成分词': [],
            '位置准确率': [],
            'BLEU分数': [],
            'Token准确率': [],
            '覆盖率': [],
            '新增率': [],
            '长度比例': [],
            '编辑距离': [],
            '归一化编辑距离': []
        }

        for result in analysis_result['detailed_results']:
            detailed_data['行号'].append(result['row_index'])
            detailed_data['参考文本'].append(result['reference_text'])
            detailed_data['生成文本'].append(result['hypothesis_text'])
            detailed_data['参考分词'].append(' | '.join(result['reference_tokens']))
            detailed_data['生成分词'].append(' | '.join(result['hypothesis_tokens']))
            detailed_data['位置准确率'].append(f"{result['metrics']['position_accuracy']:.4f}")
            detailed_data['BLEU分数'].append(f"{result['metrics']['bleu_score']:.4f}")
            detailed_data['Token准确率'].append(f"{result['metrics']['token_accuracy']:.4f}")
            detailed_data['覆盖率'].append(f"{result['metrics']['coverage']:.4f}")
            detailed_data['新增率'].append(f"{result['metrics']['novelty']:.4f}")
            detailed_data['长度比例'].append(f"{result['metrics']['length_ratio']:.4f}")
            detailed_data['编辑距离'].append(result['metrics']['edit_distance'])
            detailed_data['归一化编辑距离'].append(f"{result['metrics']['normalized_edit_distance']:.4f}")

        # 创建Excel写入对象
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 写入汇总表
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)

            # 写入详细结果表
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细结果', index=False)

            # 写入各指标分布
            metrics_df = pd.DataFrame(analysis_result['all_metrics'])
            metrics_df.to_excel(writer, sheet_name='指标分布', index=False)

        print(f"序列生成分析报告已保存至: {output_path}")


def main():
    """主函数"""
    # 创建分析器实例
    analyzer = DianhuaSeq2SeqAnalyzer()

    # Excel文件路径
    excel_file = '/Users/xujianjun/workspace/dianhua-sft/data/0420_dianhua_test_simple_prompt_output.xlsx'

    # 输出报告路径
    output_report = '/Users/xujianjun/workspace/dianhua-sft/data/seq2seq_accuracy_report.xlsx'

    print("开始分析Excel文件的序列生成质量...")

    # 执行分析
    try:
        result = analyzer.analyze_excel_file(excel_file)

        # 打印汇总结果
        print("\n=== 序列生成质量分析汇总 ===")
        print(f"总行数: {result['summary']['total_rows']}")
        print(f"平均位置准确率: {result['summary']['avg_position_accuracy']:.4f}")
        print(f"平均BLEU分数: {result['summary']['avg_bleu_score']:.4f}")
        print(f"平均Token准确率: {result['summary']['avg_token_accuracy']:.4f}")
        print(f"平均覆盖率: {result['summary']['avg_coverage']:.4f}")
        print(f"平均新增率: {result['summary']['avg_novelty']:.4f}")
        print(f"平均长度比例: {result['summary']['avg_length_ratio']:.4f}")
        print(f"平均归一化编辑距离: {result['summary']['avg_normalized_edit_distance']:.4f}")

        # 保存详细报告
        analyzer.save_analysis_report(result, output_report)

        # 显示前5行的详细分析结果
        print("\n=== 前5行详细分析结果 ===")
        for i in range(min(5, len(result['detailed_results']))):
            row = result['detailed_results'][i]
            print(f"\n行 {row['row_index']}:")
            print(f"参考文本: {row['reference_text']}")
            print(f"生成文本: {row['hypothesis_text']}")
            print(f"参考分词: {' | '.join(row['reference_tokens'])}")
            print(f"生成分词: {' | '.join(row['hypothesis_tokens'])}")
            print(f"位置准确率: {row['metrics']['position_accuracy']:.4f}")
            print(f"BLEU分数: {row['metrics']['bleu_score']:.4f}")
            print(f"Token准确率: {row['metrics']['token_accuracy']:.4f}")
            print(f"覆盖率: {row['metrics']['coverage']:.4f}")
            print(f"新增率: {row['metrics']['novelty']:.4f}")
            print("-" * 50)

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()