#!/bin/bash

# AWQ 4位量化脚本
# 注意：AWQ量化通常需要校准数据集，这里使用默认数据集

MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v11_0420_simple_prompt/merged_0420_simple_prompt"
OUTPUT_DIR="/data/xujianjun/workspace/dianhua_sft/output/v11_0420_simple_prompt/qwen3-4b-awq-int4-quantized"

# 设置环境变量
#export CUDA_VISIBLE_DEVICES=1

# 执行AWQ 4位量化
echo "开始执行AWQ 4位量化..."
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"

# 使用SWIFT进行AWQ量化
# 注意：这里使用默认的校准数据集
swift export \
  --model $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --quant_bits 4 \
  --quant_method bnb \
  --torch_dtype bfloat16

echo "AWQ量化完成！"
echo "量化模型保存在: $OUTPUT_DIR"
echo ""

# 检查模型文件
echo "检查生成的模型文件:"
ls -la $OUTPUT_DIR/

# 模型信息
echo ""
echo "模型配置信息:"
if [ -f "$OUTPUT_DIR/config.json" ]; then
  echo "模型配置:"
  cat $OUTPUT_DIR/config.json | grep -E "(model_type|vocab_size|hidden_size|num_attention_heads|num_hidden_layers)"
fi

# 文件大小统计
echo ""
echo "模型文件大小统计:"
if [ -f "$OUTPUT_DIR/model.safetensors" ]; then
  ls -lh $OUTPUT_DIR/model.safetensors
elif [ -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
  ls -lh $OUTPUT_DIR/pytorch_model.bin
fi

# 使用示例
echo ""
echo "量化模型使用示例:"
echo "CUDA_VISIBLE_DEVICES=0 python vllm_deploy_awq.py"
