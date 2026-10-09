#!/bin/bash
# AWQ 4位量化 - 使用本地数据集
# 只需将 --calib_dataset 参数改为本地路径

CUDA_VISIBLE_DEVICES=0 \
swift export \
  --model /data/xujianjun/workspace/dianhua_sft/output/v11_0420_simple_prompt/merged_0420_simple_prompt \
  --model_type qwen3 \
  --output_dir ./qwen3-4b-awq-local-quantized \
  --quant_bits 4 \
  --quant_method awq \
  --torch_dtype bfloat16 \
  --dataset "/data/xujianjun/workspace/dianhua_sft/datasets/0421_dianhua_calibration_data.jsonl" \
  --quant_n_samples 256 \
  --max_length 2048