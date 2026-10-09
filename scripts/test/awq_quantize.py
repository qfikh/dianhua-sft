# awq_quantize.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

# 1. 加载微调合并后的模型
model_path = "/path/to/your/merged_qwen3-4b"  # 您的模型路径
output_path = "/path/to/quantized/qwen3-4b-awq-w4g128"

# 2. 配置量化参数
quant_config = {
    "w_bit": 4,               # 4-bit量化
    "q_group_size": 128,      # 分组大小（平衡精度与速度）
    "zero_point": True,       # 使用零点偏移
    "version": "GEMM",        # 量化算法类型
    "calib_data": "calibration_data.txt"  # 校准数据文件
}

# 3. 加载模型和tokenizer
print("Loading model...")
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 4. 执行量化
print("Quantizing with AWQ...")
model.quantize(tokenizer, quant_config=quant_config)

# 5. 保存量化模型
print(f"Saving quantized model to {output_path}...")
model.save_quantized(output_path)
tokenizer.save_pretrained(output_path)
print("AWQ quantization completed!")