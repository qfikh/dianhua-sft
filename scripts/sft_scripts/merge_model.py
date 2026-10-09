# merge_lora_model.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def merge_lora_model(base_model_path, lora_checkpoint_path, output_dir):
    """
    将LoRA权重合并到基座模型
    """
    print("=" * 60)
    print(f"基座模型: {base_model_path}")
    print(f"LoRA检查点: {lora_checkpoint_path}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    try:
        # 1. 加载基座模型
        print("\n1. 加载基座模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # 2. 加载tokenizer
        print("2. 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # 3. 加载LoRA适配器
        print("3. 加载LoRA适配器...")
        # 尝试使用PEFT库加载
        model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
        
        # 4. 合并权重
        print("4. 合并LoRA权重到基座模型...")
        merged_model = model.merge_and_unload()
        
        # 5. 保存合并后的模型
        print("5. 保存合并后的模型...")
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
       
        print(f"\n✅ 合并完成！")
        print(f"   模型已保存到: {output_dir}")
        print(f"   总参数量: {sum(p.numel() for p in merged_model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        return False

if __name__ == "__main__":
    # 您的配置
    BASE_MODEL = "/data/xujianjun/models/Qwen/Qwen3-4B"
    LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output_v5/v3-20260409-125436/checkpoint-100"
    OUTPUT_DIR = "./output_v5/merged_model_qwen3-4b_0409_ck100"
    
    # 执行合并
    success = merge_lora_model(BASE_MODEL, LORA_CHECKPOINT, OUTPUT_DIR)
    
    if success:
        # 验证合并结果
        print("\n验证合并结果...")
        try:
            test_model = AutoModelForCausalLM.from_pretrained(
                OUTPUT_DIR,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            test_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
            print(f"✅ 验证成功！合并模型可以正常加载")
            print(f"   模型类型: {type(test_model)}")
        except Exception as e:
            print(f"⚠️ 模型加载验证失败: {e}")