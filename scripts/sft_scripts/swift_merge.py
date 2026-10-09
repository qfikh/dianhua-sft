# merge_lora_model.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    BASE_MODEL = "/data/xujianjun/models/Qwen/Qwen3-4B-Instruct-2507"
    #BASE_MODEL = "/input_model/output/Qwen3-4B-Instruct-2507"
    #BASE_MODEL = "/data/xujianjun/models/Qwen/Qwen3-4B"
    #BASE_MODEL = "/data/renpengfei/common_base_model/qwen/qwen3-32b"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output_v5/v1-20260410-224129/checkpoint-220"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v4_0408_1600/v0-20260413-103600/checkpoint-80"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v6_4800_comp_prompt/v0-20260413-171233/checkpoint-220"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v7_4080_comp_prompt/v0-20260414-100512/checkpoint-220"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v10_0420_no_prompt/v0-20260420-193906/checkpoint-764"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v11_0426_simple_prompt/v1-20260512-103632/checkpoint-764"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v13_0509_simple_prompt/v0-20260509-165245/checkpoint-720"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v14_0511_simple_prompt/v0-20260511-114128/checkpoint-764"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v15_zliu_0512_simple_prompt/checkpoint-2000"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v13_0509_simple_prompt/distilled-model/v3-20260511-142109/checkpoint-2289"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v16_0520_simple_prompt/v1-20260520-174448/checkpoint-770"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v16_0520_zliu_simple_prompt/distilled-model/v2-20260521-164410/checkpoint-1900"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v17_0520_simple_prompt/v0-20260521-112257/checkpoint-720"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v18_0526_simple_prompt/v0-20260526-191205/checkpoint-770"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/v0-20260527-113203/checkpoint-770"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/distilled-model/v0-20260520-223259/checkpoint-700"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/distilled-model/v0-20260528-202717/checkpoint-600"
    #LORA_CHECKPOINT = "/input_model/output/v20_0529_simple_prompt/v2-20260601-175737/checkpoint-720"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v21_0604_simple_prompt/v0-20260605-092744/checkpoint-770"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v22_0612_simple_prompt/v0-20260612-002217/checkpoint-770"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v24_0623_simple_prompt/v4-20260623-191243/checkpoint-868"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v25_0623_simple_prompt/v5-20260623-230705/checkpoint-840"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v26_0624_simple_prompt/v2-20260624-151004/checkpoint-918"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v27_0625_simple_prompt/v0-20260625-151254/checkpoint-918"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v29_0804_simple_prompt/v0-20260807-094415/checkpoint-918"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v30_0809_simple_prompt/v0-20260809-214913/checkpoint-2160"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v30_0810_simple_prompt/v0-20260810-105818/checkpoint-1500"
    #LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v30_0811_simple_prompt/v0-20260811-145540/checkpoint-2160"
    LORA_CHECKPOINT = "/data/xujianjun/workspace/dianhua_sft/output/v31_0818_simple_prompt/v0-20260817-111913/checkpoint-2160"
    #OUTPUT_DIR = "./output/merged_model_qwen3-4b_0410"
    #OUTPUT_DIR = "/data/xujianjun/workspace/dianhua_sft/output/v4_0408_1600/merged_0408_1600"
    #OUTPUT_DIR = "/data/xujianjun/workspace/dianhua_sft/output/v6_4800_comp_prompt/merged_0408_comp_prompt"
    #OUTPUT_DIR = "/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/merged_sft_zliu_0527_simple_prompt"
    #OUTPUT_DIR = "/data/xujianjun/workspace/dianhua_sft/output/v17_0520_simple_prompt/merged_sft_0520_simple_prompt"
    #OUTPUT_DIR = "/input_model/output/v21_0604_simple_prompt/merged_sft_0604_simple_prompt"
    #OUTPUT_DIR = "/data/xujianjun/workspace/dianhua_sft/output/v24_0623_simple_prompt/merged_sft_0623_simple_prompt"
    #OUTPUT_DIR = "/data/xujianjun/workspace/dianhua_sft/output/v27_0625_simple_prompt/merged_sft_0625_simple_prompt"
    OUTPUT_DIR = "/data/xujianjun/workspace/dianhua_sft/output/v31_0818_simple_prompt/merged_sft_0818_simple_prompt"
    
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