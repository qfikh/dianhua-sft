# start_vllm_service.sh
#!/bin/bash
# 服务启动脚本
#MODEL_PATH="/data/xujianjun/models/Qwen/Qwen3-4B"
#MODEL_PATH="/data/xujianjun/models/Qwen/Qwen3-4B-Instruct-2507"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v4_0408_1600/merged_0408_1600"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output_v5/merged_model_qwen3-4b_0410"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v6_4800_comp_prompt/merged_0408_comp_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v7_4080_comp_prompt/merged_0414_comp_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v9_0419_comp_prompt/merged_0419_comp_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v10_0420_no_prompt/merged_0420_no_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v11_0420_simple_prompt/merged_0420_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v11_0420_simple_prompt/qwen3-4b-sft-gptq-int4"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v11_0426_simple_prompt/merge_sft_0512_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v11_0420_simple_prompt/qwen3-4b-sft-gptq-int4"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v12_0426_comp_prompt/merged_v12_0426_comp_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v12_0426_comp_prompt/qwen3-4b-instruct-sft-gptq-int4"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v13_0509_simple_prompt/merge_sft_zliu_0512_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v14_0511_simple_prompt/merge_sft_0512_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v15_zliu_0512_simple_prompt/merge_sft_0512_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v16_0520_simple_prompt/merged_sft_zliu_0520_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v17_0520_simple_prompt/merged_sft_0520_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v18_0526_simple_prompt/merged_sft_0526_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/merged_sft_zliu_0527_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/merged_sft_0527_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v19_0527_simple_prompt/merged_sft_zliu_0527_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v20_0529_simple_prompt/merged_sft_0529_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v21_0604_simple_prompt/merged_sft_0604_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v22_0612_simple_prompt/merged_sft_0612_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v23_0618_simple_prompt/merged_sft_0618_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v24_0623_simple_prompt/merged_sft_0623_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v25_0623_simple_prompt/merged_sft_0623_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v26_0624_simple_prompt/merged_sft_0624_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v27_0625_simple_prompt/merged_sft_0625_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v29_0807_simple_prompt/merged_sft_0807_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v30_0809_simple_prompt/merged_sft_0809_simple_prompt"
#MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v30_0811_simple_prompt/merged_sft_0811_simple_prompt"
MODEL_PATH="/data/xujianjun/workspace/dianhua_sft/output/v31_0818_simple_prompt/merged_sft_0818_simple_prompt"
PORT=5010
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

echo "启动vLLM推理服务..."
echo "模型路径: ${MODEL_PATH}"
echo "服务端口: ${PORT}"
echo "日志目录: ${LOG_DIR}"

# 设置环境变量
PORT=5010
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

echo "启动vLLM推理服务..."
echo "模型路径: ${MODEL_PATH}"
echo "服务端口: ${PORT}"
echo "日志目录: ${LOG_DIR}"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=4
export VLLM_LOGGING_LEVEL=INFO


nohup python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --tokenizer ${MODEL_PATH} \
    --served-model-name qwen3-4b_sft \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --max-num-batched-tokens 131072 \
    --max-num-seqs 64 \
    --block-size 16 \
    --dtype auto \
    --port ${PORT} \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enable-prefix-caching \
    --kv-cache-dtype auto \
    --disable-log-stats \
    --enable-chunked-prefill \
    > ${LOG_DIR}/vllm_sft_gpu4_$(date +%Y%m%d).log 2>&1 &

# 等待服务启动
sleep 3

# 查找vLLM进程PID（更可靠的方式）
VLLM_PID=$(pgrep -f "vllm.entrypoints.openai.api_server.*${PORT}")

if [ -n "$VLLM_PID" ]; then
    echo ${VLLM_PID} > ${LOG_DIR}/vllm.pid
    echo "✅ 服务已启动，PID: ${VLLM_PID}"
else
    # 备用方式：使用 $!
    echo $! > ${LOG_DIR}/vllm.pid
    echo "✅ 服务已启动，PID: $(cat ${LOG_DIR}/vllm.pid)"
fi

echo "📊 监控日志: tail -f ${LOG_DIR}/vllm_sft_gpu4_$(date +%Y%m%d).log"