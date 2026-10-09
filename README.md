# 垫话生成 SFT 项目

汽车客服垫话生成模型微调与推理服务。
# 服务器部署位置：
/data/xujianjun/workspace/dianhua_sft/swift_sft_on_testdata

## 项目结构

```
dianhua-sft/
├── build_sft_data.py      # 构建SFT训练数据
├── merge_model.py         # 合并LoRA权重到基座模型
├── vllm_client.py         # vLLM推理客户端
├── start_vllm_service.sh  # 启动vLLM服务脚本
├── train.json             # 训练数据
└── requirements.txt       # 依赖列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用

### 1. 启动 vLLM 服务

```bash
bash start_vllm_service.sh
```

### 2. 调用推理服务

```bash
python vllm_client.py
```




# 执行脚本的位置：

-   三种数据：1.5万条数据、2千条售前场景数据、线上人机对话数据
-   处理线上人机对话数据   tools.extract_dialog_from_logs() 
-   将所有的 service -> 客服
-   中英文冒号的问题：Service：
-   异常-对话过短   query_classify.py
-   异常-角色颠倒   query_classify.py
-   1.5万条数据  三种售前场景 类别标注   build_sft_data_v24_0623.py
-   1.5万条数据  结束语、感谢语、号牌 类别标注 modify_origin_data.py classify_and_modify_cushions():
-   所有数据：客户意图 标注  query_classify.py
-   调用大模型生成结果： optimize_dianhua.py




