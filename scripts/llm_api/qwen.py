import random
from http import HTTPStatus
from dashscope import Generation, MultiModalConversation  # 建议dashscope SDK 的版本 >= 1.14.0
import dashscope  # 确保已安装dashscope SDK，版本建议 >= 1.14.0，可通过 pip install dashscope --upgrade 安装
import os
# 成功
# 如果环境变量配置无效请启用以下代码
# dashscope.api_key = 'sk-f6e88a8a1937466fa266daa30f99afde'  # 翠翠qwen优化券
# 0623 志磊给的
dashscope.api_key = 'sk-ws-H.RPXHHPR.YPRY.MEYCIQCkHnXGv1uvszkKPbx9r3AlvxvPSuB0bf7S4jBztiMNNQIhAJx6xOKDPyNZwqV7Bicih_RgkMdJ9Q2ZbDTIsE0V747H'  

def call_with_messages(model, system_prompt=None, user_content=None):
    
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if user_content:
        messages.append({'role': 'user', 'content': user_content})
    response = Generation.call(model=model,
                               messages=messages,
                               seed=random.randint(1, 10000),
                               result_format='message')
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        raise Exception('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


def call_multimodal(model, text, image_url=None):
    """调用多模态模型（如 qwen3.7-plus）
    Args:
        model: 模型名称，如 'qwen3.7-plus'
        text: 文本内容
        image_url: 图片URL（可选）
    Returns:
        模型返回的文本内容
    """
    if image_url:
        content = [
            {"image": image_url},
            {"text": text}
        ]
    else:
        content = [{"text": text}]

    messages = [{"role": "user", "content": content}]

    response = MultiModalConversation.call(
        model=model,
        messages=messages
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    import time
    # 测试千问
    print("=== 测试千问 ===")
    t1 = time.time()
    result = call_with_messages(
        # model="qwen-plus", # 通
        # model="qwen3.7-plus", # 不通
        model="qwen-max",# 通
        # model="qwen3.7-max",# 通
        system_prompt="You are a helpful assistant.",
        user_content="1+1等于几？"
    )
    print(f"千问({time.time()-t1:.2f}s): {result}")

    # 输出 {"status_code": 200, "request_id": "636f53c8-dcfa-9089-b5e1-30833c4cf0a8", "code": "", "message": "", "output": {"text": null, "finish_reason": null, "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "材料：\n西红柿2个，鸡蛋3个，油，盐，糖，葱花适量\n\n做法:\n\n1. 鸡蛋打入碗中，加入少许盐搅拌均匀。\n2. 西红柿洗净切块。\n3. 热锅凉油，油热后倒入鸡蛋液，用筷子快速搅拌成蛋花，盛出备用。\n4. 锅中再加少许油，放入西红柿翻炒至出汁，加少许糖提鲜。\n5. 倒入炒好的鸡蛋，翻炒均匀，加入适量盐调味。\n6. 最后撒上葱花，翻炒均匀即可。"}}]}, "usage": {"input_tokens": 25, "output_tokens": 133, "total_tokens": 158}}

    # 测试千问多模态
    print("=== 测试千问多模态 ===")
    t1 = time.time()
    result = call_multimodal(
        model="qwen3.6-plus", # 不通
        text="1+1等于几？"
        # image_url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/20230725/20230725164453.png"
    )
    print(f"千问多模态({time.time()-t1:.2f}s): {result}")

