import os
from http import HTTPStatus
from dashscope import Generation
import dashscope

dashscope.base_http_api_url = 'https://ws-9xiq8az2xktl0sd8.cn-beijing.maas.aliyuncs.com/api/v1'
dashscope.api_key = 'sk-ws-H.RPXHHPR.YPRY.MEYCIQCkHnXGv1uvszkKPbx9r3AlvxvPSuB0bf7S4jBztiMNNQIhAJx6xOKDPyNZwqV7Bicih_RgkMdJ9Q2ZbDTIsE0V747H'


def call_deepseek(model="deepseek-v4-pro", system_prompt=None, user_content=None, enable_thinking=False):
    """调用 DeepSeek 模型（通过阿里云百炼平台）"""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if user_content:
        messages.append({'role': 'user', 'content': user_content})

    response = Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        # enable_thinking=enable_thinking,
        stream=False
    )

    if response.status_code == HTTPStatus.OK:
        message = response.output.choices[0].message
        result = {"content": message.content}
        try:
            if message.reasoning_content:
                result["reasoning_content"] = message.reasoning_content
        except (KeyError, AttributeError):
            pass
        return result
    else:
        raise Exception('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    import time

    print("=== 测试DeepSeek ===")
    t1 = time.time()
    result = call_deepseek(
        model="deepseek-v4-pro",
        system_prompt="You are a helpful assistant.",
        user_content="1+1等于几？",
        enable_thinking=False
    )
    if "reasoning_content" in result:
        print("=" * 20 + "思考过程" + "=" * 20)
        print(result["reasoning_content"])
    print("=" * 20 + "完整回复" + "=" * 20)
    print(result["content"])
    print(f"耗时: {time.time()-t1:.2f}s")
