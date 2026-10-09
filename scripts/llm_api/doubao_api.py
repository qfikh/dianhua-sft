from openai import OpenAI

# DEFAULT_MODEL = "doubao-1-5-lite-32k-250115"
# DEFAULT_MODEL = "doubao-1-6-pro-32k-250115"
# client = OpenAI(
#     base_url="https://ark.cn-beijing.volces.com/api/v3",
#     api_key="4cf2e06b-bba3-4247-ba06-88985e4963f7"
# )

DEFAULT_MODEL = "doubao-seed-2-1-turbo-260628" # 23.45ss   关闭思考模式 3.02s 
# DEFAULT_MODEL = "doubao-seed-2-1-pro-260628" # 27.12ss   关闭思考模式 4.48s 
# DEFAULT_MODEL = "doubao-seed-2-0-lite-260428" # 默认思考模式 平均16.30s  关闭思考模式 3.14s  
# DEFAULT_MODEL = "doubao-seed-2-0-mini-260428"  # 默认思考模式 平均3.60s   关闭思考模式 0.90s
# DEFAULT_MODEL = "doubao-1-5-lite-32k-250115" # 平均1.67s  1.68s
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="ark-cb5dd8da-baa0-4efc-953e-b10d2df81d65-ac844"
)

def call_doubao_with_messages(model=None, system_prompt=None, user_content=None, thinking_enabled=False):
    model = model or DEFAULT_MODEL
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if user_content:
        messages.append({'role': 'user', 'content': user_content})

    extra = {}
    if "doubao" in model:
        thinking_type = "enabled" if thinking_enabled else "disabled"
        extra["extra_body"] = {"thinking": {"type": thinking_type}}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        **extra
    )
    return response.choices[0].message.content

# 3. 入口（已删除多余逗号）
if __name__ == '__main__':
    import time
    questions = [
        "中国的首都是哪里？",
        "1+1等于几？",
        "太阳从哪边升起？",
        "水的化学式是什么？",
        "一年有多少个月？",
        "一周有几天？",
        "英语的谢谢怎么说？",
        "世界上最高的山是什么？",
        "红色和黄色混合是什么颜色？",
        "春节是农历几月几号？",
    ]
    times = []
    for i, q in enumerate(questions):
        if i > 0:
            time.sleep(1)
        t1 = time.time()
        result = call_doubao_with_messages(
            model=DEFAULT_MODEL,
            system_prompt="你是一个乐于助人的助手。",
            user_content=q
        )
        elapsed = time.time() - t1
        times.append(elapsed)
        print(f"[{i+1}/10] {elapsed:.2f}s | {q} => {result[:30]}")
    print(f"\n平均耗时: {sum(times)/len(times):.2f}s, 最快: {min(times):.2f}s, 最慢: {max(times):.2f}s")   
    