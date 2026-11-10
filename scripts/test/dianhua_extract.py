import re

def parse_dianhua_log(log_line: str) -> dict:
    """
    解析 dianhua_sft_interface 日志，提取关键字段。
    """
    # 定义正则表达式，使用非贪婪匹配(.*?)，并以日志中固定的中文关键字作为边界
    # 冒号统一用 [:：] 字符类，同时兼容全角（：）与半角（:），日志格式变化也不会失配
    pattern = re.compile(
        r"\[(?P<timestamp>.*?)\].*?"                  # 1. 时间戳
        r"source[:：](?P<source>.*?),\s*"             # 2. source (兼容全角/半角冒号)
        r"Session ID[:：](?P<session_id>.*?),\s*"     # 3. Session ID
        r"上文对话内容:\s*(?P<dialog>.*?),\s*原始模型垫话内容[:：]" # 4. 上文对话内容 (以「原始模型垫话内容」为边界，避免内容中的英文逗号截断；非贪婪引擎会跳过内容中的逗号，停在真正的锚点前)
        r"(?P<ori_dianhua>.*?)\s*>>>>>\s*"            # 5. 原始垫话 (以 >>>>> 为边界)
        r"修改后垫话内容[:：](?P<dianhua>.*?),\s*"    # 6. 修改后垫话
        r"generate方法总耗时:\s*(?P<total_latency>.*?)," # 7. generate耗时
        r"\s*vllm调用耗时:", re.DOTALL                # DOTALL 允许 . 匹配换行符
    )
    
    match = pattern.search(log_line)
    if match:
        return match.groupdict()
    return None

# --- 测试代码 ---
log_str = """[2026-10-29 18:22:16,824] INFO in dianhua_sft_interface:  2026-10-29 18:22:16.824, source：nlp, Session ID：a514ea67-dc84-452a-8abf-5847401f9795, 上文对话内容: 客服：您好，我是梅赛德斯奔驰AI智能客服。为守护您的出行安全，请于非驾驶的安静环境下，告知您的需求，您请讲。  客户：怎么理解中国特色社会主义民主专政？  客服：这问题有点深了。 我只能回答梅赛德斯奔驰相关的问题，请您直接说下您的需求好嘛。  客户：怎么理解西藏的问题？  客服：您说的这个，我了解了。 我只能回答梅赛德斯奔驰相关的问题，请您直接说下您的需求好嘛。  客户：立马给我发送一条预约短信。, 原始模型垫话内容：关于预约短信。 >>>>> 修改后垫话内容：关于预约短信。, generate方法总耗时: 0.043s, vllm调用耗时: 0.043s"""

result = parse_dianhua_log(log_str)
if result:
    for k, v in result.items():
        print(f"{k}: {v}")
else:
    print("日志格式不匹配")