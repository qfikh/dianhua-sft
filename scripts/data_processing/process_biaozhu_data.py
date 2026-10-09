import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_api.qwen import call_with_messages

SYSTEM_PROMPT = """
你是一个对话分析专家。请根据提供的对话片段，判断是否存在角色颠倒（即service和client的角色标识与实际发言内容不符）。

角色标识规则：
- service: 表示客服
- client: 表示客户

判断标准：
只有当**明确**出现以下情况时，才判定为"角色异常"：
1. service标记的发言内容明显是客户才会说的话（如：描述自己的车辆问题、询问客服问题、表达个人诉求、投诉反馈等）
2. client标记的发言内容明显是客服才会说的话（如：问候语"您好，请问有什么可以帮您"、主动提供解决方案、询问客户信息、使用客服专业话术等）
3. client标记的发言内容明显是经销商/4S店才会说的话（如：以经销商身份回复客户问题、提供门店服务等）

重要原则：
- 默认判定为"角色正常"，只有**非常明确**的角色颠倒才判定为"角色异常"
- 如果存在疑问或不确定，请判定为"角色正常"
- 客户使用礼貌用语（如"好的""谢谢"）不算是角色异常
- 客服使用简短回应（如"嗯""好的"）不算是角色异常
- 允许存在连续两行或多行都是同一个角色的情况

角色正常的示例：
service: 好的，您稍等，我帮您查一下。
client: 嗯，好的。
service: 您看您这边车型是什么车型啊。
client: 车型是C200L。
（以上都是角色正常）

角色异常的示例：
service: 我的车打不着火了，怎么办？
client: 您好，请问您的车型是什么？
（service说了客户的话，client说了客服的话，角色异常）

service: 您好，请问有什么可以帮您？
client: 刚才有客户说他的车子在路上抛锚了，需要道路救援。
（client说了客服的话，角色异常）

你的输出格式：
- 只输出"角色异常"或"角色正常"，不要输出其他任何内容
"""

def detect_role_abnormal(dialogue):
    if pd.isna(dialogue):
        return False
    dialogue_str = str(dialogue).strip()
    lines = dialogue_str.split('\n')
    last_three_lines = lines[-4:] if len(lines) >= 3 else lines
    context = "\n".join(last_three_lines)
    try:
        response = call_with_messages(
            model="qwen-plus",
            system_prompt=SYSTEM_PROMPT,
            user_content=context
        )
        return "角色异常" in response
    except Exception as e:
        print(f"调用大模型失败: {e}")
        return False

def process_biaozhu_data(input_file, output_file):
    output_path = output_file
    test_mode= True
    test_rows= 200
    
    # 只读取 "标注结果" sheet
    df = pd.read_excel(input_file, sheet_name='标注结果')
    print(f"读取sheet: 标注结果, 列: {list(df.columns)}, 行数: {len(df)}")
    
    # 测试模式下只处理前 test_rows 行
    if test_mode:
        df = df.head(test_rows)
        print(f"测试模式: 只处理前 {test_rows} 行数据")
    
    role_abnormal_marks = []
    abnormal_count = 0
    total_count = 0
    
    for idx, row in df.iterrows():
        dialogue = row.iloc[1]
            
        if pd.isna(dialogue):
            role_abnormal_marks.append('')
            continue
        
        total_count += 1
        if detect_role_abnormal(dialogue):
            role_abnormal_marks.append('角色异常')
            abnormal_count += 1
        else:
            role_abnormal_marks.append('')
        
        if total_count % 100 == 0:
            print(f"已处理 {total_count} 条数据...")
    
    if len(df.columns) >= 6:
        df.iloc[:, 5] = role_abnormal_marks
    else:
        df['角色异常标记'] = role_abnormal_marks
    
    # 文本列只保留最后10行对话内容
    text_col_idx = 1
    df.iloc[:, text_col_idx] = df.iloc[:, text_col_idx].apply(
        lambda x: '\n'.join(str(x).split('\n')[-10:]) if pd.notna(x) else x
    )
    
    df.to_excel(output_path, index=False)
    
    print(f"处理完成！")
    print(f"总对话片段数：{total_count}")
    print(f"角色异常数：{abnormal_count}")
    print(f"输出文件：{output_path}")
    
    return output_path

if __name__ == "__main__":
    input_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0616垫话标注-替换嗯-交付数据-15362-第六版.xlsx"
    output_file = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0616第四轮垫话/0616垫话标注-替换嗯-交付数据-15362-第六版-角色判断.xlsx"

    process_biaozhu_data(input_file, output_file)
