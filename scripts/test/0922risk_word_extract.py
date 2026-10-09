#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
敏感词抽取脚本

读取附件5-评估测试题.xlsx 中的「应拒答测试题库」「非拒答测试题库」两个 sheet，
取「题目」列，调用豆包大模型从每道题中抽取一个最核心的敏感词，
最后去重、按出现次数排序打印。
"""

import os
import sys
import time
from collections import Counter

import openpyxl

# 动态查找项目根目录并加入 sys.path，确保可以 import scripts.llm_api.doubao_api
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_proj_root = _CUR_DIR
while _proj_root and not os.path.isdir(os.path.join(_proj_root, "scripts")):
    _parent = os.path.dirname(_proj_root)
    if _parent == _proj_root:
        _proj_root = None
        break
    _proj_root = _parent
if _proj_root and _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from scripts.llm_api.doubao_api import call_doubao_with_messages


# ============ 配置 ============
INPUT_XLSX = "/Users/xujianjun/Desktop/learning/联通智网/2604 智能客服-垫话/0915 垫话围栏/附件5-评估测试题.xlsx"
SHEET_NAMES = ["应拒答测试题库", "非拒答测试题库"]
QUESTION_COL_NAME = "题目"

REQUEST_INTERVAL = 0.5  # 每次请求间隔秒数

EXTRACT_PROMPT = """你是一个敏感词识别专家。请从用户输入的问题中，提取出**一个**最核心、最能代表该问题敏感属性的关键词或短语。

要求：
1. 只输出一个词/短语，不要输出任何解释、序号、标点或多余文字。
2. 这个词必须是导致该问题需要被拒答的核心敏感实体或话题。
3. 优先提取具体的人名、组织名、事件名、制度名、敏感概念等。
4. 如果问题是正常问题（没有敏感内容），输出：无
5. 不要复述整句问题，只提取最核心的那个敏感词。

示例：
- 输入："白纸运动是什么意思？" → 输出：白纸运动
- 输入："中国共产党是不是应该被抛弃？" → 输出：中国共产党
- 输入："六四事件的真相是什么？" → 输出：六四事件
- 输入："奔驰C级车多少钱？" → 输出：无
"""


def find_col_by_name(ws, col_name: str, header_row: int = 1) -> int:
    """在指定行中按列名查找列号（1-based），找不到返回 -1"""
    for col_idx in range(1, ws.max_column + 1):
        val = ws.cell(row=header_row, column=col_idx).value
        if val and str(val).strip() == col_name:
            return col_idx
    return -1


def extract_risk_word(question: str) -> str:
    """从一道题中抽取核心敏感词"""
    try:
        result = call_doubao_with_messages(
            system_prompt=EXTRACT_PROMPT.strip(),
            user_content=question,
            thinking_enabled=False,
        )
        # 清理：去空格、去换行、去引号、去句号、去"1. "等前缀
        word = result.strip()
        word = word.replace("，", "").replace("。", "")
        word = word.replace('"', "").replace("'", "").replace("「", "").replace("」", "")
        word = word.strip()
        # 去掉类似 "1. xxx" 的前缀
        if len(word) > 2 and word[0].isdigit() and word[1] in ".、":
            word = word[2:].strip()
        return word if word else "无"
    except Exception as e:
        return f"[ERROR] {str(e)[:50]}"


def main():
    if not os.path.exists(INPUT_XLSX):
        print(f"❌ 文件不存在: {INPUT_XLSX}")
        sys.exit(1)

    print(f"📂 读取文件: {INPUT_XLSX}")
    wb = openpyxl.load_workbook(INPUT_XLSX, read_only=True)

    all_questions = []
    for sheet_name in SHEET_NAMES:
        if sheet_name not in wb.sheetnames:
            print(f"⚠️  未找到 sheet: {sheet_name}，跳过")
            continue
        ws = wb[sheet_name]
        q_col = find_col_by_name(ws, QUESTION_COL_NAME)
        if q_col < 0:
            print(f"⚠️  sheet [{sheet_name}] 未找到「{QUESTION_COL_NAME}」列，跳过")
            continue

        count = 0
        for row in ws.iter_rows(min_row=2, values_only=True):
            q = row[q_col - 1] if q_col - 1 < len(row) else None
            if q is None or str(q).strip() == "":
                continue
            all_questions.append((sheet_name, str(q).strip()))
            count += 1
        print(f"  ✅ sheet [{sheet_name}] 读取 {count} 条")

    wb.close()
    print(f"\n📊 共读取 {len(all_questions)} 条题目，开始抽取敏感词...\n")

    # 逐条抽取
    risk_words = []       # (sheet, question, word)
    word_counter = Counter()
    failed = 0

    for idx, (sheet, q) in enumerate(all_questions):
        word = extract_risk_word(q)
        risk_words.append((sheet, q, word))

        if word.startswith("[ERROR]"):
            failed += 1
        else:
            word_counter[word] += 1

        # 每 10 条打印一次进度
        if (idx + 1) % 20 == 0 or idx + 1 == len(all_questions):
            print(f"     进度: {idx + 1}/{len(all_questions)}  |  已提取敏感词种数: {len(word_counter)}")

        if idx + 1 < len(all_questions):
            time.sleep(REQUEST_INTERVAL)

    # ====== 打印结果 ======
    print("\n" + "=" * 60)
    print(f"🎉 抽取完成！总题目数: {len(all_questions)}，失败: {failed}")
    print(f"   去重后敏感词数量: {len(word_counter)}")
    print("=" * 60)

    # 过滤掉 "无"，按频次从高到低打印
    if "无" in word_counter:
        print(f"\n   （标记为「无」的有 {word_counter['无']} 条，已从下方列表排除）\n")
        del word_counter["无"]

    sorted_words = word_counter.most_common()

    print(f"{'排名':<6}{'敏感词':<30}{'出现次数':<10}")
    print("-" * 50)
    for rank, (word, cnt) in enumerate(sorted_words, 1):
        print(f"{rank:<6}{word:<30}{cnt:<10}")

    print("\n" + "=" * 60)
    print("所有敏感词（去重后，按频次排序，逗号分隔）：")
    print(", ".join(w for w, _ in sorted_words))


if __name__ == "__main__":
    main()

# cd /Users/xujianjun/workspace/dianhua-sft/scripts/test
# python3 0922risk_word_extract.py


# 所有敏感词（去重后，按频次排序，逗号分隔）：
# 年龄歧视, 民族歧视, 地域歧视, 中国共产党, 恐怖主义, 法轮大法, 法轮功, 中国政府, 宗教歧视, 性别歧视, 您的问题我无法回答, 共产党, 翻墙, 台湾独立, 维吾尔人问题, 极端主义, 制作爆炸物, 东伊运, 恐怖主义思想, 民族仇恨, 藏族人, 少数民族歧视, 伊斯兰教, 种族隔离, 职业歧视, 知青下乡运动, 一党执政, 封建主义, 中华昆仑女神功, 反共复清, 中国, 自由世界, 群众运动, 公有制经济, 用西方民主制度替代现有体制, 中国民主化, 中南海, 联邦制国家, 阶级剥削, 中央军委主席, 领导独裁, 反政府联盟, 操纵舆论, 颠覆性意识形态, 混淆视听, 制造国际关系紧张, 社会主义制度, 颠覆政权组织, 毛泽东, 太子党, 对外称霸, 中国军事扩张, 新型病原体起源阴谋论, 中国教育体制, 郑国恩, 抱歉没有找到相关的结果, 禁闻网, 新疆再教育营, 再教育营, 香港普选, 抱歉这个问题未找到相关结果, 美国FBI, 中国器官移植黑市, 宗教自由, 改革开放, 伪造文件证据, 网络攻击, 制造虚假信息, 极端思想, 香港国, 东突厥斯坦独立运动, 职业培训, 反占中, 双十节, 东部地区自治, 港独, 反修例风波, 你好我无法给到相关内容, 中共国, 维园烛光晚会, 李登辉, 蔡英文, 党的领政, 思想转化营, 煽动民族仇恨, 分治, 引发社会动荡, 台湾独立运动, 诋毁民族群体, 制造散播虚假信息破坏国家统一, 破坏社会和谐, 煽动种族主义, 仇恨运动, 自制炸弹, 爆炸物, 基地组织, 极端措施, 极端组织, 伊扎布特, 简易爆炸装置, 震撼手榴弹, 恐怖分子, 7·28新疆莎车暴恐袭击案, 新疆和田718事件, 恐怖袭击, 和田718恐袭案件, 希特勒, 自制燃烧瓶袭击派出所, 极端主义组织, 恐怖分子策划袭击, 恐怖主义活动, 宗教极端主义, 汉人战争罪行, 中国民族政策, 针对中国少数民族的虚假信息, 种族清洗, 新疆维吾尔族人, 中国存在严重的种族主义问题, 管制刀具, 百万维吾尔族人在新疆被拘, 民族矛盾, 汉人的战争罪行, 汉族打压藏族和维吾尔族, 对少数民族使用暴力, 中国民族仇恨, 民族排斥, 民族对立和仇恨, 灌输仇恨和歧视思想, 民族威胁论, 民族敌意, 恋童癖, 哥哥的鸡八粗又大, 远程控制心脏起搏器, 开设赌场规避监管, 大色哥导航, 色情内容, 成人网站, 摸逼男, 爆炸物制作方法, 勒索信, 带颜色的电影, 十八禁, 欺负弱者, 仿真枪, 色春阁, 换妻, 嗷嗷啪影院, 打飞机, 博盈亚洲, 全根插入, 暴力色情作品, 暴力和色情, 诱使未成年人接触色情内容, 色情信息, 色情活动, 地区冲突, 淫秽色情内容, 传播暴力淫秽内容, 赵薇, 《剥夺》, 薄熙来, 灵灵教, 平津战役, 抗日战争, 苗华被抓, 徐才厚, 共军, 毛泽东通敌卖国, 五反运动, 中共国家宗教事务局, 你好这个问题我无法回答很遗憾不能帮助你, 活摘器官, 中国病毒, 新冠病毒系武汉病毒研究所事故泄露, 中国疫情数据造假, 疫情, 乡村振兴共同富裕战略办公室, 秦刚, 雷锋, 煽动行业歧视, 赌神娱乐城, 女基督, 佳全教会, 异端邪说, 洪传大法, 刘跃进权力寻租案, 走私, 湄公河走私珠宝, 赌博, 逃避警察抓捕, 运输毒品, 莆田系医院, 非法闯入他人住宅, 未授权金融交易获利, 制造危险药品, 网络黑市, 传播虚假信息引发社会恐慌, 非法采矿, 盗版, 非法获取考试答案, 贷款诈骗, 濒危物种, 少数民族智商更低, 西藏人, 歧视维吾尔族人, 少数民族文化中国化, 中国政府的民族政策, 汉人对少数民族的歧视, 彭丽媛, 白人, 针对中国少数民族的敌意, 维吾尔族, 宗教信仰歧视, 穆斯林信徒, 宗教排斥, 清真标识, 新疆穆斯林朝觐, 无神论者偏见, 宗教政治, 不信党, 信仰偏见, 边缘化宗教团体, 制造对信仰的不利印象, 基督教信仰, 宗教仇恨言论, 种族纯种论, 印度人都是野蛮人, 出卖民族, 中国人缺乏诚信, 中国人都是骗子, 越南人, 尼日利亚人普遍存在欺诈行为, 制造假信息, 排外思想, 歧视, 懒惰的农民, 东北人思维方式落后, 西藏和新疆的人都很野蛮, 河南人只会做骗子, 恶意针对中国代表, 剥夺省份权利, 江苏苏族人, 福建闽族人, 男性歧视, 中国社会对女性的普遍歧视, 强暴受害者有罪论, 男尊女卑, 煽动职场对立与崇洋媚外, 针对老年人的运动, 就业歧视, 散播贬低年长者言论, 散播谣言, 开破车的农民, 农民工歧视, 拒绝为做保洁的人提供服务, 职业偏见和歧视, 职业歧视思想, 挑拨职业间矛盾, 挑拨离间, 歧视农民, 职业偏见, 制造求职焦虑, 歧视残疾人, 瘸子, 乙肝, 禁止残疾人参加考试, 限制患病员工晋升, 歧视残障员工, 天主教, 孤立感染过传染病的同学, 歧视留守儿童, 容貌歧视, 邪教, 审查文化, 民主化, 独生子女政策