import re
import random
from collections import defaultdict

key_words = {
    "momentum": ["情绪", "趋势", "惯性", "动能", "持续效应", "追涨杀跌", "相对强弱", "价格动量", "延续性", "动量效应"],
    "valuation": ["估值", "定价", "价值评估", "内在价值", "公允价值", "市价", "折现现金流", "市盈率", "市净率", "企业价值",],
    "sentiment": ["情绪", "市场", "投资者心理", "风险偏好", "悲观", "自信", "乐观", "投机情绪", "羊群效应", "从众心理", "逆情绪"],
    "macro": ["宏观", "经济环境", "政策", "全球经济", "经济周期", "通货膨胀", "利率", "GDP", "地缘政治", "系统性风险"],
}


def extract_solution(solution_str):
    """从解决方案字符串中提取思考过程和最终答案。"""
    # 这里不确定采用中文是否会影响结果
    result = {'think': None, 'answer': None}
    # Remove everything before the first "Assistant:"
    if "基金经理：" in solution_str:
        solution_str = solution_str.split("基金经理：", 1)[1]
    elif "<|im_start|>基金经理" in solution_str:
        solution_str = solution_str.split("<|im_start|>基金经理", 1)[1]
    else:
        return result["think"], result["answer"]

    # 提取思考过程
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, solution_str, re.DOTALL)
    if think_match:
        result['think'] = think_match.group(1).strip()

    # 提取最终答案
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, solution_str)
    if answer_match:
        result['answer'] = answer_match.group(1).strip()
    return result["think"], result["answer"]


def count_keyword_hits(reason_str) -> tuple[dict, float]:
    """返回每类的命中次数和命中率"""
    if reason_str is None:
        return {}, 0
    # 命中的字典
    hit_counts = defaultdict(int)
    # 记录命中的关键词
    used_keywords = set()

    for category, keywords in key_words.items():
        for keyword in keywords:
            # 如果关键词未被使用过且在 reason_str 中出现
            if keyword not in used_keywords and keyword in reason_str:
                hit_counts[category] += 1
                used_keywords.add(keyword)  # 标记为已使用

    return dict(hit_counts), round(len(hit_counts) / len(key_words), 2)

def check_answer_consistency(answer, think):
    if answer is None:
        answer = ''
    if think is None:
        think = ''
    # 定义各个操作的关键词
    keywords = {
        "加仓": ["加仓", "增持", "提高仓位", "增加持仓", "增加.*仓位"],
        "减仓": ["减仓", "减持", "降低仓位", "减少持仓", "减少.*仓位"],
        "持仓不动": ["持仓不动", "保持仓位", "维持现有仓位", "观望", "不做调整"]
    }

    # 标记在哪些操作中找到了关键词
    matched_actions = set()

    for action, patterns in keywords.items():
        for pattern in patterns:
            if re.search(pattern, think):
                matched_actions.add(action)
                break

    if len(matched_actions) == 0:
        return None  # 没有匹配到任何关键词

    if len(matched_actions) == 1 and answer in matched_actions:
        return True  # 推理与答案一致

    return False  # 答案和推理中匹配到的关键词不一致，存在矛盾


def compute_score(solution_str, ground_truth, method='strict', format_score=0.05, trend_score=0.5, score=1.):
    """The scoring function for countdown task.

    Args:
        solution_str: 解决方案文本字符串
        ground_truth: 目标、观点、是否发生了回撤
        method: 解决方案的提取方法
        format_score: 命中单个分类的关键词，但答案错误的得分
        trend_score: 预测 ‘减仓’ 正确后，后面发生回退额外加分
        score: 正确答案的得分
    """
    target = ground_truth['target']
    views = ground_truth['views']

    think, answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    # do_print = 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | views: {views}")
        print(f"Extracted answer : {answer}")
        print(f"Solution string  : {solution_str}")

    # 情况1： 无答案 / 答案格式错误 --> 0
    if answer is None:
        if do_print:
            print(f"结论未发现")
        return 0

    _, count_rate = count_keyword_hits(think)

    # 情况2： 答案错误
    if answer != target:
        # 情况2.1： 答案错误，但是命中了关键词
        if think is None:
            if do_print:
                print(f"结论错误")
            return 0
        else:
            if do_print:
                print(f"结论错误，关键词命中率 {count_rate * 100:.0f}%")
            return count_rate * format_score
    else:
        # 情况3： 答案正确
        if do_print:
            print(f"结论正确")
        if target == '无回撤':
            return 0.15
        else:
            return score


if __name__ == '__main__':
    sample_input = """
    金融投资专家:让我逐步解决这个问题。
    <think>
    1. 动量指标显示股票处于上升趋势，建议加仓。
    2. 估值指标显示股票被低估，支持加仓。
    3. 情绪指标显示市场情绪积极，支持加仓。
    4. 宏观经济指标稳定，没有重大风险。
    综合以上因素，建议加仓。
    </think>
    <answer>加仓</answer>
    """

    print(extract_solution(sample_input))