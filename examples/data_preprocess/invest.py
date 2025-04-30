"""
Preprocess dataset for invest task - given a target number and N numbers, generate invest result to reach target
"""

import re
import os
import pandas as pd
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse

views = {
    "momentum": "动量",
    "valuation": "估值",
    "sentiment": "情绪",
    "macro": "宏观",
}

# 主要目录
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(project_dir, "data", "invest")


def make_prefix(dp, template_type='base'):
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""用户与基金经理之间的对话。基金经理管理着一只基金，基金里面只投资沪深300指数，剩余全是现金，你需要每天根据市场信息做出对沪深300仓位增减的决策，目标是使最大化基金收益率，最小化回撤率。用户提供市场信息，基金经理负责解答。基金经理会先在脑海中思考推理过程，然后向用户提供一个答案，思考过程与答案不能相互矛盾。
用户：请分别参考动量领域的观点："{dp['momentum']}"；估值领域的观点："{dp['valuation']}"；情绪领域的观点："{dp['sentiment']}"；情绪领域的观点："{dp['macro']}"。利用上述领域的观点提供对股票的操作建议，在"加仓"、"减仓"或者"持仓不动"中选择一个。请在<think></think>标签中展示思考过程，并在<answer></answer>标签中返回最终答案，例如<answer>加仓</answer>；
基金经理：让我逐步解决这个问题。
<think>"""
    else:
        raise NotImplementedError(f"{template_type} is not implemented")
    return prefix


def download_data():
    import jqdatasdk
    import datetime
    excel_output_path = os.path.join(data_dir, "jqdatasdk_data.csv")
    if not os.path.exists(excel_output_path):
        # 下载数据
        user, pwd = '18610934225', 'Pingan112'
        jqdatasdk.auth(user, pwd)
        start = "2006-01-01"
        end = datetime.datetime.today().strftime('%Y-%m-%d')
        # end = '2018-01-01'
        sec_list = ['000300.XSHG']
        df = jqdatasdk.get_price(sec_list, start_date=start, end_date=end, frequency='daily', skip_paused=False)
        df.to_csv(excel_output_path, encoding='utf_8_sig', index=False)
        print(f"file saved: {excel_output_path}")
    else:
        # 读取数据
        df = pd.read_csv(excel_output_path)

    # 计算后五日的最大回撤
    df = df.sort_values('time')
    df['future_5d_high'] = df['high'].rolling(window=5, min_periods=1).max().shift(-4)
    df['future_5d_low'] = df['low'].rolling(window=5, min_periods=1).min().shift(-4)
    df['future_5d_drawdown'] = (df['future_5d_high'] - df['future_5d_low']) / df['future_5d_high']

    # 计算回撤的90分位数：
    drawdown_90pct = df['future_5d_drawdown'].quantile(0.90)
    df["drawdown_90pct"] = drawdown_90pct
    df['has_drawdown'] = (df['future_5d_drawdown'] > drawdown_90pct).astype(int)
    print(f"有回撤的比例：{df['has_drawdown'].sum() / len(df):.3f}")

    # 清理中间列（可选）
    df.drop(['future_5d_high', 'future_5d_low'], axis=1, inplace=True)

    df.to_csv(excel_output_path,  encoding='utf_8_sig', index=False)


def gen_dataset():
    import pandas as pd

    # 文件路径
    excel_input_path = os.path.join(data_dir, "generated_prompts_all.xlsx")
    excel_target_path = os.path.join(data_dir, "generated_prompts_hs300.xlsx")
    csv_jqdatasdk_path = os.path.join(data_dir, "jqdatasdk_data.csv")
    excel_output_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(csv_jqdatasdk_path):
        download_data()

    df_input = pd.read_excel(excel_input_path)
    df_jqdatasdk = pd.read_csv(csv_jqdatasdk_path)
    df_target = pd.read_excel(excel_target_path)
    # 共同索引
    df_jqdatasdk['trade_date'] = pd.to_datetime(df_jqdatasdk['time'], format='%Y-%m-%d')
    df_jqdatasdk = df_jqdatasdk.set_index('trade_date')
    df_input = df_input.set_index('trade_date')
    df_target = df_target.set_index('trade_date')
    common_index = df_input.index.intersection(df_target.index)
    common_index = df_jqdatasdk.index.intersection(common_index)
    if len(common_index) == 0:
        raise UserWarning('len(common_index) is 0')

    # 目标文件
    cols = ["target", "views", "has_drawdown", "data_source", "prompt", "ability", "reward_model", "extra_info"]
    train_data = {name: [] for name in cols}

    def check_holding_status(text):
        if "没有改变持仓" in text:
            return "持仓不动"
        elif "仓位提升" in text:
            return "加仓"
        elif "仓位减少" in text:
            return "减仓"
        else:
            raise UserWarning(f"{text}")

    for i in common_index:
        dp = {eng: eval(df_input.at[i, chi])[0]["content"] for eng, chi in views.items()}
        target = eval(df_target.at[i, "波动分析"])[0]["content"]
        has_drawdown = df_jqdatasdk.at[i, "has_drawdown"]
        for col in cols:
            if col == 'target':
                train_data[col].append(check_holding_status(target))
            elif col == 'views':
                train_data[col].append(dp)
            elif col == 'has_drawdown':
                train_data[col].append(has_drawdown)
            elif col == 'data_source':
                train_data[col].append("invest")
            elif col == 'prompt':
                train_data[col].append(make_prefix(dp))
            elif col == 'ability':
                train_data[col].append("invest")
            else:
                train_data[col].append(None)
    df_output = pd.DataFrame(train_data)
    df_output = df_output[~df_output["has_drawdown"].isna()]
    df_output.to_csv(excel_output_path, encoding='utf_8_sig', index=False)
    print()

def analyze_dateset():
    excel_input_path = os.path.join(data_dir, "train.csv")
    df_train = pd.read_csv(excel_input_path)
    data = []
    for name in ["加仓", "减仓", "持仓不动", "回撤"]:
        if name != "回撤":
            data.append({
                "目标": name,
                "数量": (df_train["target"] == name).sum(),
                "比例": (df_train["target"] == name).sum() / len(df_train),
            })
        else:
            data.append({
                "目标": name,
                "数量": df_train["has_drawdown"].sum(),
                "比例": df_train["has_drawdown"].sum() / len(df_train),
            })
    df_data = pd.DataFrame(data)
    excel_output_path = os.path.join(data_dir, "analyze.csv")
    df_data.to_csv(excel_output_path, encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    # analyze_dateset()
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=data_dir)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=900)
    parser.add_argument('--test_size', type=int, default=64)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    gen_dataset()
    data_source = 'invest'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    data_path = os.path.join(args.local_dir, "train.csv")
    raw_dataset = load_dataset('csv', data_files=data_path, split='train')

    # 检查数据集大小
    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE

    # 分割数据集
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            # question = make_prefix(example, template_type=args.template_type)
            question = example['prompt']
            solution = {
                "target": example['target'],
                "views": example['views'],
                "has_drawdown": example["has_drawdown"]
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "invest",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)