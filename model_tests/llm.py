import asyncio
import pyarrow.parquet as pq
import os
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(PROJECT_PATH))
from tqdm import tqdm
from openai import AsyncOpenAI
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.reward_score.invest import compute_score, extract_solution
import pandas as pd
import inspect
from glob import glob

TEST_NUM = 32
MAX_TRY = 3
RESULT_PATH = os.path.join(PROJECT_PATH, 'model_tests', "docs")
DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'invest', 'test.parquet')

@contextmanager
def suppress_print():
    original_stdout = sys.stdout  # 保存原始 stdout
    sys.stdout = open(os.devnull, 'w')  # 重定向到 null
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout  # 恢复 stdout


class LLMTEST:
    def __init__(self, version):
        self.max_try = MAX_TRY
        self.version = version
        self.model_str = 'model_str'

    async def ask(self, prompt):
        return ''

    def test(self, num=TEST_NUM):
        table = pq.read_table(DATA_PATH)
        res = []
        num = table.num_rows if table.num_rows < num else num
        for i in tqdm(range(num), desc=f"{self.model_str}"):
            if i >= num:
                break
            row = table.slice(i, 1).to_pandas().iloc[0].to_dict()
            if inspect.iscoroutinefunction(self.ask):
                llm_res = asyncio.run(self.ask(prompt=row["prompt"][0]))
            else:
                llm_res = self.ask(prompt=row["prompt"][0])

            # 为模型强制加入身份标识
            think, answer = extract_solution(solution_str=llm_res)
            with suppress_print():
                score = compute_score(llm_res, row["reward_model"]["ground_truth"])
            res_i = {'score': score,
                     "answer": answer,
                     "think": think}
            res_i.update(row["reward_model"]["ground_truth"])
            res.append(res_i)

        df_res = pd.DataFrame(res)
        file_path = os.path.join(RESULT_PATH, f'{self.model_str}.csv')
        df_res.to_csv(file_path, index=False, encoding='utf_8_sig')



class Qwen(LLMTEST):
    def __init__(self, version='rl'):
        super().__init__(version)
        if version == 'rl':
            self.model_str = "DeepSeek-R1-Distill-Qwen-3B-RL"

            path = "/root/autodl-tmp/code/wangzhengzhuo512/TinyZero/checkpoints/TinyZero/invest-qwen2.5-3B/actor/global_step_600"
        elif version == 'none':
            self.model_str = "rem "
            path = "/root/autodl-tmp/models/llm/Qwen2.5-3B"
        else:
            raise NotImplementedError
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype="auto",          # 自动推断为 fp16 或 bf16
            device_map="auto"            # 自动分配到可用 GPU
        )

    def ask(self, prompt):
        inputs = self.tokenizer(prompt["content"], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class Deepseek(LLMTEST):
    def __init__(self, version='r1'):
        super().__init__(version)
        if version == 'r1':
            self.model_str="DeepSeek_R1"
        elif version == 'v3':
            self.model_str="DeepSeek_V3"

    async def _ask(self, prompt):
        # 正常的请求代码
        client = AsyncOpenAI(api_key="sk-d05d4ca9971248e39e0f1205b2581505", base_url="https://api.deepseek.com")
        response = await client.chat.completions.create(
            model=self.model_str,
            messages=[
                {"role": prompt["role"], "content": prompt["content"]},
            ],
            max_tokens=1024,
            temperature=0,
            stream=False,
            timeout=120,
        )
        answer = response.choices[0].message.content
        return "基金经理：" + answer

    async def ask(self, prompt):
        # 尝试多次请求
        for attempt in range(1, self.max_try + 1):
            try:
                return await asyncio.wait_for(self._ask(prompt), timeout=120)
            except asyncio.TimeoutError:
                print(f"第 {attempt} 次请求超时，重试中...")
            except Exception as e:
                print(f"第 {attempt} 次请求出错：{e}，重试中...")
        print("连续超时/失败 3 次，放弃请求。")
        return ''

def analyze_result():
    cols = {
        "model_name": "模型名称",
        "accuracy": "预测准确性",
        "mean_score": "预测得分",
        "accuracy_add": "加仓准确性",
        "accuracy_reduce": "减仓准确性",
        "accuracy_unchanged": "持仓不动准确性",
    }
    res = []
    for path in glob(os.path.join(RESULT_PATH, '*.csv')):
        fname, ext = os.path.splitext(os.path.basename(path))
        # 把结果文件跳过
        if 'result' in fname:
            continue
        df = pd.read_csv(path)
        res.append({
            "model_name": fname,
            "accuracy": round((df["answer"] == df["target"]).sum() / len(df["score"]), 3),
            "mean_score": round(df["score"].mean(), 3),
            "accuracy_add": round(((df["answer"] == df["target"]) & (df["target"] == '加仓')).sum() / (
                    df["target"] == '加仓').sum(), 3),
            "accuracy_reduce": round(((df["answer"] == df["target"]) & (df["target"] == '减仓')).sum() / (
                        df["target"] == '减仓').sum(), 3),
            "accuracy_unchanged": round(((df["answer"] == df["target"]) & (df["target"] == '持仓不动')).sum() / (
                    df["target"] == '持仓不动').sum(), 3),
        })
    df_res = pd.DataFrame(res)
    df_res.rename(columns=cols, inplace=True)
    df_res.to_csv(os.path.join(RESULT_PATH, 'result.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    # model = Qwen()
    # model.test(num=64)
    analyze_result()