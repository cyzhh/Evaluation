import json
from tqdm import tqdm
import os
import random
from openai import OpenAI, OpenAIError
import sys
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from grader import *
from parser import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL = "deepseek_v3"
TASKS = {
    "L1": ["chemical_literature_QA_1000"],
    "L2": ["reaction_mechanism_inference"],
    "L3": ["chemical_calculation", "chem_500mc"],
    # "L5": ["chemical_procedure_generation", "chemical_reagent_generation"]
    # "L3": ["balancing_chemical_equation"]
}

BASE_DATA_PATH = "/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/data/SciKnowEval"
BASE_OUTPUT_PATH = "/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/outputs"


def show_example(data):
    print(data[0].keys())
    print("Example:")
    for key, value in data[0].items():
        print(f"{key}: {value}")
    print("="*50)
    print(f"Total number of items: {len(data)}")

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    show_example(data)
    return data

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    show_example(data)
    return data

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("="*50)
    print(f"Data has been written to {filename}")
    show_example(data)

def extract_answer_from_boxed(text):
    ans = text.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def deepseek_r1(prompt):
    client = OpenAI(
        api_key="b314b522-7f28-4e8c-807c-363d9b8af8b7",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    response = client.chat.completions.create(
        model="deepseek-r1-250120",
        # model="deepseek-v3-241226",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=2048,
        temperature=0.9,
        top_p=0.7,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
    )
    # 确保返回的是一个包含 choices 的对象
    if hasattr(response, 'choices'):
        message = response.choices[0].message
        reasoning_content = getattr(message, 'reasoning_content', '')  # 使用 getattr 安全访问属性
        content = getattr(message, 'content', '')  # 使用 getattr 安全访问属性
        return reasoning_content, content
    else:
        return None, "Invalid response format: missing 'choices'"
    
def calculate_accuracy(data, task):
    """计算准确率"""
    acc = 0
    for item in data:
        pred = item['response']
        gt = parse_ground_truth(item, task)[1]
        pred = extract_answer(pred, task)
        if math_equal_process([pred, gt]):
            acc += 1
            item['score'] = True
            print(gt, pred)
    accuracy = acc / len(data)
    logging.info(f"Task: {task}, Accuracy: {accuracy:.4f}")
    return data, accuracy

def process_item(item):
    """处理单个样本"""
    pre_prompt = "Given a question and four options, please select the right answer. Please reason step by step, Your answer should be \"A\", \"B\", \"C\" or \"D\", and put your final answer within \\boxed{{}}."
    # pre_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
    prompt = item['question'] + "\n" + pre_prompt
    reasoning_response, response = deepseek_r1(prompt)
    item['reasoning_response'] = reasoning_response
    item['response'] = response
    return item

def process_task(level, task):
    """处理单个任务"""
    input_path = os.path.join(BASE_DATA_PATH, level, task, "test.jsonl")
    output_dir = os.path.join(BASE_OUTPUT_PATH, MODEL, "math_eval", "SciKnowEval", level, task)
    output_path = os.path.join(output_dir, "test.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    data = read_jsonl(input_path)
    logging.info(f"Processing task: {task} (Level: {level}), Total samples: {len(data)}")

    with ThreadPoolExecutor(max_workers=10) as executor:  # 调整 max_workers 以控制并发数
        futures = [executor.submit(process_item, item) for item in data]
        for future in tqdm(as_completed(futures), total=len(data), desc=f"Processing {task}"):
            try:
                item = future.result()
            except Exception as e:
                logging.error(f"Error processing item: {e}")

    data, accuracy = calculate_accuracy(data, task)

    write_jsonl(data, output_path)
    logging.info(f"Results saved to: {output_path}")
    return accuracy

def main():
    for level, tasks in TASKS.items():
        for task in tasks:
            try:
                process_task(level, task)
            except Exception as e:
                logging.error(f"Error processing task {task} (Level: {level}): {e}")

if __name__ == "__main__":
    main()
