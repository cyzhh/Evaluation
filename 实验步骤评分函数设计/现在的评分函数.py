from difflib import SequenceMatcher
import re
import numpy as np
from tqdm import tqdm 
import json
from scipy.optimize import linear_sum_assignment


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
    # show_example(data)
    return data

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    # show_example(data)
    return data

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("="*50)
    print(f"Data has been written to {filename}")
    # show_example(data)

def parse_step(step):
    # 去掉 "Step X:" 部分
    step_content = step.split(":", 1)[1].strip() if ":" in step else step.strip()
    
    # 分离函数名和参数
    if "(" in step_content and step_content.endswith(")"):
        func_name = step_content.split("(", 1)[0].strip()
        params_str = step_content.split("(", 1)[1].rstrip(")")
        params = {}
        if params_str:  # 如果有参数
            for param in params_str.split(","):
                param = param.strip()
                if ":" in param:
                    key, value = param.split(":", 1)
                    params[key.strip()] = value.strip().strip('"')
        return func_name, params
    else:
        # 如果没有括号，整个内容作为函数名，无参数
        return step_content.strip(), {}

def compare_params(params1, params2):
    """计算参数相似度（键 + 值）"""
    if not params1 and not params2:
        return 1.0  # 均无参数
    
    # 键的相似度
    keys1, keys2 = set(params1.keys()), set(params2.keys())
    if keys2:
        key_score = sum(1 for key in keys2 if key in keys1) / len(keys2)
        # print(f"Key Score: {key_score}")
        
        # 值的相似度（仅比较共有键）
        value_score = 0.0
        matched_keys = 0
        for key in keys2:
            if key in keys1:
                val1, val2 = params1[key], params2[key]
                similarity = SequenceMatcher(None, str(val1).lower(), str(val2).lower()).ratio()
                value_score += similarity
                matched_keys += 1
        
        value_avg = value_score / len(keys2)
        return 0.5 * key_score + 0.5 * value_avg
    elif not keys1 and not keys2:
        return 1.0
    else:
        return 0.0
    # print(f"Value and matched: {value_score}, {matched_keys}")
    

count = 0

def step_similarity(step1, step2):
    global count

    """综合函数名和参数相似度"""
    func1, params1 = parse_step(step1)
    func2, params2 = parse_step(step2)
    
    # 函数名相似度
    # func_score = SequenceMatcher(None, func1.lower(), func2.lower()).ratio()
    # 函数名只有匹配为1份，否则为0
    func_score = 1 if func1.lower() == func2.lower() else 0
    if not func_score:
        # print('*'*20)
        # print('func1:', func1)
        # print('step2:', step2)
        # print('func2:', func2)
        # print('params2:', params2)
        # print('*'*20)
        count += 1
    # # 参数相似度
    param_score = compare_params(params1, params2)

    if func_score:
        return param_score
    else:
        return 0
    # return 0.1 * func_score + 0.9 * param_score

def needleman_wunsch(pred_steps, gt_steps, gap_penalty=0.1, match_score=2, mismatch_penalty=-2):
    n, m = len(pred_steps), len(gt_steps)
    dp = np.zeros((n + 1, m + 1))
    
    # Initialize gap penalties
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
    
    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Calculate similarity between steps
            sim = step_similarity(pred_steps[i-1], gt_steps[j-1])
            if sim > 0:  # Threshold for "match"
                score = match_score*sim
            else:
                score = mismatch_penalty
            
            dp[i][j] = max(
                dp[i-1][j-1] + score,  # Match/mismatch
                dp[i-1][j] + gap_penalty,  # Gap in pred_steps
                dp[i][j-1] + gap_penalty   # Gap in gt_steps
            )
    
    # Backtrack to find alignment
    align_pred, align_gt = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (
            match_score if step_similarity(pred_steps[i-1], gt_steps[j-1]) > 0.5 else mismatch_penalty
        ):
            align_pred.append(pred_steps[i-1])
            align_gt.append(gt_steps[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_penalty:
            align_pred.append(pred_steps[i-1])
            align_gt.append("---")  # Gap in gt_steps
            i -= 1
        else:
            align_pred.append("---")  # Gap in pred_steps
            align_gt.append(gt_steps[j-1])
            j -= 1
    
    align_pred.reverse()
    align_gt.reverse()
    
    # Calculate normalized score
    max_possible_score = match_score * min(n, m)
    normalized_score = dp[-1][-1] / max_possible_score if max_possible_score != 0 else 0
    
    return normalized_score, align_pred, align_gt



name = "patent_2016_v5_formal_20250415_v1"
# name = "Qwen2.5-7B-Instruct"
# name = "deepseek_v3"
# name = "deepseek_r1"
# prompt_type="qwen-patent-formal"
prompt_type="qwen-boxed"
data = read_jsonl(f"/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/outputs/{name}/math_eval/2048/patent_formal/test_{prompt_type}_-1_seed0_t0.0_s0_e-1.jsonl")
test_data = read_jsonl("/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/data/patent_formal/test.jsonl")
num = 1
data = data[:num]
test_data = test_data[:num]



for item in test_data:
    steps = item['result'].split("\nStep")
    new_steps = []
    for step in steps:
        if step != "":
            if "Step 1" not in step:
                step = "Step " + step
            new_steps.append(step)
    item['ground_truth'] = new_steps

for item in data:
    steps = item['code'][0].split("Step")
    new_steps = []
    for step in steps:
        if step != "":
            step = "Step " + step
            new_steps.append(step)
    item['pred'] = new_steps


# total_scores = 0
# valid_comparisons = 0

# for item1, item2 in tqdm(zip(data, test_data)):
#     pred_steps = item1['pred']
#     gt_steps = item2['ground_truth']
    
#     similarity_matrix = np.zeros((len(pred_steps), len(gt_steps)))
#     max_sim = 0
#     for i, s1 in enumerate(pred_steps):
#         for j, s2 in enumerate(gt_steps):
#             if i == len(pred_steps)-1 and j == len(gt_steps)-1:
#                 similarity_matrix[i, j] = 0
#                 continue
#             similarity_matrix[i, j] = step_similarity(s1, s2)
#             if similarity_matrix[i, j] > max_sim:
#                 max_s1 = s1; max_s2 = s2
#                 max_sim = similarity_matrix[i, j]
    
#     print(f"\nMax sim {max_sim}, pairs:\n{max_s1.strip()}\n{max_s2.strip()}")

#     row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
#     func_score = similarity_matrix[row_ind, col_ind].sum()
    
#     min_steps = min(len(pred_steps), len(gt_steps))
#     total_scores += func_score / len(gt_steps)

#     valid_comparisons += 1

# if valid_comparisons > 0:
#     final_accuracy = total_scores / valid_comparisons
#     print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
# else:
#     print("No valid comparisons (all step counts differ).")

# print('parser error', count)

total_scores = 0
valid_comparisons = 0

for pred_item, gt_item in tqdm(zip(data, test_data)):
    pred_steps = pred_item['pred']
    gt_steps = gt_item['ground_truth']
    
    # Compute alignment score
    score, aligned_pred, aligned_gt = needleman_wunsch(pred_steps, gt_steps)
    total_scores += score
    valid_comparisons += 1

    # (Optional) Print alignment for debugging
    print("\nAligned Steps:")
    for p, g in zip(aligned_pred, aligned_gt):
        print("*"*20)
        print(f"Pred: {p}\nGT:   {g}\n")

final_accuracy = (total_scores / valid_comparisons) * 100
print(f"Final Accuracy: {final_accuracy:.2f}%")
