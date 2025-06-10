from difflib import SequenceMatcher
import re
import os
import numpy as np
from tqdm import tqdm 
import json
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import time
from functools import lru_cache

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
    return data

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("="*50)
    print(f"Data has been written to {filename}")

def parse_step(step):
    """改进的步骤解析函数"""
    step = step.strip()
    if not step.startswith("Step"):
        step = "Step 1: " + step
    
    # 提取步骤内容
    step_content = step.split(":", 1)[1].strip() if ":" in step else step.strip()
    
    # 分离函数名和参数
    if "(" in step_content and step_content.endswith(")"):
        func_name = step_content.split("(", 1)[0].strip()
        params_str = step_content.split("(", 1)[1].rstrip(")")
        params = {}
        for param in params_str.split(","):
            param = param.strip()
            if ":" in param:
                key, value = [x.strip() for x in param.split(":", 1)]
                params[key] = value.strip('"')
        return func_name.lower(), params
    return step_content.lower(), {}

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
        count += 1
    # # 参数相似度
    alpha = 0.1
    param_score = (1-alpha) * compare_params(params1, params2) + alpha

    if func_score:
        return param_score
    else:
        return 0
    # return 0.1 * func_score + 0.9 * param_score

def needleman_wunsch(pred_steps, gt_steps, gap_penalty=-0.2, match_reward=1.0, mismatch_penalty=-1.0, min_sim_threshold=0):
    """Improved Needleman-Wunsch algorithm where gt gaps are treated as mismatches"""
    n, m = len(pred_steps), len(gt_steps)
    
    # Precompute similarity matrix
    sim_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            sim_matrix[i,j] = step_similarity(pred_steps[i], gt_steps[j])
    
    # Initialize DP matrix
    dp = np.zeros((n+1, m+1))
    dp[0,:] = np.arange(m+1) * mismatch_penalty
    dp[:,0] = np.arange(n+1) * gap_penalty
    
    # Fill DP matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            sim = sim_matrix[i-1,j-1]
            score = match_reward * sim if sim > min_sim_threshold else mismatch_penalty
            
            dp[i,j] = max(
                dp[i-1,j-1] + score,
                dp[i-1,j] + gap_penalty,
                dp[i,j-1] + mismatch_penalty
            )
    
    # Traceback to find alignment
    align_pred, align_gt = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if j > 0 and (i == 0 or dp[i,j] == dp[i,j-1] + mismatch_penalty):
            align_pred.append("---")
            align_gt.append(gt_steps[j-1])
            j -= 1
        elif i > 0 and (j == 0 or dp[i,j] == dp[i-1,j] + gap_penalty):
            align_pred.append(pred_steps[i-1])
            align_gt.append("---")
            i -= 1
        else:
            # Normal match case
            align_pred.append(pred_steps[i-1])
            align_gt.append(gt_steps[j-1])
            i -= 1
            j -= 1
    
    align_pred.reverse()
    align_gt.reverse()
    
    # Calculate evaluation metrics
    metrics = evaluate_alignment(align_pred, align_gt)
    
    # Calculate normalized score
    max_possible = match_reward * min(n, m)
    min_possible = gap_penalty * abs(n - m) + mismatch_penalty * (max(n, m) - min(n, m))
    # normalized_score = (dp[-1,-1] - min_possible) / (max_possible - min_possible)
    raw_score = dp[-1, -1] / m  # 范围: [-∞, 1]（最差可能 << -1，最优为 1）

    normalized_score = 100 * (1 / (1 + np.exp(-raw_score)))
    # print(dp)
    # print(dp[-1][-1])
    # print(align_pred)
    # print(align_gt)
    
    return normalized_score, align_pred, align_gt, metrics

def evaluate_alignment(pred, gt):
    """评估对齐质量"""
    matches = sum(1 for p, g in zip(pred, gt) if p != "---" and g != "---")
    pred_gaps = sum(1 for p in pred if p == "---")
    gt_gaps = sum(1 for g in gt if g == "---")
    
    precision = matches / (len(pred) - pred_gaps) if (len(pred) - pred_gaps) > 0 else 0
    recall = matches / (len(gt) - gt_gaps) if (len(gt) - gt_gaps) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred_gaps': pred_gaps,
        'gt_gaps': gt_gaps,
        'matches': matches
    }

def visualize_alignment(pred, gt, title="", save_dir="alignment_visualizations"):
    """改进的可视化函数，支持保存图片并确保文字完整显示"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置画布大小（根据步骤数量动态调整）
    fig_height = max(6, len(pred) * 0.5)  # 每行0.5英寸高度
    fig_width = 12  # 固定宽度
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    for p, g in zip(pred, gt):
        # 截断过长的文本（超过100字符）
        p_display = (p[:100] + '...') if len(p) > 100 else p
        g_display = (g[:100] + '...') if len(g) > 100 else g
        table_data.append([
            p_display if p != "---" else "GAP",
            g_display if g != "---" else "GAP"
        ])
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=["Predicted", "Ground Truth"],
        loc='center',
        cellLoc='left',
        colWidths=[0.45, 0.45]
    )
    
    # 调整单元格自动换行和字体大小
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # 减小字体大小
    table.auto_set_column_width([0, 1])  # 自动调整列宽
    
    # 高亮匹配项
    for i, (p, g) in enumerate(zip(pred, gt)):
        cell_color = '#ffffff'  # 默认白色背景
        if p != "---" and g != "---":
            similarity = step_similarity(p, g)
            # 根据相似度设置颜色梯度
            cell_color = plt.cm.Greens(0.3 + 0.7 * similarity)  # 相似度越高颜色越深
        table[(i+1, 0)].set_facecolor(cell_color)
        table[(i+1, 1)].set_facecolor(cell_color)
    
    # 调整表格样式
    table.scale(1, 1.2)  # 减小行高缩放因子
    
    # 设置标题
    plt.title(f"Step Alignment: {title}", pad=20, fontsize=10)  # 减小标题字体大小
    
    # 调整布局
    plt.tight_layout(pad=3.0)  # 增加边距
    
    # 生成安全文件名
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    save_path = os.path.join(save_dir, f"{safe_title}.png")
    
    # 保存图像（使用更高DPI和bbox_inches='tight'）
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"可视化结果已保存到: {save_path}")
    plt.close()  

def process_model_prompt_combination(name, prompt_type, start_num, end_num):
    try:
        print(f"\nProcessing {name} with {prompt_type}...")
        data = read_jsonl(f"/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/outputs/{name}/math_eval/2048/patent_formal/test_{prompt_type}_-1_seed0_t0.0_s0_e-1.jsonl")
        test_data = read_jsonl("/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/data/patent_formal/test.jsonl")
        
        data = data[start_num:end_num]
        test_data = test_data[start_num:end_num]

        # 预处理步骤数据
        for item in test_data:
            steps = [s for s in item['result'].split("\n") if s.strip()]
            item['ground_truth'] = []
            for i, step in enumerate(steps, 1):
                if not step.startswith(f"Step {i}"):
                    step = f"Step {i}: {step}"
                item['ground_truth'].append(step)

        for item in data:
            steps = [s for s in item['code'][0].split("\n") if s.strip()]
            item['pred'] = []
            for i, step in enumerate(steps, 1):
                if not step.startswith(f"Step {i}"):
                    step = f"Step {i}: {step}"
                item['pred'].append(step)

        total_scores = []
        all_metrics = []
        
        for idx, (pred_item, gt_item) in enumerate(zip(data, test_data)):
            pred_steps = pred_item['pred']
            gt_steps = gt_item['ground_truth']
            
            score, aligned_pred, aligned_gt, metrics = needleman_wunsch(pred_steps, gt_steps)
            total_scores.append(score)
            all_metrics.append(metrics)
            
            # 可视化第一个样本的对齐结果
            # if idx == 9:
            # visualize_alignment(aligned_pred, aligned_gt, f"{name} - {prompt_type}")

        # 计算平均指标
        avg_score = np.mean(total_scores)
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]) * 100,
            'recall': np.mean([m['recall'] for m in all_metrics]) * 100,
            'f1': np.mean([m['f1'] for m in all_metrics]) * 100,
            'avg_gaps': np.mean([m['pred_gaps'] + m['gt_gaps'] for m in all_metrics])
        }
        
        print(f"Results for {name} - {prompt_type}:")
        print(f"  Average Alignment Score: {avg_score}")
        print(f"  Precision: {avg_metrics['precision']:.2f}%")
        print(f"  Recall: {avg_metrics['recall']:.2f}%")
        print(f"  F1 Score: {avg_metrics['f1']:.2f}%")
        print(f"  Average Gaps per Sample: {avg_metrics['avg_gaps']:.1f}")
        
        return avg_score, avg_metrics
    
    except Exception as e:
        print(f"Error processing {name} - {prompt_type}: {str(e)}")
        return None, None


@lru_cache(maxsize=1000)
def parse_step_cached(step):
    """带缓存的步骤解析函数"""
    return parse_step(step)

@lru_cache(maxsize=2000)
def step_similarity_cached(step1, step2):
    """带缓存的步骤相似度计算"""
    return step_similarity(step1, step2)


if __name__ == "__main__":
    model_name_list = ["Qwen2.5-7B-Instruct", "patent_2016_v5_formal_20250415_v1", "deepseek_v3", "deepseek_r1"]
    prompt_type_list = ["qwen-patent-formal"]
    start_num = 0
    end_num = -1
    
    results = {}
    for name in model_name_list:
        for prompt_type in prompt_type_list:
            score, metrics = process_model_prompt_combination(name, prompt_type, start_num, end_num)
            if score is not None:
                results[f"{name}_{prompt_type}"] = {
                    'score': score,
                    'metrics': metrics
                }
    
    # 保存最终结果
    with open('alignment_results.json', 'w') as f:
        json.dump(results, f, indent=2)