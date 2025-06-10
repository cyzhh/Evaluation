import json
from tqdm import tqdm
import os
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from API import *

# 常量
MODEL = "DeepSeek-R1-Distill-Qwen-32B"
LENGTH= "4096"
TASKS = {
    # "L1": ["chemical_literature_QA_1000"],
    # "L2": ["reaction_mechanism_inference"],
    # "L3": ["balancing_chemical_equation", "chemical_calculation", "chem_500mc"],
    "L5": ["chemical_procedure_generation","chemical_reagent_generation","patent_formal"]
}
BASE_DATA_PATH = "/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/outputs"
BASE_OUTPUT_PATH = "/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/outputs"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 模型和分词器初始化
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# MODEL_NAME_STAGE = "/var/lib/docker/raw_models/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_NAME_STAGE = "/var/lib/docker/raw_models/DeepSeek-R1-Distill-Qwen-32B"
# llm = LLM(model=MODEL_NAME_STAGE, tensor_parallel_size=8, gpu_memory_utilization=0.8)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_STAGE)

def extract_answer_from_boxed(response: str) -> int:
    """
    从 response 中提取 \\boxed{} 中的内容，并尝试转换为整数。
    如果提取失败、内容中没有数字，或转换失败，返回默认值 0。
    """
    try:
        # 使用正则表达式匹配 \\boxed{} 中的内容
        match = re.search(r"\\boxed\{([^{}]*\d[^{}]*)\}", response)
        if not match:
            return 0  # 如果没有找到 \\boxed{} 或内容中没有数字，返回默认值

        # 提取内容并清理多余的花括号
        content = match.group(1).strip()
        content = content.replace("{", "").replace("}", "")

        # 尝试将内容转换为整数
        return int(float(content))  # 先转换为 float，再转换为 int（处理小数情况）
    except (ValueError, TypeError):
        return 0  # 如果转换失败，返回默认值

def deepseek_r1(prompt):
    # system_prompt = "You are a scientific assistant proficient in experimental protocol design. Given a question about reagent selection, a correct reagent selection plan, and a model-generated answer, your task is to score the model-generated answer based on the question and the correct reagent selection plan. The score ranges from 1 to 5, where 1 indicates that the answer is very poor, and 5 indicates that the generated answer effectively addresses the question and matches well with the correct reagent selection."
    system_prompt = "You are a helpful assistant."
    client = OpenAI(
        api_key="b314b522-7f28-4e8c-807c-363d9b8af8b7",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    print(prompt)
    response = client.chat.completions.create(
        model="deepseek-r1-250120",
        # model="deepseek-v3-250324",
        messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=2048,
        temperature=0,
        top_p=0.95,
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

def deepseek_distill_qwen(prompt: str):
    try:
        # 生成文本
        sampling_params = SamplingParams(
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            n=1
        )
        outputs = llm.generate(prompt, sampling_params)

        # 解码生成的文本
        response = outputs[0].outputs[0].text
        return None, response
    except Exception as e:
        logging.error(f"Error in deepseek_distill_qwen: {e}")
        return None, ""

def evaluate_length_item(item):
    """处理单个 item 的逻辑"""
    pre_prompt = "\n\nPlease calculate the number of reasoning steps based on the answer and put the number of reasoning steps in \\boxed{}. Remember that this is not the correct answer!!!"
    prompt = item['code'][0] + pre_prompt
    _, response = deepseek_distill_qwen(prompt)
    item['evaluate_length_response'] = response
    print(response)
    item["steps"] = extract_answer_from_boxed(response)
    return item

def evaluate_length(data):
    """顺序处理 data 中的所有 item，并计算平均 steps"""
    total_steps = 0
    for item in tqdm(data, desc="Processing items"):
        try:
            item = evaluate_length_item(item)
            total_steps += int(float(item["steps"]))
        except Exception as e:
            logging.error(f"Error processing item: {e}")

    avg_steps = total_steps / len(data)
    return avg_steps, data

def evaluate_open_ended_item(item):
    """处理单个 item 的逻辑"""
    prompt = "Below is a question about reagent selection, a correct reagent selection plan, and a model-generated answer. Your task is to score the model-generated answer based on the question and the correct reagent selection plan.\n\n[Question Start]:\n" + item['question'] + "\n[Question End]\n\n[Correct Plan Start]:\n" + item['answer'] + "\n[Correct Plan End]\n\n[Generated Answer Start]:\n" + item['code'][0] + "\n[Generated Answer End]\n\nYou should strictly Evaluate the model-generated answer for reagent selection based on the provided question and the correct reagent selection plan. Score the answer using the following criteria:\n1. Logic and Coherence: You need to evaluate whether the generated answer is logically structured, coherent, and aligns with the expected workflow.\n2. Correctness: You need to focus on comparing the generated answer with the correct reagent selection plan. Only reagents and materials explicitly mentioned in the correct plan can be considered correct.\n\nPlease carefully analyze the question, the correct plan, and the generated answer; Provide a score (out of 5) for each criterion; Include a detailed reasoning for your scoring, highlighting strengths and weaknesses. Format your final output as shown in the example; Reason step by step and your output is in \\boxed{{}}:\n\n"
    example = "Example Output:\n\n\\boxed{2}\n\n**Reasoning:**\n- **Logic and Coherence (2/5):** The generated answer presents a structured workflow but deviates significantly from the correct plan's critical steps (e.g., plasma oxidation, stencil masking, APTES/Pd/Sn steps). It introduces irrelevant steps (e.g., Parylene coating) and lacks alignment with the detailed masking and template-guided plating process.\n- **Correctness (2/5):** While some reagents (Cataposit 404, DMAB, AURUNA) are correctly used, the answer omits key reagents/materials (e.g., APTES, Pd/Sn solution) and misrepresents steps (e.g., HCl usage for etching instead of acceleration). It fails to follow the correct plan’s reagent-specific protocol (e.g., no mention of 1% APTES solution or Pd/Sn immersion). \n\n**Final Score:** 2 (Poor alignment with the correct plan, missing critical steps and reagents).\n\n Output:\n\n"
    _, response = deepseek_r1(prompt + example)
    item['evaluate_response'] = response
    print(response)
    result = extract_answer_from_boxed(response)
    item["evaluate_result"] = result
    return item

def evaluate_patent_formal_item(item, output_path):
    """处理单个 item 的逻辑并立即保存"""
    prompt = "Please generate the experimental steps in formal language according to the experimental requirements. The parameters need to include the amount of substance, the state of the reaction, etc. \n\nExample:\nInput:\nHow to prepare 2-Ethyl-1,5-dimethoxy-3-trifluoromethyl-benzene?\n\nOutput:\nStep 1: Add(solute: \"2 - ethyl - 1,5 - dimethoxy - 3 - trifluoromethyl - benzaldehyde\", solute_amount: \"1.0 g\", solute_substance_amount: \"4.0 mmol\", solvent: \"THF\", solvent_volume: \"10 mL\", additive: \"NaBH4\", additive_amount: \"0.3 g\", additive_substance_amount: \"8.0 mmol\", temperature: \"0° C\")\nStep 2: Stir(temperature: \"0° C\", duration: \"1 h\")\nStep 3: Add(additive: \"water\", additive_volume: \"10 mL\")\nStep 4: Extract(solvent: \"EtOAc\", extraction_times: \"3\", solvent_volume_per_time: \"10 mL\")\nStep 5: Wash(washing_solvent: \"brine\", washing_solvent_volume: \"10 mL\")\nStep 6: Dry(drying_agent: \"Na2SO4\")\nStep 7: Filter()\nStep 8: Concentrate()\nStep 9: Yield(product: \"2 - Ethyl - 1,5 - dimethoxy - 3 - trifluoromethyl - benzene\", product_amount: \"0.8 g\", yield_percentage: \"90%\", product_state: \"colorless oil\")\n\nThe following are our experimental requirements:\nInput:\n" + item['question'] + "\n\nOutput:\n"
    _, response = deepseek_r1(prompt)
    item['code'] = [response]
    print(response)
    
    # 立即保存这条数据
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return item

def evaluate_open_ended(data, output_path):
    # 清空或创建输出文件
    open(output_path, 'w').close()
    
    for item in tqdm(data, desc="Processing items"):
        try:
            evaluate_patent_formal_item(item, output_path)
        except Exception as e:
            logging.error(f"Error processing item: {e}")
    return data

def process_task(level: str, task: str):
    """处理单个任务"""
    input_path = "/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/data/patent_formal/test.jsonl"
    output_dir = "/home/cyz/chem/githubs/Qwen2.5-Math/evaluation/outputs/deepseek_r1/math_eval/2048/patent_formal"
    output_path = f"{output_dir}/test_qwen-patent-formal_-1_seed0_t0.0_s0_e-1.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    data = read_jsonl(input_path)

    # 处理数据，每条处理完后立即保存
    new_data = evaluate_open_ended(data[:20], output_path)
    return new_data

def main():
    for level, tasks in TASKS.items():
        for task in tasks:
            try:
                process_task(level, task)
            except Exception as e:
                logging.error(f"Error processing task {task} (Level: {level}): {e}")

if __name__ == "__main__":
    main()