import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

from examples import get_examples


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"

    return EXAMPLES[data_name][:num_shots]


PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("### Problem: {input}\n### Solution:", " {output}", "\n\n"),
    "SciKnowEval_MCQ": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nGiven a question and four options, please select the right answer. Please reason step by step, Your answer should be \"A\", \"B\", \"C\" or \"D\", and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n"
    ),
    "SciKnowEval_filling": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n"
    ),
    "SciKnowEval_open": (
        "<|im_start|>system\nYou are an excellent expert in experimental protocol design. Given a user requirement for the experiment, and the materials that may be required, your task is to design the procedure of the experiment. Do not output any other characters.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n"
    ),
    "SciKnowEval_filling_Llama": (
        "<|user|>\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
        "<|assistant|>\n",
        "{output}",
        "\n\n"
    ),
    "SciKnowEval_MCQ_Llama": (
        "<|user|>\n{input}\nGiven a question and four options, please select the right answer. Please reason step by step, Your answer should be \"A\", \"B\", \"C\" or \"D\", and put your final answer within \\boxed{{}}."
        "<|assistant|>\n",
        "{output}",
        "\n\n"
    ),
    "SciKnowEval_filling_r1": (
        "<|im_start|>system\nYour role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.Please structure your response into two main sections: Thought and Solution.\n\nIn the Thought section, detail your reasoning process using the specified format:\n\n```\n<|begin_of_thought|>\n{{thought with steps seperated with \"\n\n\"}}\n<|end_of_thought|>\n```\n\nEach step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. Try to use casual, genuine phrases like: \"Hmm...\", \"This is interesting because...\", \"Wait, let me think about...\", \"Actually...\", \"Now that I look at it...\", \"This reminds me of...\", \"I wonder if...\", \"But then again...\", \"Let's see if...\", \"Alternatively...\", \"Let's summaize existing information...\", \"This might mean that...\", \"why/how/when/where...\", etc, to make your thought process be coherent, clear, and logically sound, effectively simulating human cognitive processes.\n\nIn the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows:\n\n```\n<|begin_of_solution|>\n{{final formatted, precise, and clear solution}}\n<|end_of_solution|>\n```\n\nNow, try to solve the following question through the above guidlines:\n<|im_end|>\n"
        "<|im_start|>user\n{input}\nGiven a question and four options, please select the right answer. Please reason step by step, Your answer should be \"A\", \"B\", \"C\" or \"D\", and put your final answer within \\boxed{{}}.<|im_end|>\n",
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n"
    ),
    "Bespoke-Stratos": (
        "<|im_start|>system\nYour role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {{thought with steps separated with '\n\n'}} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {{final formatted, precise, and clear solution}} <|end_of_solution|> Now, try to solve the following question through the above guidelines:\n<|im_end|>\n"
        "<|im_start|>user\nReturn your final response within \\boxed{{}}. {input}<|im_end|>\n",
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n"
    ),
    "qwen-patent-formal": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nPlease generate the experimental steps in formal language according to the experimental requirements. The parameters need to include the amount of substance, the state of the reaction, etc. \n\nExample:\nInput:\nHow to prepare 2-Ethyl-1,5-dimethoxy-3-trifluoromethyl-benzene?\n\nOutput:\nStep 1: Add(solute: \"2 - ethyl - 1,5 - dimethoxy - 3 - trifluoromethyl - benzaldehyde\", solute_amount: \"1.0 g\", solute_substance_amount: \"4.0 mmol\", solvent: \"THF\", solvent_volume: \"10 mL\", additive: \"NaBH4\", additive_amount: \"0.3 g\", additive_substance_amount: \"8.0 mmol\", temperature: \"0° C\")\nStep 2: Stir(temperature: \"0° C\", duration: \"1 h\")\nStep 3: Add(additive: \"water\", additive_volume: \"10 mL\")\nStep 4: Extract(solvent: \"EtOAc\", extraction_times: \"3\", solvent_volume_per_time: \"10 mL\")\nStep 5: Wash(washing_solvent: \"brine\", washing_solvent_volume: \"10 mL\")\nStep 6: Dry(drying_agent: \"Na2SO4\")\nStep 7: Filter()\nStep 8: Concentrate()\nStep 9: Yield(product: \"2 - Ethyl - 1,5 - dimethoxy - 3 - trifluoromethyl - benzene\", product_amount: \"0.8 g\", yield_percentage: \"90%\", product_state: \"colorless oil\")\n\nThe following are our experimental requirements:\nInput:\n{input}\n\nOutput:\n<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "patent-formal": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\n请你将以下的实验步骤转换成形式化语言，参数需要包含物质的量，反应的状态等。举个例子:Input:\nHeat: the reaction mixture is initially heated to 180° C\n\nOutput:Heat(target_temperature: \"190° C\", duration: \"1 hour\", starting_point: \"180° C\")\n\n如下是需要转换的实验步骤：Input:\n{actions}\n\nOutput:\n<|im_end|>\n",
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )
}



def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"

    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
    if args.prompt_type == "qwen25-math-cot":
        # Hotfix to support putting all demos into a single turn
        demo_prompt = splitter.join([q + "\n" + a for q, a in demos])
    else:
        demo_prompt = splitter.join(
            [
                input_template.format(input=q) + output_template.format(output=a)
                for q, a in demos
            ]
        )
    if args.prompt_type == "patent-formal":
        actions = ""
        for action in example["solution"]["reaction_actions"]:
            actions += "Step " + action + "\n"
        print(actions)
        context = input_template.format(input=actions)
    else:
        context = input_template.format(input=example["question"])
    if len(demo_prompt) == 0 or (
        args.adapt_few_shot and example["gt_ans"] not in ["A", "B", "C", "D", "E"]
    ):
        full_prompt = context
    else:
        if args.prompt_type == "qwen25-math-cot":
            # Hotfix to supportting put all demos into a single turn
            full_prompt = demo_prompt + splitter + example["question"]
            full_prompt = input_template.format(input=full_prompt)
        else:
            full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    # return full_prompt.strip(" ")
    return full_prompt


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
