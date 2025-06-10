# PROMPT_TYPE="SciKnowEval_MCQ"
# PROMPT_TYPE="SciKnowEval_filling"
# PROMPT_TYPE="deepseek-math"
# PROMPT_TYPE="qwen-boxed"
PROMPT_TYPE="patent-formal"
# PROMPT_TYPE="qwen-patent-formal"
# PROMPT_TYPE="deepseek_r1"
# PROMPT_TYPE="Bespoke-Stratos"
# PROMPT_TYPE="SciKnowEval_open"
# PROMPT_TYPE="SciKnowEval_MCQ_Llama_fix"
# PROMPT_TYPE="SciKnowEval_MCQ_Llama"
# PROMPT_TYPE="SciKnowEval_filling_Llama"
# export CUDA_VISIBLE_DEVICES="5,6"
# export CUDA_VISIBLE_DEVICES="3,4"
export CUDA_VISIBLE_DEVICES="6,7"
MODEL_NAME_OR_PATH="/var/lib/docker/raw_models/Qwen2.5-32B-Instruct"
MODEL_NAME="Qwen2.5-32B-Instruct"
# MODEL_NAME_OR_PATH="/var/lib/docker/trained_models/project_v4_20250508_Qwen2.5-7B-Instruct_v3"

# Meta-Llama-3-8B-Instruct
# Qwen2.5-Math-7B
# DeepSeek-R1-Distill-Qwen-7B
# Qwen2-7B-Instruct
# Qwen2.5-Math-7B-Instruct
# Qwen2.5-7B-Instruct
# Qwen2.5-32B-Instruct
# DeepSeek-R1-Distill-Qwen-32B
# Qwen2.5-72B-Instruct

bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MODEL_NAME