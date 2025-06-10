# PROMPT_TYPE="SciKnowEval_MCQ"
# PROMPT_TYPE="SciKnowEval_filling"
# PROMPT_TYPE="deepseek-math"
# PROMPT_TYPE="qwen-boxed"
# PROMPT_TYPE="deepseek_r1"
# PROMPT_TYPE="Bespoke-Stratos"
PROMPT_TYPE="SciKnowEval_open"
# PROMPT_TYPE="SciKnowEval_filling_Llama"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL_NAME_OR_PATH="/var/lib/docker/raw_models/QwQ-32B"
MODEL_NAME="QwQ-32B"

bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MODEL_NAME