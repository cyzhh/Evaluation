
PROMPT_TYPE="SciKnowEval_MCQ"
export CUDA_VISIBLE_DEVICES="7"
MODEL_NAME_OR_PATH="/var/lib/docker/raw_models/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-7B"

bash sh/eval_MCQ.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $MODEL_NAME