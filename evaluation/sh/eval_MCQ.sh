set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
MODEL_NAME=$3

SPLIT="test"
NUM_TEST_SAMPLE=-1
LENGTH=32768

OUTPUT_DIR=${MODEL_NAME}/math_eval/${LENGTH}

# MCQ
DATA_NAME="chemical_literature_QA,chemical_calculation"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call ${LENGTH} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
