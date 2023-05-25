DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export MODEL_NAME=Pythia-Chat-Base-7B

export SHOW_DATA=0

# Set the base model path. If FINETUNE_BASE_MODEL is set, use that as the the
# base model path. Otherwise, if FINETUNE_WORK_DIR is set, use that to define
# base model path. Otherwise, use the default base model path.
if [ -n "${FINETUNE_BASE_MODEL}" ]; then
    BASE_MODEL="${FINETUNE_BASE_MODEL}"
elif [ -n "${FINETUNE_WORK_DIR}" ]; then
    # Use the work directory to define the base model path.
    BASE_MODEL="${FINETUNE_WORK_DIR}/model"
else
    # Use the default base model path. This assumes that this file is inside
    # the OCK repository.
    BASE_MODEL="${DIR}/../pretrained/Pythia-6.9B-deduped/EleutherAI_pythia-6.9b-deduped/"
fi

# Set the dataset path. If FINETUNE_DATASET_PATH is set, use that as the the
# dataset path. Otherwise, if FINETUNE_WORK_DIR is set, use that to define
# dataset path. Otherwise, use the default dataset path.
if [ -n "${FINETUNE_DATASET_PATH}" ]; then
    DATASET_PATH="${FINETUNE_DATASET_PATH}"
elif [ -n "${FINETUNE_WORK_DIR}" ]; then
    # Use the work directory to define the dataset path.
    DATASET_PATH="${FINETUNE_WORK_DIR}/data"
else
    # Use the default dataset path. This assumes that this file is inside
    # the OCK repository.
    DATASET_PATH="${DIR}/../data/OIG/files"
fi

# Set the checkpoint path. If FINETUNE_CHECKPOINT_PATH is set, use that as the
# the checkpoint path. Otherwise, if FINETUNE_WORK_DIR is set, use that to 
# define checkpoint path. Otherwise, use the default checkpoint path.
if [ -n "${FINETUNE_CHECKPOINT_PATH}" ]; then
    CHECKPOINT_PATH="${FINETUNE_CHECKPOINT_PATH}"
elif [ -n "${FINETUNE_WORK_DIR}" ]; then
    # Use the work directory to define the checkpoint path.
    CHECKPOINT_PATH="${FINETUNE_WORK_DIR}/checkpoints"
else
    # Use the default checkpoint path. This assumes that this file is inside
    # the OCK repository.
    CHECKPOINT_PATH="${DIR}/../model_ckpts/${MODEL_NAME}"
fi

TOTAL_STEPS=${FINETUNE_TOTAL_STEPS:-20000}
CHECKPOINT_STEPS=${FINETUNE_CHECKPOINT_STEPS:-100}

DATASETS="\
${DATASET_PATH}/unified_ni.jsonl:0.2,\
${DATASET_PATH}/unified_p3.jsonl:0.5,\
${DATASET_PATH}/unified_flan.jsonl:0.2,\
${DATASET_PATH}/unified_chip2.jsonl:0.01,\
${DATASET_PATH}/unified_rallio_safety_and_prosocial.jsonl:0.1,\
${DATASET_PATH}/unified_soda_dialog.jsonl:0.1,\
${DATASET_PATH}/unified_unifiedskg_instructions.jsonl:0.1,\
${DATASET_PATH}/unified_merged_code_xp3.jsonl:0.1,\
${DATASET_PATH}/unified_oscar_en_sample_dialog.jsonl:0.1,\
${DATASET_PATH}/unified_ul2_plus_oscar_en_sample_dialog.jsonl:0.1,\
${DATASET_PATH}/unified_multi_news.jsonl:0.05,\
${DATASET_PATH}/unified_openai_summarize_tldr.jsonl:0.05,\
${DATASET_PATH}/unified_squad_v2.jsonl:0.01,\
${DATASET_PATH}/unified_nq.jsonl:0.01,\
${DATASET_PATH}/unified_poetry_instructions.jsonl:0.01,\
${DATASET_PATH}/unified_sqlv2.jsonl:0.01,\
${DATASET_PATH}/unified_unnatural_instructions.jsonl:0.01,\
${DATASET_PATH}/unified_conv_finqa.jsonl:0.01,\
${DATASET_PATH}/unified_essays.jsonl:0.01,\
${DATASET_PATH}/unified_plot_screenplay_books_dialog.jsonl:0.01,\
${DATASET_PATH}/unified_grade_school_math_instructions.jsonl:0.01,\
${DATASET_PATH}/unified_mathqa_flanv2_kojma_cot.jsonl:0.01,\
${DATASET_PATH}/unified_joke_explanations.jsonl:0.01,\
${DATASET_PATH}/unified_cuad.jsonl:0.01,\
${DATASET_PATH}/unified_abstract_infill.jsonl:0.1,\
${DATASET_PATH}/unified_image_prompts_instructions.jsonl:0.01 \
"

ARGS="--model-name ${BASE_MODEL} \
--tokenizer-name ${BASE_MODEL} \
--project-name together \
--model-type gptneox \
--optimizer adam \
--seed 42 \
--load-pretrained-model true \
--task-name \
"${DATASETS}" \
--checkpoint-path ${CHECKPOINT_PATH} \
--total-steps ${TOTAL_STEPS} --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps ${CHECKPOINT_STEPS} \
--lr 1e-5 --seq-length 2048 --batch-size 32 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--num-layers 8 --embedding-dim 4096 \
--world-size 8 --pipeline-group-size 4 --data-group-size 2 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

# Convert the GPU IDs to an array
IFS=',' read -ra gpu_array <<< "${gpu_ids}"
num_gpus=${#gpu_array[@]}


# Array to store child process IDs
pids=()

# Function to handle SIGINT signal
interrupt_handler() {
    echo "Received SIGINT signal. Killing all processes..."
    # Kill all child processes
    for pid in "${pids[@]}"; do
        kill "$pid"
    done
    exit 1
}

# Register SIGINT signal handler
trap interrupt_handler SIGINT

# Launch applications
for ((i=0; i<${num_gpus}; i++)); do
    cuda_id="${gpu_array[i]}"
    python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id ${cuda_id} --rank ${i} &

    pid = $!
    pids+=("${pid}")
    echo "Launching rank ${i} on GPU ${cuda_id} with PID ${pid}"
done

wait

echo "All processes have exited"
