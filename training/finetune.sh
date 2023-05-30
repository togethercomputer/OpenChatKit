#!/bin/sh

DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export SHOW_DATA=0

# Parse command line arguments.
#  -h, --help: Show help message.
#  -m, --base-model: Path to the base model.
#  -d, --dataset: Path to the dataset.
#  -c, --checkpoint: Path to the checkpoint.
#  -t, --total-steps: Total number of steps to train.
#  -s, --checkpoint-steps: Number of steps between checkpoints.
#  -b, --batch-size: Batch size.
#  -w, --work-dir: Path to the work directory.

for i in "$@"
do
case $i in
    -h|--help)
    echo "Usage: finetune.sh [OPTIONS]"
    echo "Options:"
    echo "  -m, --base-model-path: Path to the base model."
    echo "  -d, --dataset-path: Path to the dataset."
    echo "  -c, --checkpoint-path: Path to the checkpoint."
    echo "  -t, --total-steps: Total number of steps to train."
    echo "  -s, --checkpoint-steps: Number of steps between checkpoints."
    echo "  -h, --help: Show help message."
    echo "  -w, --work-path: Path to the work directory. Used to define default paths for base model, dataset, and checkpoint."
    exit 0
    ;;
    -w=*|--work-path=*)
    FINETUNE_WORK_DIR="${i#*=}"
    shift # past argument=value
    ;;
    -m=*|--base-model-path=*)
    FINETUNE_BASE_MODEL="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--dataset-path=*)
    FINETUNE_DATASET_PATH="${i#*=}"
    shift # past argument=value
    ;;
    -c=*|--checkpoint-path=*)
    FINETUNE_CHECKPOINT_PATH="${i#*=}"
    shift # past argument=value
    ;;
    -t=*|--total-steps=*)
    FINETUNE_TOTAL_STEPS="${i#*=}"
    shift # past argument=value

    # Check if the total steps is a number greater than 0.
    if ! [[ "${FINETUNE_TOTAL_STEPS}" =~ ^[0-9]+$ ]] || [ "${FINETUNE_TOTAL_STEPS}" -le 0 ]; then
        echo "Error: Total steps must be a number greater than 0."
        exit 1
    fi
    
    ;;
    -s=*|--checkpoint-steps=*)
    FINETUNE_CHECKPOINT_STEPS="${i#*=}"
    shift # past argument=value

    # Check if the checkpoint steps is a number greater than or equal to 0.
    if ! [[ "${FINETUNE_CHECKPOINT_STEPS}" =~ ^[0-9]+$ ]] || [ "${FINETUNE_CHECKPOINT_STEPS}" -lt 0 ]; then
        echo "Error: Checkpoint steps must be a number greater than or equal to 0."
        exit 1
    fi

    ;;
    -b=*|--batch-size=*)
    FINETUNE_BATCH_SIZE="${i#*=}"
    shift # past argument=value

    # Check if the batch size is a number greater than 0.
    if ! [[ "${FINETUNE_BATCH_SIZE}" =~ ^[0-9]+$ ]] || [ "${FINETUNE_BATCH_SIZE}" -le 0 ]; then
        echo "Error: Batch size must be a number greater than 0."
        exit 1
    fi
    *)
          # unknown option
    ;;
esac
done

# Set the base model path. If FINETUNE_BASE_MODEL is set, use that as the the
# base model path. Otherwise, if FINETUNE_WORK_DIR is set, use that to define
# base model path. Otherwise, use the default base model path.
if [ -n "${FINETUNE_BASE_MODEL}" ]; then
    BASE_MODEL="${FINETUNE_BASE_MODEL}"
elif [ -n "${FINETUNE_WORK_DIR}" ]; then
    # Use the work directory to define the base model path.
    BASE_MODEL="${FINETUNE_WORK_DIR}/model"
else
    # Model path is not set. Exit with error.
    echo "Error: Base model path is not set. Set FINETUNE_BASE_MODEL or FINETUNE_WORK_DIR."
    exit 1
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
    # Dataset path is not set. Exit with error.
    echo "Error: Dataset path is not set. Set FINETUNE_DATASET_PATH or FINETUNE_WORK_DIR."
    exit 1
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
    # Checkpoint path is not set. Exit with error.
    echo "Error: Checkpoint path is not set. Set FINETUNE_CHECKPOINT_PATH or FINETUNE_WORK_DIR."
    exit 1
fi

TOTAL_STEPS=${FINETUNE_TOTAL_STEPS:-20000}
CHECKPOINT_STEPS=${FINETUNE_CHECKPOINT_STEPS:-100}
BATCH_SIZE=${FINETUNE_BATCH_SIZE:-32}

# Initialize datasets with files in the DATASET_PATH directory
DATASETS=""
for file in "${DATASET_PATH}"/*.jsonl; do
    # Add each file to the DATASETS variable with a weight of 1.0
    # TODO: Add support for user specified weights
    if [[ -f "${file}" ]]; then
        if [[ -z "${DATASETS}" ]]; then
            DATASETS="${file}:1.0"
        else
            DATASETS="${DATASETS},${file}:1.0"
        fi
    fi
done

echo "DATASETS: ${DATASETS}"

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
--lr 1e-5 --seq-length 2048 --batch-size ${BATCH_SIZE} --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--num-layers 8 --embedding-dim 4096 \
--world-size 8 --pipeline-group-size 4 --data-group-size 2 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

# Function to handle SIGINT signal
function handle_sigint {
    echo "Received SIGINT. Killing all processes..."
    # Kill all child processes
    for pid in ${pids[@]}; do
        kill $pid
    done
    exit 1
}

# Trap the SIGINT signal and call the handler function
trap handle_sigint SIGINT

# Retrieve GPU IDs using nvidia-smi command
gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | awk -F',' '{print $1}' | tr '\n' ' ')
IFS=' ' read -ra gpu_ids_array <<< "$gpu_ids"
num_gpus=${#gpu_ids_array[@]}

# Create an array to store the process IDs
pids=()


# Iterate over the range of GPU IDs
for ((i=0; i<num_gpus; i++)); do
    # Get the current CUDA ID
    cuda_id=${gpu_ids_array[i]}

    # Launch the process with CUDA ID and rank
    python ${DIR}/dist_clm_train.py $(echo ${ARGS}) --cuda-id ${cuda_id} --rank ${i} &

    # Store the process ID in the array
    pid=$!
    pids+=(${pid})
    echo "Launching rank ${i} on GPU ${cuda_id} with PID ${pid}"
done

# Wait for all processes to finish
for pid in ${pids[@]}; do
    wait $pid
done

# Print a message when all processes have finished
echo "All training processes have finished."