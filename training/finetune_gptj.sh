netif=lo
export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export WANDB_NAME=gptj-fine-tune

export SHOW_DATA=0

PATH_TO_MODEL="{{PATH_TO_MODEL}}"

ARGS="--model-name ${PATH_TO_MODEL} \
--tokenizer-name ${PATH_TO_MODEL} \
--project-name together \
--model-type gptj \
--optimizer adam \
--seed 42 \
--load-pretrained-model true \
--task-name \
{{PATH_TO_DATA_0}}:0.2,\
{{PATH_TO_DATA_1}}:0.2,\
{{PATH_TO_DATA_2}}:0.3,\
{{PATH_TO_DATA_3}}:0.3 \
--checkpoint-path ./model_ckpts/$WANDB_NAME \
--total-steps 20000 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps 100 \
--lr 1e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 1 \
--dist-url tcp://127.0.0.1:7033 \
--num-layers 14 --embedding-dim 4096 \
--world-size 8 --pipeline-group-size 2 --data-group-size 4 \
--job-id 0 --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python dist_clm_train.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

