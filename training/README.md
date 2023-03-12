# OpenChatKit Training

This directory contains code for training a chat model using OpenChatKit. The main training script is `finetune_GPT-NeoXT-Chat-Base-20B.sh`. 

To customize training, make a copy of the script and modify the arguments.

## Arguments

Environment vars that should be set:
```bash
export GLOO_SOCKET_IFNAME=lo # this interface should be consistent to `--net-interface`
export NCCL_SOCKET_IFNAME=lo # this interface should be consistent to `--net-interface`
export WANDB_NAME=gptj-test # wandb run name
```

The following arguments should be carefully set:
- `--model-name`: The path of model ckpt sharded by layers.
- `--tokenizer-name`: Usually the same to `--model-name`. You can also use HF's model name.
- `--model-type`: Indicate the model type. {gptj}. More model types will be added soon.
- `--num-layers`: Number of Transformer layers **for each GPU**. E.g. GPT-J has 28 layers, if we use two GPUs to form a pipeline, `--num-layers` should be 14.
- `--embedding-dim`: The hidden size of the model. GPT-J-6B is 4096. This is used to create buffers.
- `--dist-url`: URL of rank 0 worker (master). It is the same to all workers. And this URL should be accessible by all workers. For local training (single machine multiple GPUs), this can be like `--dist-url tcp://127.0.0.1:7033`
- `--world-size`: The total number of workers. `world-size == pipeline-group-size * data-group-size`
- `--pipeline-group-size`: Number of GPU workers for each pipeline
- `--data-group-size`: Number of data parallel workers. Also the number of pipelines.
- `--net-interface`: Network interface. Should be consistent with `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME`.

The following arguments can be tuned / changed:
- `--train-log-backend `: How to log the training info. {print, loguru, wandb}. 
- `--optimizer`: Optimizer type. {adam, 8bit-adam} (8bit-adam requires `pip install bitsandbytes`)
- `--load-pretrained-model`: Whether to load model weights. Usually `true`.
- `--task-name`: The task name or the path of a `jsonl` file. For multi-task training separate task names by `,`. 
   There is an optional sampling weight after each task name, separated by `:` (default is 1.0). Sampling weights will be normalized. 
   E.g. it should be like `--task-name cot:0.1,/path_task0.jsonl:1.0,/path_task0.jsonl:1.0,/path_task0.jsonl:1.0`.
- `--checkpoint-path`: Path to save fine-tuned checkpoints.
- `--checkpoint-steps`: Save ckpt every `checkpoint-steps`.
- `--total-steps`: Total number of steps for training. (This counts all `gradient-accumulate-step`s.)
- `--warmup-steps`: LR warmup steps.
- `--lr`: learning rate
- `--seq-length`: sequence length
- `--batch-size`: batch size for each GPU device (of each gradient accumulation step).
- `--micro-batch-size`: micro batch size for pipeline parallelism. 1 works fine.
- `--gradient-accumulate-step`: Accumulate gradients for several steps before updating parameters. This is another way to achieve large batch sizes when GPU memory is not enough.

The following arguments usually do not change:
- `--dp-backend`: {nccl, gloo}, default nccl.
- `--dp-mode`: {allreduce}.
- `--fp16`: Flag to enable FP16 mixed precision training. Should always adding it for the current impl.
- `--pp-mode`: always `gpipe`
- `--profiling`: {no-profiling, tidy_profiling}. `tidy_profiling` will generate profile jsons.
