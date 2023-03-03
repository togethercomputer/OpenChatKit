# Fine-tuning Open Foundation Models

## Quick Start

### (1) Setup

- Install PyTorch env: 

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge cupy nccl cudatoolkit=11.6

pip install transformers==4.21.1
pip install datasets
pip install netifaces
pip install zstandard
pip install wandb
```

As we use wandb to manage experiments, we should also configure `wandb` before running the code
```shell
wandb login
```

### (Optional) Install bitsandbytes for 8bit Adam

```shell
pip install bitsandbytes # optional, to use 8bit-adam
```

### (2) Download / Convert Pretrained Models

We provide pretrained model checkpoints that are sharded by layers:
- [GPT-J-6B](https://drive.google.com/file/d/1xy2EzUvUhelNyeDZ43iyIok2O7PUqwWB/view?usp=share_link)
- [GPT-NeoX-20B](https://drive.google.com/file/d/1Yj-_r-0kBSasfJpVhcKH2SdU36LjtJJc/view?usp=share_link)

Please download and unzip the above ckpts to fine-tune them. 
The path of unzipped model should be passed to `--model-name` and `--tokenizer-name` for fine-tuning.

Alternatively, you can convert HF models yourself:
```shell
python convert_from_hf_gptj.py --model-name EleutherAI/gpt-j-6B --save-dir pretrained_models
# or
python convert_from_hf_gptneox.py --model-name EleutherAI/gpt-neox-20b --save-dir pretrained_models
```

### (3) Prepare Data

*(WIP: Move data here?)*

Download from [Better-OIG-Together/data](https://github.com/togethercomputer/Better-OIG-Together/tree/main/data).
And set them to `--task-name` with sampling weights, e.g.:
```
--task-name \
{{PATH_TO_DATA_0}}:0.2,\
{{PATH_TO_DATA_1}}:0.2,\
{{PATH_TO_DATA_2}}:0.3,\
{{PATH_TO_DATA_3}}:0.3
```

### (4) Run Fine-Tuning

#### An Example of GPT-NeoX-20B

Please refer to `example_scripts/finetune_gptneox.sh`, which shows an example to fine-tune GPT-NeoX-20B.
The script will launch 8 processes with a pipeline parallel degree of 8 and a data parallel degree of 1.

In case of geo-distributed training, please first make sure the network interface is correctly set and the master (rank 0 worker) IP and port are accesible by all the workers.
After that, run the corresponding process on each GPU node.

```shell
# set enviroment vars
...

# run on each GPU node
python dist_clm_train.py ... --cuda-id 0 --rank ${GLOBAL_RANK}
```

### (4)\* Convert to HF format

Here are some examples to convert training ckpts to HF ckpts.

For GPT-J:
```shell
python convert_to_hf_gptj.py --ckpt-path model_checkpoints/gptj-test/2000 --save-path GPT-J-fine-tuned --n-stages 2 --n-layer-per-stage 14
```

And NeoX
```shell
python convert_to_hf_gptneox.py --ckpt-path model_checkpoints/gptneox-test/2000 --save-path GPT-NeoX-fine-tuned --n-stages 8 --n-layer-per-stage 6
```

Then use HF's `from_pretrained(GPT-J-fine-tuned)` to load the model and do inference.

## Arguments

Enviroment vars that should be set:
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
