# OpenChatKit

OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications. The kit includes an instruction-tuned 20 billion parameter language model, a 6 billion parameter moderation model, and an extensible retrieval system for including up-to-date responses from custom repositories. It was trained on the OIG-43M training dataset, which was a collaboration between Together, LAION, and Ontocord. Much more than a model release, this is the beginning of an open source project. We are releasing a set of tools and processes for ongoing improvement with community contributions. 

# Contents

- [OpenChatKit](#openchatkit)
- [Contents](#contents)
- [Requirements](#requirements)
- [Pre-trained Weights](#pre-trained-weights)
- [Datasets](#datasets)
  * [Data Contributions](#data-contributions)
- [Pretrained Base Model](#pretrained-base-model)
  * [GPT-NeoX-20B](#gpt-neox-20b)
- [Configuration](#configuration)
  * [Arguments](#arguments)
- [Training and Finetuning](#training-and-finetuning)
  * [(Optional) 8bit Adam](#optional-8bit-adam)
  * [Train GPT-NeoX-Chat-Base-20B](#train-gpt-neox-chat-base-20b)
- [Converting Weights to Huggingface Format](#converting-weights-to-huggingface-format)
- [Inference](#inference)
- [Monitoring](#monitoring)
  * [Weights & Biases](#weights--biases)
- [Retrieval Augmented Models](#retrieval-augmented-models)
- [License](#license)
- [Citing OpenChatKit](#citing-openchatkit)
- [Acknowledgements](#acknowledgements)

# Requirements

We highly recommend using Miniconda to isolate your environment.

```shell
conda create -f environment.yml
```

# Pre-trained Weights

We are making pre-trained weights for this model available at on Huggingface as `togethercomputer/GPT-NeoXT-Chat-Base-20B`.

# Datasets

```shell
python data/OIG/prepare.py
```

This command downloads the data from Huggingface and puts it in the `data/OIG-40M` directory.

## Data Contributions

# Pretrained Base Model

## GPT-NeoX-20B

```shell
python pretrained/GPT-NeoX-20B/prepare.py
```

This will download the model from Huggingface and convert it to the right format.

# Configuration

And set them to `--task-name` with sampling weights, e.g.:
```
--task-name \
{{PATH_TO_DATA_0}}:0.2,\
{{PATH_TO_DATA_1}}:0.2,\
{{PATH_TO_DATA_2}}:0.3,\
{{PATH_TO_DATA_3}}:0.3
```

The path of unzipped model should be passed to `--model-name` and `--tokenizer-name` for fine-tuning.

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
- `--profiling`: {no-profiling, tidy\_profiling}. `tidy_profiling` will generate profile jsons.

# Training and Finetuning

## (Optional) 8bit Adam

8bit Adam does...

```shell
pip install bitsandbytes # optional, to use 8bit-adam
```

## Train GPT-NeoX-Chat-Base-20B

```shell
bash training/train-gpt-neox-chat-base-20b.sh
```

This command places the model checkpoints in the `model_ckpt` directory.

Please refer to `example_scripts/finetune_gptneox.sh`, which shows an example to fine-tune GPT-NeoX-20B.

The script will launch 8 processes with a pipeline parallel degree of 8 and a data parallel degree of 1.

In case of geo-distributed training, please first make sure the network interface is correctly set and the master (rank 0 worker) IP and port are accesible by all the workers.
After that, run the corresponding process on each GPU node.

# Converting Weights to Huggingface Format

```shell
mkdir huggingface_models \
&& python tools/convert_to_hf_gptneox.py \
     --ckpt-path model_ckpts/GPT-Neo-XT-Chat-Base-20B/checkpoint_5 
     --save-path /huggingface_models/GPT-NeoXT-Chat-Base-20B 
     --n-stages 8 
     --n-layer-per-stage 6
```

# Inference

```shell
python inference/bot.py
```

to get the REPL and start chatting with the model. By default the script will load the model named GPT-NeoXT-Chat-Base-20B model under `model_ckpt` but you can override that behavior by specifying `--model` to the script.

```shell
$ python inference/bot.py 
Loading /home/csris/src/github.com/togethercomputer/OpenChatKit/inference/../huggingface_models/GPT-NeoXT-Chat-Base-20B to cuda:1...
Welcome to OpenChatKit shell.   Type /help or /? to list commands.

>>> Hello.
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Hello human.

>>> 
```

`ctrl-c` exits.

# Monitoring

The training code uses [loguru](https://github.com/Delgan/loguru) by default, but can be configured to report results to Weights & Biases.

## Weights & Biases

To use Weights & Biases, first login with your Weights & Biases token.

```shell
wandb login
```

# Retrieval Augmented Models

The code in `/retrieval` implements a python package to load a FAISS index and query it in memory (no web service).

Run

```shell
python data/wikipedia-blahblah/prepare.py
```

to download the wikipedia data and build into a FAISS index.

Run

```shell
python inference/bot.py --retrieval <index_path>
```

to enable retrieval mode.


# License

All code in this repository was developed by Together Computer.  Copyright (c) 2023, Together Computer.  All rights reserved. The code is licensed under the Apache 2.0 license.


```
Copyright 2023 Together Computer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

This repository also contains code written by a number of other authors. Such contributions are marked and the relevant licensing is included where appropriate.

For full terms, see the LICENSE file. If you have any questions, comments, or concerns about licensing please [contact us](https://www.together.xyz/contact).

# Citing OpenChatKit

```bibtex
@software{openchatkit,
  title = {{OpenChatKit: An Open Toolkit and Base Model for Dialogue-style Applications}},
  author = {Together Computer, LAION, Ontocord},
  url = {https://github.com/togethercomputer/OpenChatKit}
  month = {3},
  year = {2023},
  version = {0.15},
}
```

# Acknowledgements

Our model is a fine-tuned version of [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b), a large language model trained by [Eleuther AI](https://www.eleuther.ai). We evaluated our model on [HELM](https://crfm.stanford.edu/helm/latest/) provided by the [Center for Research on Foundation Models](https://crfm.stanford.edu). And we collaborated with both [CRFM](https://crfm.stanford.edu) and [HazyResearch](http://hazyresearch.stanford.edu) at Stanford to build this model.
