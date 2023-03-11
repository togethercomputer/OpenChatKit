# OpenChatKit

OpenChatKit provides a powerful, open-source base to create both specialized and general purpose chatbots for various applications. The kit includes an instruction-tuned 20 billion parameter language model, a 6 billion parameter moderation model, and an extensible retrieval system for including up-to-date responses from custom repositories. It was trained on the OIG-43M training dataset, which was a collaboration between [Together](https://www.together.xyz/), [LAION](https://laion.ai), and [Ontocord.ai](https://ontocord.ai). Much more than a model release, this is the beginning of an open source project. We are releasing a set of tools and processes for ongoing improvement with community contributions. 

In this repo, you'll find code for:
- Training an OpenChatKit model
- Testing inference using the model
- Augmenting the model with additional context from a retrieval index

# Contents

- [Requirements](#requirements)
- [Pre-trained Weights](#pre-trained-weights)
- [Datasets](#datasets)
  * [Data Contributions](#data-contributions)
- [Pretrained Base Model](#pretrained-base-model)
- [Training and Finetuning](#training-and-finetuning)
  * [(Optional) 8bit Adam](#optional-8bit-adam)
  * [Train GPT-NeoX-Chat-Base-20B](#train-gpt-neox-chat-base-20b)
- [Converting Weights to Huggingface Format](#converting-weights-to-huggingface-format)
- [Inference](#inference)
- [Monitoring](#monitoring)
  * [Loguru](#loguru)
  * [Weights & Biases](#weights--biases)
- [Experimental: Retrieval-Augmented Models](#experimental-retrieval-augmented-models)
- [License](#license)
- [Citing OpenChatKit](#citing-openchatkit)
- [Acknowledgements](#acknowledgements)

# Requirements

Before you begin, you need to install PyTorch and other dependencies.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) from their website.
2. Create an environment called OpenChatKit using the `environment.yml` file at the root of this repo.

```shell
conda env create -f environment.yml
```

This repo also uses [Git LFS](https://git-lfs.com/) to manage some files. Install it using the instructions on their site then run:

```shell
git lfs install
```

# Pre-trained Weights

GPT-NeoXT-Chat-Base-20B is a 20B-parameter variant of GPT-NeoX, fine-tuned on conversational datasets. We are releasing pre-trained weights for this model as [togethercomputer/GPT-NeoXT-Chat-Base-20B](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B) on Huggingface.

More details can be found on the model card for [GPT-NeoXT-Chat-Base-20B](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B) on Huggingface.

# Datasets

The chat model was trained on the [OIG](https://huggingface.co/datasets/laion/OIG) dataset built by [LAION](https://laion.ai/), [Together](https://www.together.xyz/), and [Ontocord.ai](https://www.ontocord.ai/). To download the dataset from Huggingface run the command below from the root of the repo.

```shell
python data/OIG/prepare.py
```

Once the command completes, the data will be in the `data/OIG/files` directory.

## Data Contributions

You can help make this chat model better by contributing data! See the [OpenDataHub](https://github.com/togethercomputer/OpenDataHub) repo for more details.

# Pretrained Base Model

As mentioned above, the chat model is a fine-tuned variant of GPT-NeoX-20B from Eleuther AI. To download GPT-NeoX-20B and prepare it for fine tuning, run this command from the root of the repo.

```shell
python pretrained/GPT-NeoX-20B/prepare.py
```

The weights for this model will be in the `pretrained/GPT-NeoX-20B/EleutherAI_gpt-neox-20b`.

# Training and Finetuning

## (Optional) 8bit Adam

To use 8bit-adam during training, install the `bitsandbytes` package.

```shell
pip install bitsandbytes # optional, to use 8bit-adam
```

## Train GPT-NeoX-Chat-Base-20B

The `training/finetune_GPT-NeoXT-Chat-Base-20B.sh` script configures and runs the training loop. After downloading the dataset and the base model, run:

```shell
bash training/finetune_GPT-NeoXT-Chat-Base-20B.sh
```

The script launches 8 processes with a pipeline-parallel degree of 8 and a data-parallel degree of 1.

As the training loop runs, checkpoints are saved to the `model_ckpts` directory at the root of the repo.

Please see [the training README](training/README.md) for more details about customizing the training run.

# Converting Weights to Huggingface Format

Before you can use this model to perform inference, it must be converted to the Hugginface format.

```shell
mkdir huggingface_models \
&& python tools/convert_to_hf_gptneox.py \
     --ckpt-path model_ckpts/GPT-Neo-XT-Chat-Base-20B/checkpoint_5 
     --save-path /huggingface_models/GPT-NeoXT-Chat-Base-20B 
     --n-stages 8 
     --n-layer-per-stage 6
```

# Inference

To help you test the model, we provide a simple test command line test harness to interact with the bot. 

```shell
python inference/bot.py
```

By default the script will load the model named GPT-NeoXT-Chat-Base-20B model under the `huggingface_models` directory, but you can override that behavior by specifying `--model`.

For example, if you want to load the base model from our Huggingface, repo, you can run the following command which downloads the weights from HuggingFace.

```shell
python inference/bot.py --model togethercomputer/GPT-NeoXT-Chat-Base-20B
```

Once the model has loaded, enter text at the prompt and the model will reply.

```shell
$ python inference/bot.py 
Loading /home/csris/src/github.com/togethercomputer/OpenChatKit/inference/../huggingface_models/GPT-NeoXT-Chat-Base-20B to cuda:1...
Welcome to OpenChatKit shell.   Type /help or /? to list commands.

>>> Hello.
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Hello human.

>>> 
```

Commands are prefixed with a `/`, and the `/quit` command exits.

# Monitoring

By default, the training script simply prints the loss as training proceeds, but it can also output metrics to a file using [loguru](https://github.com/Delgan/loguru) or report them to Weights & Biases.

## Loguru

Add the flag `--train-log-backend loguru` to your training script to log to `./logs/file_{time}.log`

## Weights & Biases

To use Weights & Biases, first login with your Weights & Biases token.

```shell
wandb login
```

And set `--train-log-backend wandb` in the training script to enable logging to Weights & Biases.

# Experimental: Retrieval-Augmented Models

*Note: Retrieval is still experimental.*

The code in `/retrieval` implements a python package for querying a Faiss index of Wikipedia. The following steps explain how to use this index to augment queries in the test harness with context from the retriever.

1. Donwload the Wikipedia index.

```shell
python data/wikipedia-3sentence-level-retrieval-index/prepare.py
```

2. Run the bot with the `--retrieval` flag.

```shell
python inference/bot.py --retrieval
```

After starting, the bot will load both the chat model and the retrieval index, which takes a long time. Once the model and the index are loaded, all queries will be augmented with extra context.


```shell
$ python inference/bot.py --retrieval
Loading /OpenChatKit/inference/../huggingface_models/GPT-NeoXT-Chat-Base-20B to cuda:0...
Loading retrieval index...
Welcome to OpenChatKit shell.   Type /help or /? to list commands.

>>> Where is Zurich?
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Where is Zurich?
Zurich is located in Switzerland.

>>>
```

# License

All code in this repository was developed by Together Computer except where otherwise noted.  Copyright (c) 2023, Together Computer.  All rights reserved. The code is licensed under the Apache 2.0 license.


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
  author = {Together Computer},
  url = {https://github.com/togethercomputer/OpenChatKit}
  month = {3},
  year = {2023},
  version = {0.15},
}
```

# Acknowledgements

Our model is a fine-tuned version of [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b), a large language model trained by [Eleuther AI](https://www.eleuther.ai). We evaluated our model on [HELM](https://crfm.stanford.edu/helm/latest/) provided by the [Center for Research on Foundation Models](https://crfm.stanford.edu). And we collaborated with both [CRFM](https://crfm.stanford.edu) and [HazyResearch](http://hazyresearch.stanford.edu) at Stanford to build this model.

We collaborated with [LAION](https://laion.ai/) and [Ontocord.ai](https://www.ontocord.ai/) to build the training data used to fine tune this model.
