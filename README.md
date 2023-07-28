# OpenChatKit

OpenChatKit provides a powerful, open-source base to create both specialized and general purpose models for various applications. The kit includes an instruction-tuned language models, a moderation model, and an extensible retrieval system for including up-to-date responses from custom repositories. OpenChatKit models were trained on the OIG-43M training dataset, which was a collaboration between [Together](https://www.together.xyz/), [LAION](https://laion.ai), and [Ontocord.ai](https://ontocord.ai). 

In this repo, you'll find code for:
- Training GPT-NeoXT-Chat-Base-20B, a 20B parameter chat model (see [docs/GPT-NeoXT-Chat-Base-20B.md](docs/GPT-NeoXT-Chat-Base-20B.md))
- Fine-tuning Llama-2-7B-32K-beta, a 7B parameter long context model
- Training Pythia-Chat-Base-7B, a 7B parameter chat model
- Testing inference using either of the chat models
- Augmenting the model with additional context from a retrieval index

# Contents

- [Getting Started](#getting-started)
  * [Requirements](#requirements)
  * [Chatting with Pythia-Chat-Base-7B](#chatting-with-pythia-chat-base-7b)
- [Fine-tuning Llama-2-7B-32K-beta](#fine-tuning-llama-2-7b-32k-beta)
  * [Downloading and converting the base model](#downloading-and-converting-the-base-model)
  * [Fine-tuning the model](#fine-tuning-the-model)
  * [Converting trained weights to Hugging Face format](#converting-trained-weights-to-hugging-face-format)
- [Reproducing Pythia-Chat-Base-7B](#reproducing-pythia-chat-base-7b)
  * [Downloading training data and the base model](#downloading-training-data-and-the-base-model)
  * [(Optional) 8bit Adam](#optional-8bit-adam)
  * [Training the model](#training-the-model)
  * [Converting weights to Hugging Face format](#converting-weights-to-hugging-face-format)
  * [Testing the new model](#testing-the-new-model)
- [Monitoring](#monitoring)
  * [Loguru](#loguru)
  * [Weights & Biases](#weights--biases)
- [Experimental: Retrieval-Augmented Models](#experimental-retrieval-augmented-models)
- [See Also](#see-also)
- [License](#license)
- [Citing OpenChatKit](#citing-openchatkit)
- [Acknowledgements](#acknowledgements)

# Getting Started

In this tutorial, you will download Pythia-Chat-Base-7B, an instruction-tuned language model, and run some some inference requests against it using a command-line tool.

Pythia-Chat-Base-7B is a 7B-parameter fine-tuned variant of Pythia-6.9B-deduped from Eleuther AI. Pre-trained weights for this model are available on Hugging Face as [togethercomputer/Pythia-Chat-Base-7B](https://huggingface.co/togethercomputer/Pythia-Chat-Base-7B) under an Apache 2.0 license.

More details can be found on the model card for [Pythia-Chat-Base-7B](https://huggingface.co/togethercomputer/Pythia-Chat-Base-7B) on Hugging Face.

## Requirements

Before you begin, you need to install PyTorch and other dependencies.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) from their website.

2. Install [Git LFS](https://git-lfs.com/) from their website.

3. Install the `git lfs` hooks.

```shell
git lfs install
```

4. Install mamba in the `base` environment so it's available in all environments.

```shell
conda install mamba -n base -c conda-forge
```

5. Create an environment called OpenChatKit using the `environment.yml` file at the root of this repo.

> **Note**
> Use `mamba` to create the environment. It's **much** faster than using `conda`.

```shell
mamba env create -f environment.yml 
```

6. Activate the new conda environment.

```shell
conda activate OpenChatKit
```

## Chatting with Pythia-Chat-Base-7B

To help you try the model, [`inference/bot.py`](inference/bot.py) is a simple command-line test harness that provides a shell inferface enabling you to chat with the model. Simply enter text at the prompt and the model replies. The test harness also maintains conversation history to provide the model with context.


Start the bot by calling `bot.py` from the root for the repo.

```shell
python inference/bot.py --model togethercomputer/Pythia-Chat-Base-7B
```

Loading the model can take some time, but once it's loaded, you are greeted with a prompt. Say hello.

```shell
$ python inference/bot.py 
Loading /home/csris/src/github.com/togethercomputer/OpenChatKit/inference/../huggingface_models/GPT-NeoXT-Chat-Base-20B to cuda:1...
Welcome to OpenChatKit shell.   Type /help or /? to list commands.

>>> Hello.
Hello human.

>>> 
```

Enter additional queries at the prompt, and the model replies. Under the covers, the shell is forming a prompt with all previous queries and passes that to the model to generate more text.

The shell also supports additional commands to inspect hyperparamters, the full prompt, and more. Commands are prefixed with a `/`.

> **Note**
> The `/quit` command exits the shell.

Please see [the inference README](inference/README.md) for more details about arguments, running on multiple/specific GPUs, and running on consumer hardware.

# Fine-tuning Llama-2-7B-32K-beta

Llama-2-7B-32K-beta model can be fine-tuned using various datasets. In this tutorial, we will use the multi-document natural questions dataset and BookSum dataset.

## Downloading and converting the base model

To download model Llama-2-7B-32K-beta and prepare it for fine-tuning, run this command from the root of the repository.

```shell
python pretrained/Llama-2-7B-32K-beta/prepare.py
```

The weights for this model will be in the `pretrained/Llama-2-7B-32K-beta/togethercomputer_Llama-2-7B-32K-beta` directory.


## Fine-tuning the model

The `training/finetune_llama-2-7b-32k-mqa.sh` and `training/finetune_llama-2-7b-32k-booksum.sh` scripts configure and run the training loop.

1. To fine-tune the multi-document natural questions dataset, run:
   ```shell
   bash training/finetune_llama-2-7b-32k-mqa.sh
   ```

2. To fine-tune the BookSum dataset, run:
   ```shell
   bash training/finetune_llama-2-7b-32k-booksum.sh
   ```

As the training loop runs, checkpoints are saved to the `model_ckpts` directory at the root of the repo.

Please see [the training README](training/README.md) for more details about customizing the training run.

## Converting trained weights to Hugging Face format

Before you can use this model to perform inference, it must be converted to the Hugging Face format. Run this command from the root of the repo to do so.

For example
```shell
mkdir huggingface_models \
  && python tools/convert_to_hf_llama.py \
       --config-name togethercomputer/Llama-2-7B-32K-beta \
       --ckpt-path model_ckpts/llama-2-7b-32k-mqa/checkpoint_10 \
       --save-path huggingface_models/llama-2-7b-32k-mqa \
       --n-stages 4 \
       --n-layer-per-stage 8 \
       --fp16
```
where the `--fp16` flag will load and store models in fp16.

Make sure to replace model_ckpts/llama-2-7b-32k-mqa/checkpoint_10` with the latest checkpoint in the `model_ckpts/llama-2-7b-32k-mqa` or `model_ckpts/llama-2-7b-32k-booksum` directory.


# Reproducing Pythia-Chat-Base-7B

This tutorial walks through reproducing the Pythia-Chat-Base-7B model by fine-tuning Eleuther AI's Pythia-6.9B-deduped model using the OIG dataset.

## Downloading training data and the base model

The chat model was trained on the [OIG](https://huggingface.co/datasets/laion/OIG) dataset built by [LAION](https://laion.ai/), [Together](https://www.together.xyz/), and [Ontocord.ai](https://www.ontocord.ai/). To download the dataset from Hugging Face run the command below from the root of the repo.

```shell
python data/OIG/prepare.py
```
> **Note** 
> You can help make this chat model better by contributing data! See the [OpenDataHub](https://github.com/togethercomputer/OpenDataHub) repo for more details.

Once the command completes, the data will be in the `data/OIG/files` directory.

Pythia-Chat-Base-7B is a fine-tuned variant of Pythia-6.9B-deduped from Eleuther AI. To download the model and prepare it for fine tuning, run this command from the root of the repo.

```shell
python pretrained/Pythia-6.9B-deduped/prepare.py
```

The weights for this model will be in the `pretrained/Pythia-6.9B-deduped/EleutherAI_pythia-6.9b-deduped` directory.

## (Optional) 8bit Adam

To use 8bit-adam during training, install the `bitsandbytes` package.

```shell
pip install bitsandbytes # optional, to use 8bit-adam
```

## Training the model

The `training/finetune_Pythia-Chat-Base-7B.sh` script configures and runs the training loop. After downloading the dataset and the base model, run:

```shell
bash training/finetune_Pythia-Chat-Base-7B.sh
```

As the training loop runs, checkpoints are saved to the `model_ckpts` directory at the root of the repo.

Please see [the training README](training/README.md) for more details about customizing the training run.

## Converting weights to Hugging Face format

Before you can use this model to perform inference, it must be converted to the Hugging Face format. Run this command from the root of the repo to do so.

```shell
mkdir huggingface_models \
  && python tools/convert_to_hf_gptneox.py \
       --config-name EleutherAI/pythia-6.9b-deduped \
       --ckpt-path model_ckpts/Pythia-Chat-Base-7B/checkpoint_100 \
       --save-path huggingface_models/Pythia-Chat-Base-7B \
       --n-stages 4 \
       --n-layer-per-stage 8 \
       --fp16
```
where the `--fp16` flag will load and store models in fp16.

Make sure to replace `model_ckpts/Pythia-Chat-Base-7B/checkpoint_100` with the latest checkpoint in the `model_ckpts/Pythia-Chat-Base-7B` directory.

## Testing the new model

You can use the OpenChatKit Shell test harness to chat with the new model. From the root of the repo, run

```shell
python inference/bot.py
```

By default the script will load the model named Pythia-Chat-Base-7B under the `huggingface_models` directory, but you can override that behavior by specifying `--model`.

```shell
python inference/bot.py --model ./huggingface_models/GPT-NeoXT-Chat-Base-20B
```

Once the model has loaded, enter text at the prompt and the model will reply.

```shell
$ python inference/bot.py 
Loading /home/csris/src/github.com/togethercomputer/OpenChatKit/inference/../huggingface_models/GPT-NeoXT-Chat-Base-20B to cuda:1...
Welcome to OpenChatKit shell.   Type /help or /? to list commands.

>>> Hello.
Hello human.

>>> 
```

The shell also supports additional commands to inspect hyperparamters, the full prompt, and more. Commands are prefixed with a `/`.

> **Note**
> The `/quit` command exits the shell.

Please see [the inference README](inference/README.md) for more details about arguments, running on multiple/specific GPUs, and running on consumer hardware.

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

> **Warning**
> Retrieval support is experimental.

The code in `/retrieval` implements a python package for querying a Faiss index of Wikipedia. The following steps explain how to use this index to augment queries in the test harness with context from the retriever.

1. Download the Wikipedia index.

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
Where is Zurich?
Zurich is located in Switzerland.

>>>
```

# See Also
* [docs/GPT-NeoXT-Chat-Base-20B.md](docs/GPT-NeoXT-Chat-Base-20B.md). OpenChatKit also provides a larger, 20B parameter chat model that was trained on GPT-NeoXT-Chat-Base-20B from Eleuther AI.

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

Our models are fine-tuned versions of large language models trained by [Eleuther AI](https://www.eleuther.ai). We evaluated our model on [HELM](https://crfm.stanford.edu/helm/latest/) provided by the [Center for Research on Foundation Models](https://crfm.stanford.edu). And we collaborated with both [CRFM](https://crfm.stanford.edu) and [HazyResearch](http://hazyresearch.stanford.edu) at Stanford to build this model.

We collaborated with [LAION](https://laion.ai/) and [Ontocord.ai](https://www.ontocord.ai/) to build the training data used to fine tune this model.
