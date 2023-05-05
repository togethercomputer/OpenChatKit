# RedPajama-Chat-INCITE-2.8B-v1

>> add some description.

In this doc, you'll find steps for:
- Training an RedPajama-Chat-INCITE-2.8B-v1
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
- [Acknowledgements](#acknowledgements)

# Requirements

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

```shell
mamba env create -f environment.yml 
```

6. Activate the new conda environment.

```shell
conda activate OpenChatKit
```

# Pre-trained Weights

GPT-NeoXT-Chat-Base-20B is a 20B-parameter variant of GPT-NeoX, fine-tuned on conversational datasets. We are releasing pre-trained weights for this model as [togethercomputer/GPT-NeoXT-Chat-Base-20B](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B) on Huggingface.

More details can be found on the model card for [GPT-NeoXT-Chat-Base-20B](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B) on Huggingface.

# Datasets

>> do we want to mention oasst+dolly 2 dataset?

## Data Contributions

You can help make this chat model better by contributing data! See the [OpenDataHub](https://github.com/togethercomputer/OpenDataHub) repo for more details.

# Pretrained Base Model

As mentioned above, the chat model is a fine-tuned variant of GPT-NeoX-20B from Eleuther AI. To download GPT-NeoX-20B and prepare it for fine tuning, run this command from the root of the repo.

```shell
python pretrained/RedPajama-3B/prepare.py
```

The weights for this model will be in the `pretrained/RedPajama-3B/togethercomputer_RedPajama-Base-INCITE-2.8B-v1`.

# Training and Finetuning

## (Optional) 8bit Adam

To use 8bit-adam during training, install the `bitsandbytes` package.

```shell
pip install bitsandbytes # optional, to use 8bit-adam
```

## Train GPT-NeoX-Chat-Base-20B

The `training/finetune_RedPajama-3B.sh` script configures and runs the training loop. After downloading the dataset and the base model, run:

```shell
bash training/finetune_RedPajama-3B.sh
```

(Don't forget to replace YOUR_DATASET_PATH to the actual path in the script!)

The script launches 8 processes with a pipeline-parallel degree of 4 and a data-parallel degree of 2.

As the training loop runs, checkpoints are saved to the `model_ckpts` directory at the root of the repo.

Please see [the training README](training/README.md) for more details about customizing the training run.

# Converting Weights to Huggingface Format

Before you can use this model to perform inference, it must be converted to the Huggingface format. Run this command from the root of the repo to do so.

```shell
mkdir huggingface_models \
  && python tools/convert_to_hf_gptneox.py \
       --ckpt-path model_ckpts/RedPajama-3B-chat/checkpoint_100  \
       --save-path huggingface_models/GPT-NeoXT-Chat-Base-20B  \
       --n-stages 8  \
       --n-layer-per-stage 6 \
       --fp16
```
where the `--fp16` flag will load and store models in fp16.

Make sure to replace `model_ckpts/RedPajama-3B-chat/checkpoint_100` with the latest checkpoint in the `model_ckpts/RedPajama-3B-chat` directory.


# Inference

To help you test the model, we provide a simple test command line test harness to interact with the bot. 

```shell
python inference/bot.py
```

By default the script will load the model named GPT-NeoXT-Chat-Base-20B model under the `huggingface_models` directory, but you can override that behavior by specifying `--model`.

For example, if you want to load the base model from our Huggingface, repo, you can run the following command which downloads the weights from HuggingFace.

```shell
python inference/bot.py --model togethercomputer/RedPajama-Chat-INCITE-2.8B-v1
```

Once the model has loaded, enter text at the prompt and the model will reply.

```shell
$ python inference/bot.py 
Loading /home/csris/src/github.com/togethercomputer/OpenChatKit/inference/../huggingface_models/RedPajama-Chat-INCITE-2.8B-v1 to cuda:1...
Welcome to OpenChatKit shell.   Type /help or /? to list commands.

>>> Hello.
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Hello human.

>>> 
```

Commands are prefixed with a `/`, and the `/quit` command exits.

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

*Note: Retrieval is still experimental.*

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
Loading /OpenChatKit/inference/../huggingface_models/RedPajama-Chat-INCITE-2.8B-v1 to cuda:0...
Loading retrieval index...
Welcome to OpenChatKit shell.   Type /help or /? to list commands.

>>> Where is Zurich?
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Where is Zurich?
Zurich is located in Switzerland.

>>>
```

# Acknowledgements

>> todo...