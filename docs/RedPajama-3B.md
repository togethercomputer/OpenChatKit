# Fine Tuning RedPajama-INCITE-Base-3B

In order to fine-tune the Fine Tuning RedPajama-INCITE-Base-3B model, please follow these steps:

First clone the OpenChatKit repo:

```shell
git clone git@github.com:togethercomputer/OpenChatKit.git
```

Next install dependencies as instructed by the OpenChatKit repo.

# Prepare Weights

```shell
python pretrained/RedPajama-3B/prepare.py
```

This script will download the weight from HuggingFace and prepare it for finetuning. The prepared weights will be saved at 

```
pretrained/RedPajama-3B/togethercomputer_RedPajama-INCITE-Chat-3B-v1
```

# Prepare Fine Tuning Data

We now need to prepare the training data.  We provide an example script that downloads a small slice of data from OIG. 
To download this sample dataset, please run:
 
```
bash data/OIG-chip2/prepare.sh
````
 
The sample dataset will be saved at 

```
data/OIG-chip2/unified_chip2.jsonl.
```

# Run Fine Tuning Script

We provide an example training script.  Please configure the parameters (e.g., learning_rate, batch_size, dataset_path) according to your hardware configuration. 
Then to start training, simply run

```
bash training/finetune_RedPajama-INCITE-Chat-3B-v1.sh
```

# Convert to Huggingface Format

Convert to HF format. The fine-tuned model will be saved to 

```
model_ckpts/rp-incite-chat-3b-finetuned/checkpoint_{steps}
```

In order to use it for inference, you will need to convert it to the HuggingFace format. To do so, run the following script 
(as an example, please change the checkpoint path, n-stages and n-layer-per-stage according to the training script):

```
python tools/convert_to_hf_gptneox.py --config-name togethercomputer/RedPajama-INCITE-Chat-3B-v1 --ckpt-path model_ckpts/rp-incite-chat-3b-fintuned/checkpoint_100/ --save-path model_ckpts/hf --n-stages 4 --n-layer-per-stage 8
```

Then you are ready to go.
