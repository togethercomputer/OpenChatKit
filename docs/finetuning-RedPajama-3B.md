# RedPajama-3B

In this tutorial, you will learn how to fine-tune a base LLM on a sample of data. By the end of 
the tutorial, you will have fine-tuned the RedPajama-INCITE-Chat-3B model using a sample of 
chat data from the OIG dataset. You can adapt this tutorial to fine-tune with your own data.

In order to fine-tune the RedPajama 3B models, please follow these steps:

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

We now need to preapre the training data.  We provide an example script that downloads a small slice of data from OIG. 
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

The fine-tuned model will be saved to 

```
model_ckpts/rp-incite-chat-3b-finetuned/checkpoint_{steps}
```

In order to use it for inference, you will need to convert it to the HuggingFace format. To do so, run the following script 
(as an example, please change the checkpoint path, n-stages and n-layer-per-stage according to the training script):

The default for n-stages used in the training script is 10 and the n-layer-per-stage is 8.

```
python tools/convert_to_hf_gptneox.py --config-name togethercomputer/RedPajama-INCITE-Chat-3B-v1 --ckpt-path model_ckpts/redpajama-incite-chat-3b-sample/checkpoint_10/ --save-path model_ckpts/hf --n-stages 4 --n-layer-per-stage 8
```

Then you are ready to go! You can load the model with HuggingFace and use it for inference, for example:

```python
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("./model_ckpts/hf", torch_dtype=torch.float16)
model = model.to('cuda:0')

prompt = "<human>: Who is Alan Turing?\n<bot>:"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
input_length = inputs.input_ids.shape[1]
outputs = model.generate(
    **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
)
token = outputs.sequences[0, input_length:]
output_str = tokenizer.decode(token)
print(output_str)

```
