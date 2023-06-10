import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

peft_model_path = 'outputs/redpajama-incite-chat-3b-sample-lowrank'

config = PeftConfig.from_pretrained(peft_model_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    device_map='auto')

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_path)

model = model.merge_and_unload()

model.save_pretrained('outputs/redpajama-incite-chat-3b-sample-lowrank-merged')
