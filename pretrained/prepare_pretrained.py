import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DIR = os.path.dirname(os.path.abspath(__file__))
USE_AUTH_TOKEN = False

# Load pretrained model from HuggingFace and save it to disk
def prepare_pretrained(save_path, model_name, offload_dir=None):
    os.makedirs(save_path, exist_ok=True)
    
    print('loading model from HF...')
    config = AutoConfig.from_pretrained(model_name, use_auth_token=USE_AUTH_TOKEN)
    config.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=USE_AUTH_TOKEN)
    tokenizer.save_pretrained(save_path)

    # offload model from memory to disk if offload-dir is specified
    if offload_dir is not None:
        os.makedirs(offload_dir, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     torch_dtype=torch.float16,
                                                     device_map="auto",
                                                     offload_folder=offload_dir,
                                                     use_auth_token=USE_AUTH_TOKEN)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.float16,
                                                     use_auth_token=USE_AUTH_TOKEN)
    print('loaded model from HF...')
    
    print('converting the embedding layer...')
    item = {}
    item['embed_in.weight'] = model.gpt_neox.embed_in.weight
    torch.save(item, os.path.join(save_path, 'pytorch_embs.pt'))
    print('converted the embedding layer.')

    for i in range(len(model.gpt_neox.layers)):
        print(f'converting the {i}-th transformer layer...')
        torch.save(model.gpt_neox.layers[i].state_dict(), os.path.join(save_path, f'pytorch_{i}.pt'))
        print(f'converted the {i}-th transformer layer.')

    print('converting the lm_head layer...')
    item = {}
    item['embed_out.weight'] = model.embed_out.weight
    item['final_layer_norm.weight'] = model.gpt_neox.final_layer_norm.weight
    item['final_layer_norm.bias'] = model.gpt_neox.final_layer_norm.bias
    torch.save(item, os.path.join(save_path, 'pytorch_lm_head.pt'))
    print('converted the lm_head layer.')

# python pretrained/prepare_pretrained.py --model-name EleutherAI/gpt-neox-125M --save-dir pretrained/files --offload-dir pretrained/files/offload
def main():
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, required=True, 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, required=True, 
                        help='model-name')
    parser.add_argument('--offload-dir', type=str, default=None,
                        help='directory to offload from memory')
    args = parser.parse_args()
    
    prepare_pretrained(args.save_dir, args.model_name, args.offload_dir)

if __name__ == '__main__':
    main()