import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='EleutherAI/gpt-neox-20b', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default=DIR, 
                        help='model-name')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name.replace('/', '_'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    
    item = {}
    item['embed_in.weight'] = model.gpt_neox.embed_in.weight
    torch.save(item, os.path.join(save_path, 'pytorch_embs.pt'))

    for i in range(len(model.gpt_neox.layers)):
        torch.save(model.gpt_neox.layers[i].state_dict(), os.path.join(save_path, f'pytorch_{i}.pt'))
    
    item = {}
    item['embed_out.weight'] = model.embed_out.weight
    item['final_layer_norm.weight'] = model.gpt_neox.final_layer_norm.weight
    item['final_layer_norm.bias'] = model.gpt_neox.final_layer_norm.bias
    torch.save(item, os.path.join(save_path, 'pytorch_lm_head.pt'))
