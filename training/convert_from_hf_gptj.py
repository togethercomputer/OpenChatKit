import os
import argparse
import torch
from transformers import GPTJModel, GPTJForCausalLM, AutoTokenizer, GPTJConfig
# from modules.gpt_modules import GPTEmbeddings, GPTBlock

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='EleutherAI/gpt-j-6B', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default='./pretrained_models', 
                        help='model-name')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name.replace('/', '_'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    config = GPTJConfig.from_pretrained(args.model_name)
    config.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_path)
    model = GPTJForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)

    torch.save({
        'wte.weight': model.transformer.wte.state_dict()['weight'],
    }, os.path.join(save_path, 'pytorch_embs.pt'))

    for i in range(len(model.transformer.h)):
        # make sure causal mask is back
        max_positions = 2048
        model.transformer.h[i].attn.bias[:] = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
            1, 1, max_positions, max_positions
        )
        torch.save(model.transformer.h[i].state_dict(), os.path.join(save_path, f'pytorch_{i}.pt'))
        
    torch.save({
        'ln_f.weight': model.transformer.ln_f.state_dict()['weight'],
        'ln_f.bias': model.transformer.ln_f.state_dict()['bias'],
        'lm_head.weight': model.lm_head.state_dict()['weight'],
        'lm_head.bias': model.lm_head.state_dict()['bias'],
    }, os.path.join(save_path, 'pytorch_lm_head.pt'))
