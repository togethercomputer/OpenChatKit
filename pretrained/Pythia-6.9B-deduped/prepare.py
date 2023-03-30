import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='EleutherAI/pythia-6.9b-deduped', 
                        help='model-name')
    parser.add_argument('--save-dir', type=str, default=DIR, 
                        help='model-name')
    parser.add_argument('--offload-dir', type=str, default=None,
                        help='directory to offload from memory')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_path = os.path.join(args.save_dir, args.model_name.replace('/', '_'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    print('loading model from HF...')
    config = AutoConfig.from_pretrained(args.model_name)
    config.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(save_path)
    # offload model from memory to disk if offload-dir is specified
    if args.offload_dir is not None:
        if not os.path.exists(args.offload_dir):
            os.mkdir(args.offload_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto", offload_folder=args.offload_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
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
