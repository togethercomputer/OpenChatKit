import torch
import torch.nn as nn

import argparse

from transformers import GPTJForCausalLM

from transformers import AutoConfig, AutoTokenizer

from transformers.modeling_utils import no_init_weights
import os


def create_emtpy_gptj(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = GPTJForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=14):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.transformer.h)
    assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.transformer.wte.weight.data[:] = _tmp['wte.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.transformer.h[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if 'lm_head.weight' in _tmp:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.transformer.h[i*n_layer_per_stage + j].load_state_dict(_tmp)
            else:
                _tmp = {k[len(f"{n_layer_per_stage}."):]:v for k,v in checkpoint.items() if k.startswith(f"{n_layer_per_stage}.")}
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.transformer.ln_f.weight.data[:] = _tmp['ln_f.weight']
            model.transformer.ln_f.bias.data[:] = _tmp['ln_f.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.transformer.h[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--ckpt-path', type=str, default=None, 
                        help='model-name')
    parser.add_argument('--save-path', type=str, default=None, 
                        help='model-name')
    parser.add_argument('--n-stages', type=int, default=2, 
                        help='pipeline group size')
    parser.add_argument('--n-layer-per-stage', type=int, default=14, 
                        help='n layers per GPU device')
    args = parser.parse_args()
    
    assert args.ckpt_path is not None
    assert args.save_path is not None
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_path)

    config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    model = create_emtpy_gptj(config)
    load_decentralized_checkpoint(
        model, args.ckpt_path, n_stages=args.n_stages, n_layer_per_stage=args.n_layer_per_stage,
    )
    
    model.save_pretrained(args.save_path)
    config.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)