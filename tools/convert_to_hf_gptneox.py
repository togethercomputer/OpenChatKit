import torch
import torch.nn as nn

import argparse

from transformers import GPTNeoXForCausalLM

from transformers import AutoConfig, AutoTokenizer

from transformers.modeling_utils import no_init_weights
import os


def create_empty_gptneox(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = GPTNeoXForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=14):
    input_path = checkpoint_path

    assert n_stages * n_layer_per_stage >= len(model.gpt_neox.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.gpt_neox.embed_in.weight.data[:] = _tmp['embed_in.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                model.gpt_neox.layers[j].load_state_dict(_tmp)

        if i != 0 and i == n_stages - 1 or n_stages == 1:
            if n_stages != 1:
                for j in range(n_layer_per_stage):
                    _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                    if len(_tmp) == 0:
                        break
                    # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                    model.gpt_neox.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)
                    if i*n_layer_per_stage + j == len(model.gpt_neox.layers) - 1:
                        j += 1
                        break
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
            else:
                _tmp = {k[len(f"{n_layer_per_stage+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{n_layer_per_stage+1}.")} #Added line
            if len(_tmp) == 0:
                break
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.gpt_neox.final_layer_norm.weight.data[:] = _tmp['final_layer_norm.weight']
            model.gpt_neox.final_layer_norm.bias.data[:] = _tmp['final_layer_norm.bias']
            model.embed_out.weight.data[:] = _tmp['embed_out.weight']
            if 'embed_out.bias' in _tmp:
                model.embed_out.bias.data[:] = _tmp['embed_out.bias']

        elif i != 0:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                model.gpt_neox.layers[i*n_layer_per_stage + j].load_state_dict(_tmp)

    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--config-name', type=str, default='EleutherAI/gpt-neox-20b',
                        help='config-name')
    parser.add_argument('--ckpt-path', type=str, default=None, 
                        help='ckpt-path')
    parser.add_argument('--save-path', type=str, default=None, 
                        help='save-path')
    parser.add_argument('--n-stages', type=int, default=8, 
                        help='pipeline group size')
    parser.add_argument('--n-layer-per-stage', type=int, default=6, 
                        help='n layers per GPU device')
    parser.add_argument('--fp16', default=False, action='store_true')
    args = parser.parse_args()
    
    assert args.ckpt_path is not None
    assert args.save_path is not None
    
    os.makedirs(args.save_path, exist_ok=True)

    print('loading config...')
    config = AutoConfig.from_pretrained(args.config_name)
    print('loaded config.')
    print('loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    print('loaded tokenizer.')
    print('creating empty model...')
    model = create_empty_gptneox(config)
    if args.fp16:
        model = model.half()
    print('created empty model.')
    print('loading model ckpt...')
    load_decentralized_checkpoint(
        model, args.ckpt_path, n_stages=args.n_stages, n_layer_per_stage=args.n_layer_per_stage,
    )
    print('loaded model ckpt.')
    
    print('saving HF model...')
    model.save_pretrained(args.save_path)
    print(f'saved HF model to `{args.save_path}`')
    config.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
