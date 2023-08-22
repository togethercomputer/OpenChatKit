import os
import argparse
import torch

import torch
import torch.nn as nn

from transformers import LlamaForCausalLM
from transformers import AutoConfig, AutoTokenizer

from transformers.modeling_utils import no_init_weights
import os


def create_emtpy_llama(config):

    import torch
    import torch.nn as nn

    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kargs):
        pass
    nn.Linear.reset_parameters = dummy

    # 1. disable init for faster initialization
    # 2. avoid tie token embeddings with lm_head, as we train them separately.
    with no_init_weights(_enable=True):
        model = LlamaForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear

    return model

def merge_lora_weights(_tmp):
    to_pop = []
    for k in _tmp:
        if 'lora_qv_proj.lora_A' in k:
            print('merging lora weights...')
            src_A_k = k
            src_B_k = k.replace('lora_A', 'lora_B')
            tgt_Q_k = k.replace('lora_qv_proj.lora_A', 'q_proj')
            tgt_V_k = k.replace('lora_qv_proj.lora_A', 'v_proj')

            src_B_Q, src_B_V = _tmp[src_B_k].chunk(2, dim=0)
            lora_Q = src_B_Q.float().matmul(_tmp[src_A_k].float())
            lora_V = src_B_V.float().matmul(_tmp[src_A_k].float())
                        
            _tmp[tgt_Q_k].data += lora_Q
            _tmp[tgt_V_k].data += lora_V

            to_pop.append(src_A_k)
            to_pop.append(src_B_k)
    for k in to_pop:
        _tmp.pop(k)
    # do nothing if there is no lora weight.
    return _tmp
    
def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=16, ):
    input_path = checkpoint_path

    n_layers = len(model.model.layers)
    assert n_stages * n_layer_per_stage >= len(model.model.layers)
    # assert model.lm_head.weight.data is not model.transformer.wte.weight.data

    for i in range(n_stages):

        print(f'loading stage {i}')

        checkpoint = torch.load(os.path.join(input_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))

        if i == 0:
            _tmp = {k[len(f"{0}."):]:v for k,v in checkpoint.items() if k.startswith(f"0.")}
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_embs.pt'))
            model.model.embed_tokens.weight.data[:] = _tmp['embed_tokens.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                _tmp = merge_lora_weights(_tmp)
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{j}.pt'))
                ret = model.model.layers[j].load_state_dict(_tmp, strict=False)
                if len(ret.missing_keys):
                    print('The following weight keys are missing:')
                    print(ret.missing_keys)
                if len(ret.unexpected_keys):
                    print('The following weight keys are unexpected:')
                    print(ret.unexpected_keys)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                if i*n_layer_per_stage + j == n_layers:
                    break
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                _tmp = merge_lora_weights(_tmp)
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                ret = model.model.layers[i*n_layer_per_stage + j].load_state_dict(_tmp, strict=False)
                if len(ret.missing_keys):
                    print('The following weight keys are missing:')
                    print(ret.missing_keys)
                if len(ret.unexpected_keys):
                    print('The following weight keys are unexpected:')
                    print(ret.unexpected_keys)
            else:
                j += 1

            _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
            if len(_tmp) == 0:
                break
            _tmp = merge_lora_weights(_tmp)
            # torch.save(_tmp, os.path.join(output_path, f'pytorch_lm_head.pt'))
            model.model.norm.weight.data[:] = _tmp['norm.weight']
            if 'norm.bias' in _tmp:
                model.model.norm.bias.data[:] = _tmp['norm.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']
            if 'lm_head.bias' in _tmp:
                model.lm_head.bias.data[:] = _tmp['lm_head.bias']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]:v for k,v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                # torch.save(_tmp, os.path.join(output_path, f'pytorch_{i*n_layer_per_stage + j}.pt'))
                ret = model.model.layers[i*n_layer_per_stage + j].load_state_dict(_tmp, strict=False)
                if len(ret.missing_keys):
                    print('The following weight keys are missing:')
                    print(ret.missing_keys)
                if len(ret.unexpected_keys):
                    print('The following weight keys are unexpected:')
                    print(ret.unexpected_keys)

    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--config-name', type=str, default='togethercomputer/Llama-2-7B-32K-beta',
                        help='config-name')
    parser.add_argument('--ckpt-path', type=str, default=None, 
                        help='ckpt-path')
    parser.add_argument('--save-path', type=str, default=None, 
                        help='save-path')
    parser.add_argument('--n-stages', type=int, default=8, 
                        help='pipeline group size')
    parser.add_argument('--n-layer-per-stage', type=int, default=4, 
                        help='n layers per GPU device')
    parser.add_argument('--fp16', default=False, action='store_true')
    args = parser.parse_args()
    
    assert args.ckpt_path is not None
    assert args.save_path is not None
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # LlamaForCausalLM LlamaConfig LlamaTokenizer
    print('loading config...')
    config = AutoConfig.from_pretrained(args.config_name)
    print('loaded config.')
    print('loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    print('loaded tokenizer.')
    print('creating empty model...')
    model = create_emtpy_llama(config)
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
