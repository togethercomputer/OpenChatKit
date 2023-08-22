import numpy as np
from torch import nn
from comm.comm_utils import *

from copy import deepcopy


class GPTStageBase(nn.Module):
    def __init__(self, args, config):
        super(GPTStageBase, self).__init__()
        self._to_cpu = (args.dist_backend == "gloo")
        self._embedding_dim = args.embedding_dim  # embedding dimension
        self._seq_length = args.seq_length
        # the dimension of the feedforward aws_network model in nn.TransformerEncoder
        self._feedforward_dim = args.embedding_dim * 4
        self._num_heads = args.num_heads  # the number of heads in the multi-head attention models
        self._num_layers = args.num_layers
        self._layer_begin = get_pipeline_parallel_rank() * args.num_layers
        self._layer_end = min(self._layer_begin + args.num_layers, args.max_layers)
        
        self._task_type = getattr(args, 'task_type', 'language_model')

        self.use_lora = 'lora' in args.model_type
        self.load_pretrained_model = args.load_pretrained_model
        self.model_name = args.model_name
        self.config = config
        
        if hasattr(args, 'model_type'):
            if args.model_type == "gpt2":
                from .hf_gpt2_modules import GPTEmbeddings, GPTBlock, GPTLMHead
            elif args.model_type == "gptj":
                from .hf_gptj_modules import GPTEmbeddings, GPTBlock, GPTLMHead
            elif args.model_type == "gptneox":
                from .hf_gptneox_modules import GPTEmbeddings, GPTBlock, GPTLMHead
            elif args.model_type == 'llama':
                from .llama_modules import GPTEmbeddings, GPTBlock, GPTLMHead
            elif args.model_type == 'llama_lora':
                from .llama_lora_modules import GPTEmbeddings, GPTBlock, GPTLMHead
            else:
                raise Exception("unknown")
        else:
            raise Exception("!!!! model type not defined")
            
        self._GPTEmbeddings = GPTEmbeddings
        self._GPTBlock = GPTBlock
        self._GPTLMHead = GPTLMHead

    def _create_first_layer(self):
        layer = self._GPTEmbeddings(deepcopy(self.config))
        if self.load_pretrained_model:
            print('loading embs')
            ret = layer.load_state_dict(
                torch.load(f'{self.model_name}/pytorch_embs.pt'), strict=False
            )
            if len(ret.missing_keys):
                print('The following weight keys are missing:')
                print(ret.missing_keys)
            if len(ret.unexpected_keys):
                print('The following weight keys are unexpected:')
                print(ret.unexpected_keys)
        if self.use_lora:
            for n, p in layer.named_parameters():
                if 'lora' in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        return layer

    def _create_last_layer(self):
        layer = self._GPTLMHead(deepcopy(self.config))
        if self.load_pretrained_model:
            print('loading lm_head')
            ret = layer.load_state_dict(
                torch.load(f'{self.model_name}/pytorch_lm_head.pt'), strict=False
            )
            if len(ret.missing_keys):
                print('The following weight keys are missing:')
                print(ret.missing_keys)
            if len(ret.unexpected_keys):
                print('The following weight keys are unexpected:')
                print(ret.unexpected_keys)
        if self.use_lora:
            for n, p in layer.named_parameters():
                if 'lora' in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        return layer

    def _create_transformer_layer(self, layer_idx=0):
        config = deepcopy(self.config)
        layer = self._GPTBlock(config, layer_id=layer_idx) # TODO: checkpoint
        if self.load_pretrained_model:
            print(f'loading layer {layer_idx}')
            ret = layer.load_state_dict(
                torch.load(f'{self.model_name}/pytorch_{layer_idx}.pt'), strict=False
            )
            if len(ret.missing_keys):
                print('The following weight keys are missing:')
                print(ret.missing_keys)
            if len(ret.unexpected_keys):
                print('The following weight keys are unexpected:')
                print(ret.unexpected_keys)
        if self.use_lora:
            for n, p in layer.named_parameters():
                if 'lora' in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        return layer
    

class GPTStageFull(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageFull, self).__init__(args, config)
        self.device = device
        module_list = [self._create_first_layer()]
        for layer_idx in range(self._layer_begin, self._layer_end):
            module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
        if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
            pass
        else:
            module_list.append(self._create_last_layer())
        self.model = nn.Sequential(*module_list)

    def forward(self, x, **kargs):
        for module in self.model:
            x = module(x, **kargs)
        return x


class GPTStageFirst(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageFirst, self).__init__(args, config)
        self.device = device
        module_list = [self._create_first_layer()]
        for layer_idx in range(self._layer_begin, self._layer_end):
            module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
        self.model = nn.Sequential(*module_list)

    def forward(self, x, **kargs):
        for module in self.model:
            x = module(x, **kargs)
        return x
        # out = self.model(x.to(self.device), **kargs)
        # return out.cpu() if self._to_cpu else out


class GPTStageMiddle(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageMiddle, self).__init__(args, config)
        self.device = device
        module_list = []
        for layer_idx in range(self._layer_begin, self._layer_end):
            module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
        self.model = nn.Sequential(*module_list)

    def forward(self, x, **kargs):
        for module in self.model:
            x = module(x, **kargs)
        return x
        # out = self.model(x.to(self.device), **kargs) if self._to_cpu else self.model(x)
        # return out.cpu() if self._to_cpu else out


class GPTStageLast(GPTStageBase):
    def __init__(self, args, config, device):
        super(GPTStageLast, self).__init__(args, config)
        self.device = device
        module_list = []
        for layer_idx in range(self._layer_begin, self._layer_end):
            module_list.append(self._create_transformer_layer(layer_idx=layer_idx))
            
        if hasattr(args, 'skip_lm_head') and args.skip_lm_head:
            pass
        else:
            module_list.append(self._create_last_layer())
        
        self.model = nn.Sequential(*module_list)
        
        # self.upscale_last = nn.Linear(args.embedding_dim, 9216).to(device)
        
    def forward(self, x, **kargs):
        for module in self.model:
            x = module(x, **kargs)
        
        return x

#     def forward(self, x, **kargs):
#         for module in self.model[:-1]:
#             x = module(x, **kargs)
#         hid = x
#         x = self.model[-1](x, **kargs)
        
#         hid = self.upscale_last(hid)
#         loss = torch.nn.functional.mse_loss(hid, kargs['teacher_hidden_states'])
#         print(loss.item())
#         return x, loss
    