import os
import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.gptj.modeling_gptj import ACT2FN
from transformers.models.gptj.modeling_gptj import GPTJAttention as _GPTJAttention
from transformers.models.gptj.modeling_gptj import GPTJMLP as _GPTJMLP
from transformers.models.gptj.modeling_gptj import GPTJBlock as _GPTJBlock
from transformers.models.gptj.modeling_gptj import GPTJModel as _GPTJModel
from transformers.models.gptj.modeling_gptj import fixed_pos_embedding
from transformers.models.gptj.configuration_gptj import GPTJConfig as GPTConfig
from transformers.models.gptj.modeling_gptj import fixed_pos_embedding, rotate_every_two, apply_rotary_pos_emb


# @torch.jit.script
def gpt_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

# put things on GPU to avoid high CPU usage
def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len, device=x.device), inv_freq).float()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
    
            
class GPTJMLP(_GPTJMLP):
    def __init__(self, intermediate_size, config, device='cpu'):  # in MLP: intermediate_size= 4 * embed_dim
        super(_GPTJMLP, self).__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Linear(embed_dim, intermediate_size, device=device)
        self.fc_out = nn.Linear(intermediate_size, embed_dim, device=device)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)


class GPTJAttention(_GPTJAttention):
    
    def __init__(self, config, device='cpu'):
        super(_GPTJAttention, self).__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, device=device)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, device=device)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, device=device)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, device=device)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim
            
    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        prefix_masks=None,
    ):

        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
        
        if prefix_masks is not None:
            bsz = query.size(0)
            causal_mask = causal_mask.repeat(bsz, 1, 1, 1) # (bsz, 1, src_len, tgt_len)
            causal_mask = causal_mask.permute(0, 3, 1, 2) # (bsz, tgt_len, 1, src_len)
            causal_mask[prefix_masks.bool()] = 1
            causal_mask = causal_mask.permute(0, 2, 3, 1) # (bsz, 1, src_len, tgt_len)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        offset=None,
        use_cache=False,
        output_attentions=False,
        prefix_masks=None,
    ):

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        seq_len = key.shape[1]

        if layer_past is not None:
            if offset is None:
                offset = layer_past[0].shape[-2]
            seq_len += layer_past[0].shape[-2]
            
        if offset is None:
            offset = 0

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, prefix_masks=prefix_masks)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTEmbeddings(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim, device=device)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        # module = cls(config).eval()
        module = torch.nn.utils.skip_init(cls, config).eval() # fast init
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_embs.pt',
            )))
        except:
            print(f'Cannot load from <model_path>. The model is randomly initialized.')
        return module
        
    def forward(self, input_ids, *args, **kargs):
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.wte(input_ids)
        return hidden_states
    

class GPTBlock(_GPTJBlock):
    def __init__(self, config, *args, use_checkpoint=True, device='cpu', **kargs):
        super(_GPTJBlock, self).__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, device=device)
        self.attn = GPTJAttention(config, device=device)
        self.mlp = GPTJMLP(inner_dim, config, device=device)
        self.config = config
        self.use_checkpoint = use_checkpoint

        def block_forward(x: torch.Tensor, attention_mask: torch.Tensor, prefix_masks: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_1(x)
            x_a = self.attn(x, prefix_masks=prefix_masks, attention_mask=attention_mask)[0]
            x_m = self.mlp(x)
            return res + x_a + x_m
        
        self.block_forward = block_forward
        
    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        # module = cls(config).eval()
        module = torch.nn.utils.skip_init(cls, config).eval() # fast init
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, f'pytorch_{layer_index}.pt',
            )))
        except Exception as e:
            print('Cannot load from <model_name>. The model is randomly initialized.')
            
        return module

    def forward(self, x: torch.Tensor, prefix_masks=None, layer_past=None, mask=None, skip_ln=False, **kargs) -> torch.Tensor:
        
        if mask is not None:
            # bool -> float
            attention_mask = (1e4)*(mask[:, None, None, :]-1.0)
        else:
            attention_mask = None
            
        if mask is None:
            if layer_past is not None:
                offset = layer_past[0].size(2)
            else:
                offset = 0
        else:
            # masked tokens
            offset = (mask-1).sum(-1, keepdims=False)
            if layer_past is not None:
                offset += layer_past[0].size(2)
                
        if self.training:
            
            if self.use_checkpoint:
                x.requires_grad_(True)
                x = checkpoint(self.block_forward, x, attention_mask, prefix_masks)
            else:
                x = self.block_forward(x, prefix_masks=prefix_masks)
            
            return x
           
        else:
            res = x
            if not skip_ln:
                x = self.ln_1(x)
            x_a = self.attn(x, use_cache=False, layer_past=layer_past, attention_mask=attention_mask, offset=offset, prefix_masks=prefix_masks)[0]
            x_m = self.mlp(x)
            return x_a + x_m + res
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon, device=device)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, device=device)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        # module = cls(config).eval()
        module = torch.nn.utils.skip_init(cls, config).eval() # fast init
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_lm_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module
        
    def forward(self, x, **kargs):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
