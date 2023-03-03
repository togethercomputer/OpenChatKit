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
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention as _GPTNeoXAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer as _GPTNeoXBlock
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel as _GPTNeoXModel
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig as GPTConfig

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset = 0):
    if isinstance(offset, torch.Tensor):
        realidx = torch.arange(q.shape[-2], device=q.device).view(1, q.shape[-2]) + offset[:, None]
        cos = cos.squeeze(0).squeeze(0)[realidx].view(offset.size(0), 1, q.shape[-2], cos.size(-1))
        sin = sin.squeeze(0).squeeze(0)[realidx].view(offset.size(0), 1, q.shape[-2], sin.size(-1))
    else:
        cos = cos[..., offset : q.shape[-2] + offset, :]
        sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GPTNeoXAttention(_GPTNeoXAttention):
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        offset=None,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        
        if layer_past is not None:
            if offset is None:
                offset = layer_past[0].shape[-2]
            seq_len += layer_past[0].shape[-2]
            
        if offset is None:
            offset = 0
        
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = None if use_cache else (key, value)

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    # fix nan problem
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros( # empty sometimes gives nan
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.embed_in = nn.Embedding(config.vocab_size, self.embed_dim)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
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
        hidden_states = self.embed_in(input_ids)
        return hidden_states
    

class GPTBlock(_GPTNeoXBlock):
    def __init__(self, config, *args, use_checkpoint=True, **kargs):
        super(_GPTNeoXBlock, self).__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)
        self.config = config
        self.use_checkpoint = use_checkpoint
        
        def block_forward(x: torch.Tensor, attention_mask: torch.Tensor, prefix_masks: torch.Tensor) -> torch.Tensor:
            res = x
            ln_out = self.input_layernorm(x)
            x_a = self.attention(ln_out, attention_mask=attention_mask)[0]
            x_m = self.mlp(self.post_attention_layernorm(x))
            return res + x_a + x_m
        
        self.block_forward = block_forward

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval().half()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, f'pytorch_{layer_index}.pt',
            )))
        except Exception as e:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module
    
    def forward(self, x: torch.Tensor, layer_past=None, mask=None, **kargs) -> torch.Tensor:
        
        if mask is not None:
            # bool -> float
            attention_mask = 1e9*(mask[:, None, None, :]-1)
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
                x = checkpoint(self.block_forward, x, attention_mask, None)
            else:
                x = self.block_forward(x, prefix_masks=prefix_masks)
            
            return x
           
        else:
        
            residual = x
            ln_out = self.input_layernorm(x)
            attention_layer_outputs = self.attention(
                ln_out,
                attention_mask=attention_mask,
            )
            attn_output = attention_layer_outputs[0]  # output_attn: a, present, ...

            mlp_output = self.mlp(self.post_attention_layernorm(x))
            x = mlp_output + attn_output + residual

            return x
    
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(
                model_path, 'pytorch_lm_head.pt',
            )))
        except:
            print('Cannot load from <model_name>. The model is randomly initialized.')
        return module
        
    def forward(self, x, *args, **kargs):
        x = self.final_layer_norm(x)
        x = self.embed_out(x)
        return x