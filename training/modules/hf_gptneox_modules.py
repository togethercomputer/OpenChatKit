import os
import torch
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
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXRotaryEmbedding


try:
    from flash_attn.flash_attention import FlashAttention
    flash_attn_installed = True
    print('>>>>> using flash attention')
except ImportError:
    flash_attn_installed = False

try:
    from fav2.fav2_interface import flash_attn_qkvpacked_func as fav2_qkvpacked_func
    flash_attn_v2_installed = True
    print('>>>>> using flash attention v2')

    class FlashAttentionV2(nn.Module):
        """Implement the scaled dot product attention with softmax.
        Arguments
        ---------
            softmax_scale: The temperature to use for the softmax attention.
                          (default: 1/sqrt(d_keys) where d_keys is computed at
                          runtime)
            attention_dropout: The dropout rate to apply to the attention
                               (default: 0.0)
        """
        def __init__(self, softmax_scale=None, attention_dropout=0.0):
            super().__init__()
            self.softmax_scale = softmax_scale
            self.dropout_p = attention_dropout
    
        def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                    max_s=None, need_weights=False):
            """Implements the multihead softmax attention.
            Arguments
            ---------
                qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                    if unpadded: (nnz, 3, h, d)
                key_padding_mask: a bool tensor of shape (B, S)
            """
            assert not need_weights
            assert qkv.dtype in [torch.float16, torch.bfloat16]
            assert qkv.is_cuda
            assert key_padding_mask is None
            assert cu_seqlens is None
            assert max_s is None

            output = fav2_qkvpacked_func(
                qkv, self.dropout_p if self.training else 0.0, 
                softmax_scale=self.softmax_scale, causal=causal
            )
    
            return output, None
except ImportError:
    flash_attn_v2_installed = False

    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    if isinstance(offset, torch.Tensor):
        realidx = torch.arange(q.shape[-2], device=q.device).view(
            1, q.shape[-2]) + offset[:, None]
        cos = cos.squeeze(0).squeeze(0)[realidx].view(offset.size(0),
                                                      1, q.shape[-2],
                                                      cos.size(-1))
        sin = sin.squeeze(0).squeeze(0)[realidx].view(offset.size(0),
                                                      1, q.shape[-2],
                                                      sin.size(-1))
    else:
        cos = cos[..., offset : q.shape[-2] + offset, :]
        sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GPTNeoXAttention(_GPTNeoXAttention):
    
    def __init__(self, config):
        super(_GPTNeoXAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = GPTNeoXRotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if flash_attn_v2_installed:
            self.flash_attn = FlashAttentionV2(softmax_scale=1.0/self.norm_factor, attention_dropout = 0)
        elif flash_attn_installed:
            self.flash_attn = FlashAttention(softmax_scale=1.0/self.norm_factor, attention_dropout = 0)

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
        
        bsz, tgt_len, _ = hidden_states.shape
        
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads,
                                           3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., :self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size:2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., :self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., :self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]

        if layer_past is not None:
            if offset is None:
                offset = layer_past[0].shape[-2]
            seq_len += layer_past[0].shape[-2]

        if offset is None:
            offset = 0

        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot,
                                          key_rot,
                                          cos,
                                          sin,
                                          offset=offset)
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
        if flash_attn_installed or flash_attn_v2_installed:
            
            query = query.permute(0, 2, 1, 3).half()
            key = key.permute(0, 2, 1, 3).half()
            value = value.permute(0, 2, 1, 3).half()
            qkv = torch.stack(
                [
                    query.reshape((bsz, tgt_len, self.num_attention_heads, self.head_size)),
                    key.reshape((bsz, tgt_len, self.num_attention_heads, self.head_size)),
                    value.reshape((bsz, tgt_len, self.num_attention_heads, self.head_size)),
                ],
                dim=2
            )

            attn_weights = None
            attn_output, _ = self.flash_attn(qkv, causal=True)
            attn_output = attn_output.view(bsz, tgt_len, self.num_attention_heads * self.head_size)
        else:
            attn_output, attn_weights = self._attn(query, key, value,
                                                   attention_mask, head_mask)
            # Reshape outputs
            attn_output = self._merge_heads(attn_output, self.num_attention_heads,
                                            self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights, )

        return outputs

    # fix nan problem
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size(
        )
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length].bool()

        query = query.view(batch_size * num_attention_heads, query_length,
                           attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length,
                       attn_head_size)
        attn_scores = torch.zeros(  # empty sometimes gives nan
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
        attn_scores = attn_scores.view(batch_size, num_attention_heads,
                                       query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
            attn_scores.device)
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
            module.load_state_dict(
                torch.load(os.path.join(
                    model_path,
                    'pytorch_embs.pt',
                )))
        except:
            print(
                f'Cannot load from <model_path>. The model is randomly initialized.'
            )
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
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.layer_norm_eps)
        self.attention = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)
        self.config = config
        self.use_checkpoint = use_checkpoint

        def block_forward(x: torch.Tensor, attention_mask: torch.Tensor,
                          prefix_masks: torch.Tensor) -> torch.Tensor:
            res = x
            """
            To be compatible with https://github.com/huggingface/transformers/blob/a0ae2310ec46a2c592950babc85cf02e325bf6a7/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L336-L347
            """
            layer_norm_out = self.input_layernorm(x)
            attention_layer_output = self.attention(layer_norm_out, attention_mask=attention_mask)
            attn_output = attention_layer_output[0]
            # outputs = attention_layer_output[1:]

            if hasattr(self.config, 'use_parallel_residual') and self.config.use_parallel_residual:
                # x = x + attn(ln1(x)) + mlp(ln2(x))
                # x_a = attn_output, 
                mlp_out = self.mlp(self.post_attention_layernorm(x))
                return res + attn_output + mlp_out
            else:
                # x = x + attn(ln1(x)) 
                # x = x + mlp(ln2(x))
                attn_output = attn_output + x
                mlp_out = self.mlp(self.post_attention_layernorm(attn_output))
                return attn_output + mlp_out

        self.block_forward = block_forward

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        assert layer_index is not None
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval().half()
        try:
            module.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        f'pytorch_{layer_index}.pt',
                    )))
        except Exception as e:
            print(
                'Cannot load from <model_name>. The model is randomly initialized.'
            )
        return module

    def forward(self,
                x: torch.Tensor,
                layer_past=None,
                mask=None,
                **kargs) -> torch.Tensor:

        if mask is not None:
            # bool -> float
            attention_mask = 1e9 * (mask[:, None, None, :] - 1)
        else:
            attention_mask = None

        if mask is None:
            if layer_past is not None:
                offset = layer_past[0].size(2)
            else:
                offset = 0
        else:
            # masked tokens
            offset = (mask - 1).sum(-1, keepdims=False)
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
            attn_output = attention_layer_outputs[
                0]  # output_attn: a, present, ...

            mlp_output = self.mlp(self.post_attention_layernorm(x))
            x = mlp_output + attn_output + residual

            return x


class GPTLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)
        self.embed_out = nn.Linear(config.hidden_size,
                                   config.vocab_size,
                                   bias=False)

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = GPTConfig.from_pretrained(model_path)
        module = cls(config).eval()
        try:
            module.load_state_dict(
                torch.load(os.path.join(
                    model_path,
                    'pytorch_lm_head.pt',
                )))
        except:
            print(
                'Cannot load from <model_name>. The model is randomly initialized.'
            )
        return module

    def forward(self, x, *args, **kargs):
        x = self.final_layer_norm(x)
        x = self.embed_out(x)
        return x
