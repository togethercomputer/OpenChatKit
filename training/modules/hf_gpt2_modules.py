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
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as _GPT2Attention
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP as _GPT2MLP
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as _GPT2Block
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as _GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel as _GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification as _GPT2ForSequenceClassification
from transformers.models.gpt2.configuration_gpt2 import GPT2Config as GPTConfig
from typing import Optional, Tuple, Union


# @torch.jit.script
def gpt_loss_func(input, target):
    lm_logits, labels = input, target
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


class GPTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        
    def forward(self, input_ids, **kargs):
        
        device = input_ids.device
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)
        
        return hidden_states
    
class GPTAttention(_GPT2Attention):
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None, prefix_masks=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            if prefix_masks is not None:
                bsz = query.size(0)
                causal_mask = causal_mask.repeat(bsz, 1, 1, 1) # (bsz, 1, src_len, tgt_len)
                causal_mask = causal_mask.permute(0, 3, 1, 2) # (bsz, tgt_len, 1, src_len)
                causal_mask[prefix_masks.bool()] = 1
                causal_mask = causal_mask.permute(0, 2, 3, 1) # (bsz, 1, src_len, tgt_len)
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        prefix_masks = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, prefix_masks=prefix_masks)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    

class GPTBlock(_GPT2Block):
    def __init__(self, config, layer_idx=None, use_checkpoint=True):
        super(_GPT2Block, self).__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = _GPT2MLP(inner_dim, config)
        
        self.config = config
        self.use_checkpoint = use_checkpoint
        
        def attn_res(x: torch.Tensor, prefix_masks: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_1(x)
            x = self.attn(x, prefix_masks=prefix_masks)[0]
            return x + res
        self.attn_res = attn_res
        
        def mlp_res(x: torch.Tensor) -> torch.Tensor:
            res = x
            x = self.ln_2(x)
            x = self.mlp(x)
            return x + res
        self.mlp_res = mlp_res
        

    def forward(self, x: torch.Tensor, prefix_masks=None, **kargs) -> torch.Tensor:
        
        if not self.training:
            x = self.attn_res(x, prefix_masks=prefix_masks)
            x = self.mlp_res(x)
            return x
        
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.attn_res, x, prefix_masks)
        else:
            x = self.attn_res(x, prefix_masks=prefix_masks)
        if self.use_checkpoint:
            x.requires_grad_(True)
            x = checkpoint(self.mlp_res, x)
        else:
            x = self.mlp_res(x)
        return x
    
    
class GPTModel(_GPT2Model):
    def __init__(self, config):
        super(_GPT2Model, self).__init__(config)

        self.embed_dim = config.hidden_size
        
        emb_layer = GPTEmbeddings(config)
        self.wte = emb_layer.wte
        self.wpe = emb_layer.wpe

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTBlock(config, layer_idx=i, use_checkpoint=True) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(self, input_ids, attention_mask=None, **kargs):
        
        device = input_ids.device
        
        # input ids
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_shape[0]
        
        # position ids
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        hidden_states_tuple = tuple()
        for layer in self.h:
            hidden_states_tuple = hidden_states_tuple + (hidden_states,)
            hidden_states = layer(hidden_states)
        hidden_states = self.ln_f(hidden_states)
        hidden_states_tuple = hidden_states_tuple + (hidden_states,)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=hidden_states_tuple,
            attentions=None,
            cross_attentions=None,
        )
    
class GPTLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, x, **kargs):
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
    
class GPTLMHeadModel(_GPT2LMHeadModel):

    def __init__(self, config):
        super(_GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # ln_f will be calculated in self.transformer

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        # Initialize weights and apply final processing
        self.post_init()
        
class GPTClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.score = nn.Linear(config.n_embd, config.num_labels, bias=False)
        
    def forward(self, hidden_states, input_ids=None):
        
        batch_size, sequence_length = hidden_states.shape[:2]
        if input_ids is not None:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
        
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        logits = self.score(self.ln_f(pooled_hidden_states))
        
        return logits
        
class GPTForClassification(_GPT2ForSequenceClassification):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
        
#     def forward(self, input_ids, labels=None):
        
#         ret = self.transformer(input_ids)
#         pool_hidden_state = ret.last_hidden_state[:, -1]
        
#         logits = self.score(pool_hidden_state)
        
#         loss = functional.cross_entropy(logits, labels)
        
#         return loss
        