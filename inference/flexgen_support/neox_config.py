"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os
import shutil
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level = logging.DEBUG)

@dataclasses.dataclass(frozen=True)
class NeoXConfig:
    name: str = "togethercomputer/GPT-NeoXT-Chat-Base-20B"
    num_hidden_layers: int = 44
    max_position_embeddings: int = 2048
    hidden_size: int = 6144
    n_head: int = 64
    input_dim: int = 6144
    ffn_embed_dim: int = 24576
    pad: int = 1
    activation_fn: str = 'gelu_fast'
    rotary_pct: float=0.25
    rotary_emb_base: int=10000
    initializer_range: float=0.02
    layer_norm_eps: float=1e-5
    vocab_size: int = 50432
    pad_token_id: int = 0 # TODO Check this, this value could be wrong. 
    dtype: type = np.float16

    def model_bytes(self):
        h = self.input_dim
        return 	2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_neox_config(name, **kwargs):
    
    if name == "togethercomputer/GPT-NeoXT-Chat-Base-20B":
        config = NeoXConfig(name=name,
            max_position_embeddings=2048, num_hidden_layers=44, n_head=64,
            hidden_size=6144, input_dim=6144, ffn_embed_dim=6144 * 4,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_neox_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.gpt_neox.modeling_gpt_noex.GPTNeoXPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


def download_neox_weights(model_name, path):
    logging.debug(f"<download_neox_weights> enter.")
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    folder = snapshot_download(model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    # if "/" in model_name:
    #     model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in state.items():
            # name = name.replace("model.", "")
            # name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            logging.debug(f"{name}: {param.shape}")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            #if "decoder.embed_tokens.weight" in name:
            #    shutil.copy(param_path, param_path.replace(
            #        "decoder.embed_tokens.weight", "lm_head.weight"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="/root/flexgen_weights")
    args = parser.parse_args()

    download_neox_weights(args.model, args.path)
