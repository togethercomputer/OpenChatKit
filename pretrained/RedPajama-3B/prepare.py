import sys
import os

# Import the prepare_data function
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from prepare_pretrained import prepare_pretrained

if __name__ == "__main__":
    model_name = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
    save_path = os.path.join(current_dir, model_name.replace('/', '_'))
    prepare_pretrained(save_path, model_name)
