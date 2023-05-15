import sys
import os

# Import the prepare_data function
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from prepare_data import prepare_data

if __name__ == "__main__":
    dest_dir = os.path.join(current_dir, "files")
    prepare_data("https://huggingface.co/datasets/ChristophSchuhmann/wikipedia-3sentence-level-retrieval-index", dest_dir)
