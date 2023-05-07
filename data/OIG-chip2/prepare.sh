DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
wget https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl -O ${DIR}/unified_chip2.jsonl