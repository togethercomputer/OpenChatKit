import os
import subprocess

DIR = os.path.dirname(os.path.abspath(__file__))


process = subprocess.run(
    'git lfs env | grep -q \'git config filter.lfs.smudge = "git-lfs smudge -- %f"\'',
    shell=True
)
if process.returncode != 0:
    print('error: git lfs not installed. please install git-lfs and run `git lfs install`')


process = subprocess.run(
    f"git clone https://huggingface.co/datasets/ontocord/OIG-moderation {DIR}/files",
    shell=True,
    check=True
)
