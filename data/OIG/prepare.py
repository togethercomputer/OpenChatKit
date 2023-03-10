import glob
import gzip
import os
import shutil
import subprocess

DIR = os.path.dirname(os.path.abspath(__file__))


process = subprocess.run(
    'git lfs env | grep -q \'git config filter.lfs.smudge = "git-lfs smudge -- %f"\'',
    shell=True
)
if process.returncode != 0:
    print('error: git lfs not installed. please install git-lfs and run `git lfs install`')


process = subprocess.run(
    f"git clone https://huggingface.co/datasets/laion/OIG {DIR}/files",
    shell=True,
    check=True
)

for f in glob.glob(f"{DIR}/files/*.gz"):
    out_path, _ = os.path.splitext(f)
    with (
        gzip.open(f, 'rb') as infile, 
        open(out_path, 'wb') as outfile
    ):
        shutil.copyfileobj(infile, outfile)
