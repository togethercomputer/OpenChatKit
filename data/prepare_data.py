import argparse
import glob
import gzip
import os
import shutil
import subprocess
import sys

# Check if git-lfs is installed.
def is_git_lfs_installed():
    try:
        process = subprocess.run(['git', 'lfs', 'version'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
        return process.returncode == 0
    except FileNotFoundError:
        return False

# Check that the current git repository has git-lfs installed. If the git-lfs
# is not installed, then run `git lfs install` if git-lfs is installed. If 
# git-lfs is not installed, then print an error message and exit.
def check_git_lfs():
    process = subprocess.run(
        'git lfs env | grep -q \'git config filter.lfs.smudge = "git-lfs smudge -- %f"\'',
        shell=True
    )

    # Check if git-lfs is installed
    if process.returncode != 0 and is_git_lfs_installed():
        subprocess.run('git lfs install', shell=True, check=True)
        process = subprocess.run(
            'git lfs install',
            shell=True
        )

    if process.returncode != 0:
        print('error: git lfs not installed. please install git-lfs and run `git lfs install`')
        sys.exit(1)

# Perepare data will clone the git repository given by data_source into the
# destination_dir.
def prepare_data(data_source, destination_dir):
    if os.path.exists(destination_dir):
        print(f"Destination directory {destination_dir} already exists. Skipping data preparations.")
        return
    
    check_git_lfs()

    subprocess.run(f"git clone {data_source} {destination_dir}", shell=True,
                   check=True)
    
    # Extract gzipped files, if present
    for file_path in glob.glob(f"{destination_dir}/*.gz"):
        out_path, _ = os.path.splitext(file_path)
        with gzip.open(file_path, 'rb') as infile, open(out_path, 'wb') as outfile:
            shutil.copyfileobj(infile, outfile)

def main():
    parser = argparse.ArgumentParser(description="Script for cloning a git repository and extracting files.")
    parser.add_argument("-s", "--data-source", required=True, help="URL of the data source (git repository)")
    parser.add_argument("-d", "--dest", required=True, help="Destination directory to clone the repository and extract files")

    args = parser.parse_args()
    prepare_data(args.data_source, args.dest)


if __name__ == "__main__":
    main()
