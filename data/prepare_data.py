import argparse
from shutil import copyfile
import boto3
import botocore
import glob
import gzip
import os
import re
import requests
import shutil
import subprocess
import sys
from urllib.parse import urlparse


# Check if git-lfs is installed.
def is_git_lfs_installed():
    try:
        process = subprocess.run(['git', 'lfs', 'version'], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
        return process.returncode == 0
    except FileNotFoundError:
        return False

# Check if a url is a Hugging Face git URL.
def is_huggingface_git_url(url):
    # Regular expression pattern for Hugging Face git URLs
    hf_git_pattern = r'^https://huggingface\.co/datasets/[A-Za-z0-9_\.\-/]+$'
    
    # Match the pattern against the URL
    # Return True if a match is found, False otherwise
    return re.match(hf_git_pattern, url) is not None

# Check if the path is a GitHub repository URL.
def is_github_repo_url(url):
    # Regular expression patterns for GitHub repository URLs
    ssh_pattern = r'^git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$'
    http_pattern = r'^https?://(www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$'
    
    # Match the patterns against the path
    # Return True if a match is found in either SSH or HTTP pattern, False otherwise
    return re.match(ssh_pattern, url) is not None or re.match(http_pattern, url) is not None


# Check if the path is an S3 or R2 repository URL.
def is_s3_url(url):
    # Regular expression pattern for S3 URLs
    s3_pattern = r'^https?://(s3(-[a-z0-9-]+)?\.amazonaws|[a-fA-F0-9]+\.r2\.cloudflarestorage)\.com/[a-z0-9][a-z0-9\.\-]{1,61}[a-z0-9]/[0-9a-zA-Z!\-_\.*\'()/]+$'
    
    # Match the pattern against the URL
    # Return True if a match is found, False otherwise
    if re.match(s3_pattern, url) is None:
        return False
    
    # Check for a valid bucket name
    bucket_name = url.split('/')[3]
    if bucket_name.startswith("xn--"):
        return False
    if bucket_name.endswith("-s3alias"):
        return False
    if bucket_name.endswith("--ol-s3"):
        return False
    if re.match(r'\.\.', bucket_name) is not None:
        return False
    if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', bucket_name) is not None:
        return False
    if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', bucket_name) is not None:
        return False
    
    return True


# Check that the current git repository has git-lfs installed. If the git-lfs
# is not installed, then run `git lfs install` if git-lfs is installed. If 
# git-lfs is not installed, then print an error message and exit.
def clone_git_repo(data_source, destination_dir):
    process = subprocess.run(
        'git lfs env | grep -q \'git config filter.lfs.smudge = "git-lfs smudge -- %f"\'',
        shell=True
    )

    # Check if the git repository has already been cloned
    if os.path.exists(os.path.join(destination_dir, ".git")):
        print(f"Git repository already exists at {destination_dir}. Skipping clone.")
        return

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

    # Clone a GitHub repository.
    try:
        subprocess.run(f"git clone {data_source} {destination_dir}", shell=True,
                       check=True)
    except subprocess.CalledProcessError:
        print(f"error: failed to clone repository {data_source}")
        sys.exit(1)

    

# Download all files from an S3 compatible storage service.
def download_from_s3(url, destination_dir, access_key_id = None,
                     secret_access_key = None, session_token = None, debug = False):
    # Get the access key ID and secret access key from the environment variables
    if access_key_id is None:
        access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    if secret_access_key is None:
        secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if session_token is None:
        session_token = os.environ.get('AWS_SESSION_TOKEN')
    
    # Create an S3 client
    parsed_url = url.split('/')
    endpoint_url = f"{parsed_url[0]}//{parsed_url[2]}"
    bucket_name = parsed_url[3]
    key_prefix = "/".join(parsed_url[4:-1])
    base_file = parsed_url[-1] if not url.endswith('/') else ""
    
    print(f"endpoint_url={endpoint_url} ...")
    if debug:
        print(f"access_key_id={access_key_id}")
        print(f"secret_access_key={secret_access_key}")
        print(f"bucket_name={bucket_name}")
        print(f"key_prefix={key_prefix}")
        print(f"base_file={base_file}")

    s3 = boto3.resource('s3',
        endpoint_url = endpoint_url,
        aws_access_key_id = access_key_id,
        aws_secret_access_key = secret_access_key,
        aws_session_token=session_token,
        region_name = "auto"
    )
    
    # Create the destination directory if it does not exist
    os.makedirs(destination_dir, exist_ok=True)

    try:
        print(f"Downloading file(s) from S3 {url} to {destination_dir} ...")
        bucket = s3.Bucket(bucket_name)
        
        # Otherwise, download the file at the prefix
        if url.endswith('/'):
            # Download the file from the S3 path
            for obj in bucket.objects.filter(Prefix=key_prefix):
                if not obj.key.endswith('/'):
                    destination_file = os.path.join(destination_dir, os.path.basename(obj.key))
                    if not os.path.exists(destination_file):
                        print(f"Downloading {obj.key} ...")
                        bucket.download_file(obj.key, destination_file)
                    else:
                        print(f"File already exists, skipping {obj.key}")
        else:
            destination_file = os.path.join(destination_dir, base_file)
            if not os.path.exists(destination_file):
                print(f"Downloading {base_file} ...")
                bucket.download_file(f'/{key_prefix}/{base_file}', destination_file)
            else:
                print(f"File already exists, skipping {base_file}")

        print("Download completed successfully.")
        return
    
    except botocore.exceptions.NoCredentialsError:
        print("Error: AWS credentials not found.") 
    except botocore.exceptions.EndpointConnectionError:
        print("Error: Unable to connect to the S3 endpoint.")
    except botocore.exceptions.ParamValidationError as e:
        print(f"Error: Invalid S3 URL: {e}")
    except botocore.exceptions.ClientError as e:
        print(f"Error: {e.response['Error']['Message']}")
    
    # Something went wrong, exit with error.
    sys.exit(1)

def download_from_url(url, destination_dir):
    print(f"Downloading file from {url} to {destination_dir} ...")
    try:
        # Parse the URL to extract the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # Construct the destination file path
        destination_file = os.path.join(destination_dir, filename)
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
        print("Download completed successfully.")
        return
    
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the URL.")
    except requests.exceptions.Timeout:
        print("Error: Connection timed out.")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    # Something went wrong, exit with error.
    sys.exit(1)

# Perepare data will clone the git repository given by data_source into the
# destination_dir.
def prepare_data(data_source, destination_dir, access_key_id=None, secret_access_key=None, debug=False):

    # Check that destination_dir is a directory. If it does not exist, then
    # create it.
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    elif not os.path.isdir(destination_dir):
        print(f"Error: {destination_dir} is not a directory.")
        sys.exit(1)

    if os.path.isfile(data_source):
        # Handle the case where the data source is a local file
        print(f"Copying file {data_source} to {destination_dir} ...")
        copyfile(data_source, destination_dir)
    elif is_github_repo_url(data_source) or is_huggingface_git_url(data_source):
        # Handle the case where the data source is a GitHub or Hugging Face repository
        clone_git_repo(data_source, destination_dir)
    elif is_s3_url(data_source):
        # Handle the case where the data source is an S3 URL
        download_from_s3(url=data_source, destination_dir=destination_dir, access_key_id=access_key_id, 
                         secret_access_key=secret_access_key, debug=debug)
    elif data_source.startswith('http://') or data_source.startswith('https://'):
        # Handle the case where the data source is a URL
        download_from_url(data_source, destination_dir)
    else:
        print(f"Error: Invalid data source: {data_source}")
        sys.exit(1)

    # Extract gzipped files, if present
    for file_path in glob.glob(f"{destination_dir}/*.gz"):
        out_path, _ = os.path.splitext(file_path)
        with gzip.open(file_path, 'rb') as infile, open(out_path, 'wb') as outfile:
            shutil.copyfileobj(infile, outfile)
        os.remove(file_path)
    
def main():
    parser = argparse.ArgumentParser(description="Script for cloning a git repository and extracting files.")
    parser.add_argument("-s", "--data-source", required=True, help="URL of the data source (git repository)")
    parser.add_argument("-d", "--dest", required=True, help="Destination directory to clone the repository and extract files")
    parser.add_argument("-a", "--access-key-id", required=False, help="AWS access key ID")
    parser.add_argument("-k", "--secret-access-key", required=False, help="AWS secret access key")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")

    args = parser.parse_args()
    prepare_data(args.data_source, args.dest, args.access_key_id, args.secret_access_key, args.debug)


if __name__ == "__main__":
    main()
