import argparse
import boto3
import concurrent.futures
import os
import re
import sys
import time

from utils.event_report import *

class UploadManager:
    def __init__(self, aws_endpoint_url: str, aws_access_key_id: str,
                 aws_secret_access_key: str, aws_session_token: str = None,
                 aws_region: str = "auto", event_reporter: EventReporter = None,
                 n_stages: int = 1, max_wait_sec: int = 600, dry_run: bool = False):

        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.futures = []

        if aws_endpoint_url is not None and aws_access_key_id is not None and aws_secret_access_key is not None and aws_region is not None:
            # Create an S3 client
            self.aws_access_key_id = aws_access_key_id
            self.aws_secret_access_key = aws_secret_access_key
            self.aws_session_token = aws_session_token
            self.aws_region = aws_region
            self.aws_endpoint_url = aws_endpoint_url
            self.enabled = True
        else:
            self.aws_access_key_id = None
            self.aws_secret_access_key = None
            self.aws_session_token = None
            self.aws_region = None
            self.aws_endpoint_url = None
            self.enabled = False

        self.event_reporter = event_reporter
        self.dry_run = dry_run
        if n_stages < 1 and self.enabled:
            raise ValueError("n_stages must be greater than or equal to 1")
        self.n_stages = n_stages
        self.max_wait_sec = max_wait_sec

    def add_task(self, directory: str, checkpoint_upload_prefix: str, step: int = 0):
        if self.enabled:
            # Check that the provided checkpoint upload s3 prefix is valid regex
            if not re.match(r"s3://[a-zA-Z0-9.\-_]{3,255}/.+", checkpoint_upload_prefix):
                raise ValueError("checkpoint_upload_prefix must start with s3://")
            # Get the s3 bucket and key from the checkpoint upload prefix
            s3_bucket = checkpoint_upload_prefix.split("/")[2]
            s3_key_prefix = "/".join(checkpoint_upload_prefix.split("/")[3:])
            if not s3_key_prefix.endswith("/"):
                s3_key_prefix += "/"
            print(f"Uploading checkpoint to bucket=\"{s3_bucket}\", prefix=\"{s3_key_prefix}\"")

            future = self.executor.submit(self._execute_task, directory, s3_bucket, s3_key_prefix, step)
            self.futures.append(future)

    def wait(self):
        if self.enabled:
            concurrent.futures.wait(self.futures)

    def _wait_for_file_write_to_finish(self, file_path: str, wait_start_time: float) -> bool:
        try:
            file_size = os.stat(file_path).st_size
            while True:
                time.sleep(2)
                file_size_after = os.stat(file_path).st_size
                if file_size == file_size_after:
                    return True
                if time.time() - wait_start_time > self.max_wait_sec:
                    return False
                file_size = file_size_after
        except Exception as e:
            print(f"Exception while waiting for file write to finish: {e}")
            return False

    def _execute_task(self, directory, s3_bucket, s3_key_prefix, step: int):
        try:
            # Create an S3 client
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.aws_region
            )
            s3_client = session.client('s3', endpoint_url=self.aws_endpoint_url)

            print(f"Step {step} - Wait for all checkpoint stages to finish ...")

            wait_start_time = time.time()
            finished_files = set()

            # Wait for all stages to finish
            # Each stage is written by a separate process. We don't know which process
            # will finish first. So we wait for all stages to finish before proceeding.
            while True:
                # Get the list of files in the directory
                files = os.listdir(directory)
                print(f"Step {step} - Found {len(files)} of expected {3 * self.n_stages + 1} files in directory: {directory}")

                # Check if all stages have finished
                all_finished = False
                if len(files) == 3 * self.n_stages + 1:
                    all_finished = True
                    # Check if all files are closed
                    for file in files:
                        print(f"Step {step} - Checking if {file} has is finished writing ...")
                        if file not in finished_files:
                            if self._wait_for_file_write_to_finish(os.path.join(directory, file), wait_start_time) == False:
                                all_finished = False
                                break
                            else:
                                print(f"Step {step} - Checking if {file} has is finished writing ... Done")
                                finished_files.add(file)

                else:
                    all_finished = False

                if all_finished:
                    break

                # Check if we have timed out waiting for all stages to finish
                if time.time() - wait_start_time > self.max_wait_sec:
                    print(f"Step {step} - Timeout waiting for all stages to finish")
                    return
                
                time.sleep(10)

            print(f"Step {step} - Compressing files in directory: {directory}")
            tar_file_path = f"{directory}.tar.zst"

            # Get the tar file path
            tar_file_name = os.path.basename(tar_file_path)

            # Compress the directory via cli
            if not self.dry_run:
                if os.system(f"tar -cf - -C \"{directory}\" . | zstd -3 -T4 > \"{tar_file_path}\"") != 0:
                    print(f"Step {step} - Failed to compress {directory}")
                    return

            s3_key = f"{s3_key_prefix}{tar_file_name}"
            print(f"Step {step} - Uploading checkpoint to s3://{s3_bucket}/{s3_key}")
            if not self.dry_run:
                # Try uploading the tar file to s3. If it fails, try again after
                # 20 seconds.
                for i in range(3):
                    try:
                        s3_client.upload_file(tar_file_path, s3_bucket, s3_key)
                        break
                    except Exception as e:
                        print(f"Step {step} - Failed to upload checkpoint to s3: {e}")
                        if i == 2:
                            self.event_reporter.report(object=EventReporter.OBJECT_FINE_TUNE,
                                                    message=f"Step {step}, failed to upload checkpoint",
                                                    event_type=EventReporter.EVENT_TYPE_JOB_ERROR,
                                                    level=EventReporter.LEVEL_ERROR,
                                                    requires_is_enabled=False)
                            return
                        time.sleep(20)

                os.remove(tar_file_path)

            if self.event_reporter is not None:
                print(f"Step {step} - Reporting event")
                try:
                    self.event_reporter.report(object=EventReporter.OBJECT_FINE_TUNE,
                                            message=f"Uploaded checkpoint, at step {step}",
                                            event_type=EventReporter.EVENT_TYPE_CHECKPOINT_SAVE,
                                            checkpoint_path=f"s3://{s3_bucket}/{s3_key}",
                                            requires_is_enabled=False)
                except Exception as e:
                    print(f"Step {step} - Failed to report event: {e}")
            else:
                print(f"Step {step} - Event reporter is disabled, skipping reporting event")
        except Exception as e:
            print(f"Exception: Step {step} - {e}")
            self.event_reporter.report(object=EventReporter.OBJECT_FINE_TUNE,
                                       message=f"Step {step}, failed to upload checkpoint",
                                       event_type=EventReporter.EVENT_TYPE_JOB_ERROR,
                                       level=EventReporter.LEVEL_ERROR,
                                       requires_is_enabled=False)
            


def add_aws_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--aws-endpoint-url', help='AWS endpoint URL')
    parser.add_argument('--aws-access-key-id', help='AWS access key ID')
    parser.add_argument('--aws-secret-access-key', help='AWS secret access key')
    parser.add_argument('--aws-session-token', help='AWS session token')
    parser.add_argument('--aws-region', default='auto', help='AWS region (default: auto)')

def aws_process_args(args: argparse.Namespace):
    if args.aws_endpoint_url is None:
        args.aws_endpoint_url = os.environ.get('AWS_ENDPOINT_URL', 'https://s3.amazonaws.com')
    if args.aws_access_key_id is None:
        args.aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        if args.aws_access_key_id is None:
            print("Error: AWS_ACCESS_KEY_ID is not set")
            sys.exit(1)
    if args.aws_secret_access_key is None:
        args.aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if args.aws_secret_access_key is None:
            print("Error: AWS_SECRET_ACCESS_KEY is not set")
            sys.exit(1)
    if args.aws_session_token is None:
        args.aws_session_token = os.environ.get('AWS_SESSION_TOKEN')

def main():
    parser = argparse.ArgumentParser(description='Process S3 file objects with a specific prefix')
    parser.add_argument('--bucket-name', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', required=True, help='Prefix for the S3 objects')
    add_aws_arguments(parser)
    add_entry_reporter_arguments(parser)
    parser.add_argument('--job-id', '-j', type=str, required=True, help='job id')
    parser.add_argument('--n-stages', type=int, default=1, help='Number of stages')
    parser.add_argument('--dry-run', action='store_true', default=False, 
                        help='Perform a dry run (only print file paths)')
    parser.add_argument('directories', nargs='+', help='Directories to upload')

    args = parser.parse_args()
    aws_process_args(args)

    event_reporter = None
    if args.event_host is not None and args.event_auth_token is not None and args.job_id is not None:
        event_reporter = EventReporter(host=args.event_host, auth_token=args.event_auth_token, job_id=args.job_id)

    task_manager = UploadManager(aws_endpoint_url = args.aws_endpoint_url,
                                 aws_access_key_id = args.aws_access_key_id,
                                 aws_secret_access_key = args.aws_secret_access_key,
                                 aws_session_token = args.aws_session_token,
                                 aws_region = args.aws_region,
                                 event_reporter = event_reporter,
                                 n_stages = args.n_stages,
                                 dry_run = args.dry_run)

    checkpoint_upload_prefix = f"s3://{args.bucket_name}/{args.prefix}/"
    step = 0
    for directory in args.directories:
        print(f"Adding task for directory: {directory}")
        step += 1
        task_manager.add_task(directory=directory, checkpoint_upload_prefix=checkpoint_upload_prefix, step=step)
        time.sleep(20)

    print("Waiting for tasks to complete...")
    start_time = time.time()
    task_manager.wait()
    end_time = time.time()
    print(f"Tasks completed in {end_time - start_time} sec")

if __name__ == "__main__":
    main()