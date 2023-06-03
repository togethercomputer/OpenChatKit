#!/usr/bin/env python3

# This application reports events that are stored in the event log REST service.
# Events will be reported to the event log REST service via POST at:
#
# http://<endpoint>:<port>/v1/internal/fine-tunes/<job_id>/event
#
# with Bearer authorization tokens.
#
# The ouput formate is a JSON object with the following fields:
# - "object": <object type>
# - "created_at": <timestamp>
# - "level": <event level>
# - "message": <event message>
# - "type": <event type>
# - "param_count": <number of parameters> (optional)
# - "token_count": <number of tokens> (optional)
# - "checkpoint_path": <path to checkpoint> (optional)
# - "model_path": <path to model> (optional)


import argparse
import json
import requests
import time

class EventReporter:

    supported_event_types = [
        "JOB_START",
        "MODEL_DOWNLOAD_COMPLETE",
        "TRAINING_DATA_DOWNLOAD_COMPLETE",
        "TRAINING_START",
        "EPOCH_COMPLETE",
        "TRAINING_COMPLETE",
        "JOB_COMPLETE",
        "JOB_ERROR",
    ]

    supported_event_levels = [
        "Info",
        "Warning",
        "Error",
    ]

    supported_object_types = [
        "fine-tune",
    ]

    object_type_to_endpoint = {
        "fine-tune": "fine-tunes",
        "file": "files",
    }

    def __init__(self, url, auth_token, job_id):
        self.url = url
        self.auth_token = auth_token
        self.job_id = job_id

        # Validate the URL.
        if self.url is None:
            raise ValueError("URL is required")
        
        # Validate the authorization token.
        if self.auth_token is None:
            raise ValueError("Authorization token is required")
        
        # Validate the job ID.
        if self.job_id is None:
            raise ValueError("Job ID is required")


    def report(self, object, message, event_type,
               level=supported_event_levels[0], checkpoint_path=None,
               model_path=None, param_count=None, token_count=None):
        # Get the creation timestamp.
        created_at = time.time()
        
        # Validate the object type.
        if object is None:
            raise ValueError("Object type is required")
        elif object not in self.supported_object_types:
            raise ValueError(f"Invalid object type : {object}")
        
        # Validate the message.
        if message is None:
            raise ValueError("Message is required")

        # Validate the event type.
        if event_type is None:
            raise ValueError("Event type is required")
        elif event_type not in self.supported_event_types:
            raise ValueError(f"Invalid event type : {event_type}")
        
        # Validate the event level.
        if level is None:
            level = self.supported_event_levels[0]
        elif level not in self.supported_event_levels:
            raise ValueError(f"Invalid event level : {level}")

        # Create the JSON object.
        event = {
            "object": object,
            "created_at": created_at,
            "level": level,
            "message": message,
            "type": event_type
        }
        if checkpoint_path is not None:
            event["checkpoint_path"] = checkpoint_path
        if model_path is not None:
            event["model_path"] = model_path
        if param_count is not None:
            event["param_count"] = param_count
        if token_count is not None:
            event["token_count"] = token_count
        event_str = json.dumps(event)

        # Send the event to the event log REST service.
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        endpoint = f"{self.url}/v1/internal/{self.object_type_to_endpoint[object]}/{self.job_id}/event"
        response = requests.post(endpoint, headers=headers, data=event_str)
        if response.status_code != 200:
            raise ValueError(f"Failed to send event to event log REST service: {response.text}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--event_url', type=str, required=True, help='endpoint:port')
    parser.add_argument('-a', '--auth_token', type=str, required=True, help='Bearer authorization token')
    parser.add_argument('-j', '--job_id', type=str, required=True, help='job id')
    parser.add_argument('-o', '--object', type=str, required=True, help='object type')
    parser.add_argument('-m', '--message', type=str, required=True, help='event message')
    parser.add_argument('-e', '--event_type', type=str, required=True, help='event type')
    parser.add_argument('-c', '--created_at', type=str, required=False, help='timestamp')
    parser.add_argument('-C', '--checkpoint_path', type=str, required=False, help='S3 checkpoint path')
    parser.add_argument('-M', '--model_path', type=str, required=False, help='S3 model path')
    parser.add_argument('-p', '--param_count', type=int, required=False, help='number of parameters')
    parser.add_argument('-t', '--token_count', type=int, required=False, help='number of tokens')
    parser.add_argument('-l', '--level', type=str, required=False, help='event level (Info, Warning, Error)')
    parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
    args = parser.parse_args()

    # Create the event reporter.
    event_reporter = EventReporter(url=args.event_url,
                                   auth_token=args.auth_token,
                                   job_id=args.job_id)
    
    event_reporter.report(object=args.object,
                          message=args.message,
                          event_type=args.event_type,
                          level=args.level,
                          checkpoint_path=args.checkpoint_path,
                          model_path=args.model_path,
                          param_count=args.param_count,
                          token_count=args.token_count)

# Usage: 
# python3 event_report.py \
#     -u, --url <endpoint>:<port> \
#     -a, --auth_token <authorization token>
#     -j, --job_id <job_id> \
#     -o, --object <object type>
#     -C, --checkpoint_path <S3 checkpoint path>
#     -M, --model_path <S3 model path>
#     -p, --param_count <number of parameters>
#     -t, --token_count <number of tokens>
#     -l, --level <event level>
#     -m, --message <event message>
if __name__ == '__main__':
    main()