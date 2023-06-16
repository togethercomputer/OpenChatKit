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
import sys
import time

class EventReporter:

    # Event type constants
    EVENT_TYPE_JOB_START = "JOB_START"
    EVENT_TYPE_MODEL_DOWNLOAD_COMPLETE = "MODEL_DOWNLOAD_COMPLETE"
    EVENT_TYPE_TRAINING_DATA_DOWNLOAD_COMPLETE = "TRAINING_DATA_DOWNLOAD_COMPLETE"
    EVENT_TYPE_TRAINING_START = "TRAINING_START"
    EVENT_TYPE_CHECKPOINT_SAVE = "CHECKPOINT_SAVE"
    EVENT_TYPE_EPOCH_COMPLETE = "EPOCH_COMPLETE"
    EVENT_TYPE_TRAINING_COMPLETE = "TRAINING_COMPLETE"
    EVENT_TYPE_JOB_COMPLETE = "JOB_COMPLETE"
    EVENT_TYPE_JOB_ERROR = "JOB_ERROR"

    supported_event_types = [
        EVENT_TYPE_JOB_START,
        EVENT_TYPE_MODEL_DOWNLOAD_COMPLETE,
        EVENT_TYPE_TRAINING_DATA_DOWNLOAD_COMPLETE,
        EVENT_TYPE_TRAINING_START,
        EVENT_TYPE_CHECKPOINT_SAVE,
        EVENT_TYPE_EPOCH_COMPLETE,
        EVENT_TYPE_TRAINING_COMPLETE,
        EVENT_TYPE_JOB_COMPLETE,
        EVENT_TYPE_JOB_ERROR,
    ]

    # Event level constants
    LEVEL_INFO = "Info"
    LEVEL_WARNING = "Warning"
    LEVEL_ERROR = "Error"

    supported_event_levels = [
        LEVEL_INFO,
        LEVEL_WARNING,
        LEVEL_ERROR,
    ]

    # Object type constants
    OBJECT_FINE_TUNE = "fine-tune"

    supported_object_types = [
        OBJECT_FINE_TUNE,
    ]

    object_type_to_endpoint = {
        "fine-tune": "fine-tunes",
    }

    def __init__(self, host=None, auth_token=None, job_id=None):
        self.host = host
        self.auth_token = auth_token
        self.job_id = job_id

    def is_enabled(self) -> bool:
        # Validate the URL.
        if self.host is None:
            return False
        
        # Validate the authorization token.
        if self.auth_token is None:
            return False
        
        # Validate the job ID.
        if self.job_id is None:
            return False
        
        return True

    # Report an event to the event log REST service.
    # The event will be reported to the event log REST service via POST at:
    # http://<endpoint>:<port>/v1/internal/fine-tunes/<job_id>/event
    # with Bearer authorization tokens.
    # The ouput formate is a JSON object with the following fields:
    # - "object": object type to be reported. Supported object types are given by
    #   `supported_object_types`
    # - "created_at": The creation timestamp for the event. If not specified, the
    #   current time will be used.
    # - "level": Event level. Supported event levels are given by `supported_event_levels`
    # - "message": Event message.
    # - "type": Event type. Supported event types are given by `supported_event_types`
    # - "param_count": Report the number of model parameters. (optional)
    # - "token_count": Report the number of tokens in the training data. (optional)
    # - "checkpoint_path": The path to a checkpoint file(s) (optional)
    # - "model_path": The path to model file(s) (optional)
    # - "requires_is_enabled": When true, verify that is_enabled to return true 
    #   and raises an exception if it does not. When false, this function silently
    #   exits without error. (optional)
    def report(self, object, message, event_type,
               level=LEVEL_INFO, checkpoint_path=None,
               model_path=None, param_count=None, token_count=None, 
               requires_is_enabled=True):

        if requires_is_enabled:
            # Validate the host.
            if self.host is None:
                raise ValueError("Host is required")
            
            # Validate the authorization token.
            if self.auth_token is None:
                raise ValueError("Authorization token is required")
            
            # Validate the job ID.
            if self.job_id is None:
                raise ValueError("Job ID is required")
        elif not self.is_enabled():
            print("Event reporting is disabled {self.host} {self.auth_token} {self.job_id}")
            return
        
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
        if checkpoint_path is not None and len(checkpoint_path) > 0:
            event["checkpoint_path"] = checkpoint_path
        if model_path is not None and len(model_path) > 0:
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
        endpoint = f"{self.host}/v1/internal/{self.object_type_to_endpoint[object]}/{self.job_id}/event"
        response = requests.post(endpoint, headers=headers, data=event_str)
        if response.status_code != 200:
            raise ValueError(f"Failed to send event to event log REST service: ({response.status_code}) \"{response.text}\"")
        print(f"Event reported: {event_str}")
        
def add_entry_reporter_arguments(parser):
    parser.add_argument('--event-host', type=str, required=False,
                        metavar='endpoint:port', help='Event reporting entrypoint URL')
    parser.add_argument('--event-auth-token', type=str, required=False,
                        help='Bearer authorization token')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--event-host', type=str, required=True,
                        metavar='<scheme><hostname>:<port>',
                        help='Event reporting entrypoint URL (e.g. https://127.0.0.1:8895)')
    parser.add_argument('-a', '--auth-token', type=str, required=True,
                        help='Bearer authorization token')
    parser.add_argument('-j', '--job-id', type=str, required=True, help='job id')
    parser.add_argument('-o', '--object', type=str, required=True, help='object type',
                        metavar="|".join(EventReporter.supported_object_types))
    parser.add_argument('-m', '--message', type=str, required=True, help='event message')
    parser.add_argument('-e', '--event-type', type=str, required=True, help='event type',
                        metavar="|".join(EventReporter.supported_event_types))
    parser.add_argument('-c', '--created-at', type=str, required=False, help='timestamp')
    parser.add_argument('-C', '--checkpoint-path', type=str, required=False, help='S3 checkpoint path')
    parser.add_argument('-M', '--model-path', type=str, required=False, help='S3 model path')
    parser.add_argument('-p', '--param-count', type=int, required=False, help='number of parameters')
    parser.add_argument('-t', '--token-count', type=int, required=False, help='number of tokens')
    parser.add_argument('-l', '--level', type=str, required=False, help='event level',
                        metavar="|".join(EventReporter.supported_event_levels))
    args = parser.parse_args()

    # Create the event reporter.
    event_reporter = EventReporter(host=args.event_host,
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

#usage: event_report.py [-h] -u <scheme><hostname>:<port> -a AUTH_TOKEN -j
#                       JOB_ID -o fine-tune -m MESSAGE -e
#                       JOB_START|MODEL_DOWNLOAD_COMPLETE|TRAINING_DATA_DOWNLOAD_COMPLETE|TRAINING_START|CHECKPOINT_SAVE|EPOCH_COMPLETE|TRAINING_COMPLETE|JOB_COMPLETE|JOB_ERROR
#                       [-c CREATED_AT] [-C CHECKPOINT_PATH] [-M MODEL_PATH]
#                       [-p PARAM_COUNT] [-t TOKEN_COUNT]
#                       [-l Info|Warning|Error]
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -u, --event-host <scheme><hostname>:<port>
#                        Event reporting entrypoint URL (e.g.
#                        https://127.0.0.1:8895)
#  -a, --auth-token AUTH_TOKEN
#                        Bearer authorization token
#  -j, --job-id JOB_ID
#                        job id
#  -o, --object fine-tune
#                        object type
#  -m, --message MESSAGE
#                        event message
#  -e, --event-type JOB_START|MODEL_DOWNLOAD_COMPLETE|TRAINING_DATA_DOWNLOAD_COMPLETE|TRAINING_START|CHECKPOINT_SAVE|EPOCH_COMPLETE|TRAINING_COMPLETE|JOB_COMPLETE|JOB_ERROR
#                        event type
#  -c, --created-at CREATED_AT
#                        timestamp
#  -C, --checkpoint-path CHECKPOINT_PATH
#                        S3 checkpoint path
#  -M, --model-path MODEL_PATH
#                        S3 model path
#  -p, --param-count PARAM_COUNT
#                        number of parameters
#  -t, --token-count TOKEN_COUNT
#                        number of tokens
#  -l, --level Info|Warning|Error
#                        event level
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)
    
    sys.exit(0)