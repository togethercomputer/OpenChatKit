import os
import time
import random
import json
import numpy as np
import torch

from comm.comm_utils import *


def load_checkpoint(pipe, args):
    
    if os.path.isfile(os.path.join(args.checkpoint_path, 'latest')):
        with open(os.path.join(args.checkpoint_path, 'latest')) as f:
            latest_step = int(f.read())
    else:
        print('no checkpoint available, skipping')
        return
    
    checkpoint_step_path = os.path.join(args.checkpoint_path, f"checkpoint_{latest_step}")
    
    try:
        with open(os.path.join(checkpoint_step_path, 'meta.json')) as f:
            meta = json.load(f)
    except:
        print('failed to load meta.')
        
    pipe.global_step = latest_step
    
    try:
        pipe.model.model.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_checkpoint.pt'
                ), map_location=torch.device('cpu')
            )
        )
    except:
        print('failed to load model params.')
    
    try:
        pipe.optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_optimizer.pt'
                ), map_location=torch.device('cpu')
            )
        )
    except:
        print('failed to load optim states.')
    
    try:
        pipe.scheduler.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_scheduler.pt'
                )
            )
        )
    except:
        print('failed to load scheduler states.')
        
            
def save_checkpoint(pipe, args):
    
    latest_step = pipe.global_step
    checkpoint_step_path = os.path.join(args.checkpoint_path, f"checkpoint_{latest_step}")
    
    os.system(f"mkdir -p {checkpoint_step_path}")

    torch.save(
        pipe.model.model.state_dict(),
        os.path.join(
            checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_checkpoint.pt'
        )
    )
    
    torch.save(
        pipe.optimizer.state_dict(),
        os.path.join(
            checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_optimizer.pt'
        )
    )
    
    torch.save(
        pipe.scheduler.state_dict(),
        os.path.join(
            checkpoint_step_path, f'prank_{get_pipeline_parallel_rank()}_scheduler.pt'
        )
    )
    
    with open(os.path.join(checkpoint_step_path, 'meta.json'), 'w') as f:
        json.dump({
            'step': latest_step,
        }, f)
    
    with open(os.path.join(args.checkpoint_path, 'latest'), 'w') as f:
        f.write(f"{latest_step}")
        
        
def save_stream_dataloader_state_dict(dataloader, pipe, args):
    
    latest_step = pipe.global_step
    checkpoint_step_path = os.path.join(args.checkpoint_path, f"checkpoint_{latest_step}")
    
    os.system(f"mkdir -p {checkpoint_step_path}")
    
    torch.save(
        dataloader.dataset.state_dict(),
        os.path.join(
            checkpoint_step_path, f'dataset_state_dict.pt'
        )
    )
    
def load_stream_dataloader_state_dict(dataloader, pipe, args):
    
    latest_step = pipe.global_step
    checkpoint_step_path = os.path.join(args.checkpoint_path, f"checkpoint_{latest_step}")
    
    try:
        state_dict = torch.load(
            os.path.join(
                checkpoint_step_path, f'dataset_state_dict.pt'
            )
        )

        dataloader.data.load_state_dict(state_dict)
    
    except Exception as e:
        
        print('failed to load dataset state_dict.')