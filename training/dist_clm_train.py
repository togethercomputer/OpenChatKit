import argparse
import time
import random
import numpy as np
import torch
import torch.autograd.profiler as profiler
from tasks.data_loaders.data_utils import get_train_data_loader, get_eval_data_loader
from modules.utils import gpt_loss_func
from modules.tokenizer import build_tokenizer
from pipeline_parallel.dist_pp_utils import get_pp_module

from transformers import AutoConfig
import datasets

from utils.dist_args_utils import *
from utils.dist_checkpoint_utils import *
from utils.logging_utils import *
from utils.event_report import *
from comm.comm_utils import *

from utils.upload_manager import *


def test_loop(args, pipe, device, test_data_loader):
    
    if test_data_loader is None:
        return
    
    print('testing starts.....')
    
    pipe.model.eval()
    
    if get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        
        def _lm_pred_func(x, y):
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits = x[:, :-1, :].contiguous().float()
            labels = y[:, 1:].contiguous()
            loss = loss_fct(logits.transpose(-1, -2), labels).mean(1).detach().cpu()
            return loss
        
        loss_list = []
        for i, data in enumerate(test_data_loader):
            
            if args.evaluation_num_batch is not None and i >= args.evaluation_num_batch:
                break
                
            input_ids = data['input_ids'].to(device)
            labels = input_ids.clone()
            pipe.infer_iter(input_ids, labels, output_=loss_list, pred_func=_lm_pred_func)
            
        loss = torch.tensor(loss_list).mean()
        ppls = torch.exp(loss)
        metric = {"valid.perplexity": ppls.item(), "valid.loss": loss.item()}
        
        print(metric)
        train_log(
            metric, 
            step=pipe.global_step,
        )
        
    else:
        for i, data in enumerate(test_data_loader):
            
            if args.evaluation_num_batch is not None and i >= args.evaluation_num_batch:
                break
            
            input_ids = data['input_ids'].to(device)
            labels = input_ids.clone()
            current_iter_time = pipe.infer_iter(input_ids, labels)
    
    pipe.model.train()
    


def train_loop(args, pipe, device, train_data_loader, test_data_loader, steps_per_epoch):
    
    print('training starts......')

    event_reporter = EventReporter(host=args.event_host, auth_token=args.event_auth_token, job_id=args.job_id)

    pipe.model.train() # Flag .training to True to enable Dropout
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        # dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1
    pp_comm = get_pipeline_parallel_comm()
    
    stop_flag = torch.zeros(1, dtype=torch.int64).to(device)
    
    input_ids = torch.zeros(
        [args.batch_size, args.seq_length], 
        dtype=torch.int64
    ).to(device)
    
    do_sync_before_save = (args.dp_mode in ['local'] and use_dp)

    # Get the number of model parameters for the model
    param_count = torch.zeros(1, dtype=torch.int64).to(device)
    local_param_count = sum(p.numel() for p in pipe.model.parameters())
    param_count.data[:] = local_param_count
    pp_comm.reduce(param_count, 0)

    if get_pipeline_parallel_rank() == 0 and dp_rank == 0:

        upload_checkpoints_enabled = args.checkpoint_upload_prefix is not None 
        upload_manager = UploadManager(aws_endpoint_url = args.aws_endpoint_url,
                                       aws_access_key_id = args.aws_access_key_id,
                                       aws_secret_access_key = args.aws_secret_access_key,
                                       aws_session_token = args.aws_session_token,
                                       aws_region = args.aws_region,
                                       event_reporter = event_reporter,
                                       n_stages = args.pipeline_group_size)

        if event_reporter is not None:

            # Get the number of tokens in the dataset
            token_count = train_data_loader.dataset.get_dataset_token_count()

            # Report training start
            event_reporter.report(object=EventReporter.OBJECT_FINE_TUNE,
                                  message=f"Training started for model {args.model_name}",
                                  event_type=EventReporter.EVENT_TYPE_TRAINING_START,
                                  param_count=param_count.item(),
                                  token_count=token_count,
                                  requires_is_enabled=False)
        
        for i, data in enumerate(train_data_loader):
            # if i < pipe.global_step:
            #     continue
                
            if use_dp:
                get_data_parallel_comm().broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)
            
            if stop_flag.item() == 1:
                break
            
            input_ids_global = data['input_ids'].to(torch.int64).to(device)
            
            input_ids_list = input_ids_global.chunk(dp_size)
            
            if use_dp:
                for j in range(1, dp_size):
                    get_data_parallel_comm().send(
                        input_ids_list[j], j,
                    )
                
            input_ids = input_ids_list[0]
            
            pp_comm.broadcast(input_ids, 0)
            
            labels = input_ids.clone()
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func)

            if event_reporter is not None and (pipe.global_step >= args.total_steps or pipe.global_step % steps_per_epoch == 0):
                event_reporter.report(object=EventReporter.OBJECT_FINE_TUNE,
                                      message=f"Epoch completed, at step {pipe.global_step}",
                                      event_type=EventReporter.EVENT_TYPE_EPOCH_COMPLETE,
                                      requires_is_enabled=False)
            
            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)
            
            if pipe.global_step >= args.total_steps or pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    checkpoint_step_path = save_checkpoint(pipe, args)
                    if upload_checkpoints_enabled:
                        upload_manager.add_task(directory=checkpoint_step_path,
                                                checkpoint_upload_prefix=args.checkpoint_upload_prefix,
                                                step=pipe.global_step)

                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
            
            if pipe.global_step >= args.total_steps:
                stop_flag.data[:] = 1
        
        if upload_checkpoints_enabled:
            upload_manager.wait()
            
    elif get_pipeline_parallel_rank() == 0:
        
        while True:
            
            get_data_parallel_comm().broadcast(stop_flag, 0)
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
                
            get_data_parallel_comm().recv(
                input_ids, 0,
            )
            pp_comm.broadcast(input_ids, 0)
            
            labels = input_ids.clone()
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func)
            
            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)
                
            if pipe.global_step >= args.total_steps or pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
            
            
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        
        while True:
            
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
                
            pp_comm.broadcast(input_ids, 0)
            labels = input_ids.clone()
            current_iter_time = pipe.sgd_iter(input_ids, labels, loss_func=gpt_loss_func) # lm loss func
            
            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)
                
            if pipe.global_step >= args.total_steps or pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                    pipe.save_on_disk(args.checkpoint_path)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()
    else:
        while True:
            pp_comm.broadcast(stop_flag, 0)
            if stop_flag.item() == 1:
                break
            pp_comm.broadcast(input_ids, 0)
            current_iter_time = pipe.sgd_iter(None, None)
            
            if args.evaluation_steps > 0 and pipe.global_step % args.evaluation_steps == 0:
                test_loop(args, pipe, device, test_data_loader)
                
            if pipe.global_step >= args.total_steps or pipe.global_step % args.checkpoint_steps == 0:
                if do_sync_before_save:
                    pipe.dp_optim.allreduce_parameters()
                if dp_rank == 0:
                    save_checkpoint(pipe, args)
                if do_sync_before_save:
                    pipe.dp_optim.rollback_parameters()

# Compute the total number of training steps, steps per epoch, and steps per
# checkpoint
def calculate_training_steps(args, train_data_loader) -> int:
    print(f"args.total_steps={args.total_steps}")
    print(f"args.nepochs={args.nepochs}")
    print(f"args.checkpoint_steps={args.checkpoint_steps}")
    print(f"args.num_checkpoints={args.num_checkpoints}")

    total_steps = 0
    steps_per_epoch = 0
    steps_per_checkpoint = 0

    token_count = train_data_loader.dataset.get_dataset_token_count()

    # Check the inputs to calculate the total steps
    if args.batch_size is None or args.world_size is None or args.pipeline_group_size is None or token_count is None or args.seq_length is None:
        print("Missing required arguments for calculating total steps based on epochs.")
        sys.exit(1)

    global_batch_size = (args.batch_size * args.world_size + args.pipeline_group_size - 1) // args.pipeline_group_size
    tokens_per_batch = global_batch_size * args.seq_length
    steps_per_epoch = (token_count + tokens_per_batch - 1) // tokens_per_batch

    if args.total_steps is not None:
        if args.nepochs is not None:
            print("WARNING: total_steps ({args.toal_steps}) supercedes nepochs ({args.nepochs}).")
        total_steps = args.total_steps
    elif args.nepochs is not None:
        total_steps = steps_per_epoch * args.nepochs
    else:
        total_steps = len(train_data_loader)

    # Set the minimum number of total steps
    if total_steps < 10:
        total_steps = 10

    # Ensure that the steps per epoch are consistent with total steps
    # Note: This does not strictly follow the definition of an epoch. It just
    # approximately distributes the reporting of epochs over the total number of
    # steps.
    if args.nepochs is not None:
        steps_per_epoch = (total_steps + args.nepochs - 1) // args.nepochs

    # clamp steps_per_epoch to [1, total_steps]
    if steps_per_epoch > total_steps:
        steps_per_epoch = total_steps
    if steps_per_epoch < 1:
        steps_per_epoch = 1

    # Set the number of steps per epoch based on user input.
    if args.checkpoint_steps is not None and args.checkpoint_steps > 0:
        steps_per_checkpoint = args.checkpoint_steps
    elif args.num_checkpoints is not None and args.num_checkpoints > 0:
        steps_per_checkpoint = (total_steps + args.num_checkpoints - 1) // args.num_checkpoints
    else:
        steps_per_checkpoint = total_steps
    
    # Clamp steps_per_checkpoint to [1, total_steps]
    if steps_per_checkpoint > total_steps:
        steps_per_checkpoint = total_steps
    if steps_per_checkpoint < 1:
        steps_per_checkpoint = 1

    print(f"total_steps={total_steps}")
    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"steps_per_checkpoint={steps_per_checkpoint}")

    # Set the args base on what we computed above
    args.total_steps = total_steps
    args.checkpoint_steps = steps_per_checkpoint
    return steps_per_epoch

def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    add_entry_reporter_arguments(parser)
    parser.add_argument('--model-name', type=str, default='gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', metavar='S',
                        help='tokenizer name or path')
    parser.add_argument('--model-type', type=str, default='gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--checkpoint-path', type=str, default='model_checkpoints/gpt2')
    parser.add_argument('--task-name', type=str, default='cot', metavar='S',
                        help='task name')
    parser.add_argument('--warmup-steps', type=int, default=0, help='-')
    parser.add_argument('--train-warmup-steps', type=int, default=0, help='-')
    parser.add_argument('--nepochs', type=int, default=None, help='-')
    parser.add_argument('--total-steps', type=int, default=None, help='-')
    parser.add_argument('--load-pretrained-model', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--load-checkpoint', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='no-profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--evaluation-steps', 
                        type=int, default=0, metavar='S',
                        help='every x steps, do evaluation. (0 means do not do evaluation)')
    parser.add_argument('--evaluation-data',
                        type=str, default=None, help="path of eval data in jsonl")
    parser.add_argument('--evaluation-num-batch',
                        type=int, default=None, help="for debug purpose, only eval the first several batch.")
    parser.add_argument('--checkpoint-steps', 
                        type=int, default=0, metavar='S',
                        help='every x steps, save checkpoint. (0 means do not save checkpoint)')
    parser.add_argument('--num-checkpoints', 
                        type=int, default=0, metavar='S',
                        help='number of checkpoints to save')
    parser.add_argument('--net-interface', 
                        type=str, default='lo', metavar='S',
                        help='net_interface')
    parser.add_argument('--job-id', 
                        type=str, default="0", metavar='S',
                        help='an uuid')
    
    # Add AWS arguments for uploading checkpoints to S3
    parser.add_argument('--checkpoint-upload-prefix', required=True, help='S3 bucket name')
    add_aws_arguments(parser)

    args = parser.parse_args()
    aws_process_args(args)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
        
    init_communicators(args)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        dp_comm = get_data_parallel_comm()
        dp_rank = get_data_parallel_rank()
        dp_size = get_data_parallel_world_size()
    else:
        dp_rank = 0
        dp_size = 1
    
    config = AutoConfig.from_pretrained(args.model_name)
    
    # num layer globally
    if hasattr(config, 'num_hidden_layers'):
        args.max_layers = config.num_hidden_layers
    elif hasattr(config, 'num_layers'):
        args.max_layers = config.num_layers 
    else:
        args.max_layers = config.n_layer
    
    tokenizer = build_tokenizer(args)
    tokenizer.model_max_length = args.seq_length
    # config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    print("token vocab size:", config.vocab_size)
    
    train_data_loader = get_train_data_loader(args, tokenizer)
        
    if args.evaluation_data is not None and dp_rank == 0:
        test_data_loader = get_eval_data_loader(args, tokenizer)
    else:
        test_data_loader = None
    
    # calculate total steps
    steps_per_epoch = calculate_training_steps(args, train_data_loader)
    
    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")
    
    pipe = get_pp_module(args, config, device, use_dp)
    
    if args.load_checkpoint:
        load_checkpoint(pipe, args)

    if args.fp16:
        pipe.optimizer.reload_model_params()

    if args.profiling == 'no-profiling':
        train_loop(args, pipe, device, train_data_loader, test_data_loader, steps_per_epoch)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            try:
                train_loop(args, pipe, device, train_data_loader, test_data_loader, steps_per_epoch)
            except Exception as e:
                raise e
                print(get_pipeline_parallel_rank(), e)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                train_loop(args, pipe, device, train_data_loader, test_data_loader, steps_per_epoch)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
    print(get_pipeline_parallel_rank(), 'finished.')

if __name__ == '__main__':
    main()
