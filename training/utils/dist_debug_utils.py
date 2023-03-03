import torch


def print_cuda_memory(args, info: str, device=None):
    if args.debug_mem:
        if device is None:
            device = torch.device('cuda', args.cuda_id)
        print("<{}>: current memory allocated: {:2.3f} MB, peak memory: {:2.3f} MB".format(
            info, torch.cuda.memory_allocated(device)/1048576, torch.cuda.max_memory_allocated(device)/1048576))


def print_multi_cuda_memory(args, info: str):
    if args.debug_mem:
        for local_gpu_rank in range(args.cuda_num):
            device = torch.device('cuda', local_gpu_rank)
            print("<{}>({}): current memory allocated: {:2.3f} MB, peak memory: {:2.3f} MB".format(info, local_gpu_rank,
                  torch.cuda.memory_allocated(device)/1048576, torch.cuda.max_memory_allocated(device)/1048576))
