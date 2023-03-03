from .dist_dp_allreduce import AllReduceDP
from .dist_dp_sharded_ps import ShardedPSDP
from .dist_dp_local import LocalDP


def get_dp_module(args, device, module, optimizer):
    print("Data parallel implementation: ", args.dp_mode)
    if args.dp_mode == 'allreduce':
        return AllReduceDP(args, device, module, optimizer, flatten=False) 
        # flatten gradient is not compatible with fp16 now
    elif args.dp_mode == 'local':
        return LocalDP(args, device, module, optimizer, flatten=False)
    elif args.dp_mode == 'sharded_ps':
        return ShardedPSDP(args, device, module, optimizer, flatten=False)
    else:
        print("Not recognize this data parallel mode.")
        assert False
