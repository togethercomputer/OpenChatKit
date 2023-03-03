import os

try:
    import wandb
    _has_wandb = True
except:
    _has_wandb = False
    print("wandb is not installed.")
    
try:
    import loguru
    _has_loguru = True
except:
    _has_loguru = False
    print("loguru is not installed.")
    
train_log_backend = None
    
def init_train_logger(args):
    
    global train_log_backend
    train_log_backend = getattr(args, 'train_log_backend', 'print')
    
    if train_log_backend == 'print':
        pass
    elif train_log_backend == 'loguru':
        os.system("mkdir -p logs")
        loguru.logger.add("logs/file_{time}.log")
    elif train_log_backend == 'wandb':
        
        assert _has_wandb
        
        if not hasattr(args, 'project_name'):
            import re
            args.project_name = "test-" + \
                re.sub('[^a-zA-Z0-9 \n\.]', '_', args.task_name)

        wandb.init(
            project=args.project_name, 
            config=args,
        )
        
    else:
        raise Exception('Unknown logging backend.')
        
def train_log(x, *args, **kargs):
    
    if train_log_backend == 'print':
        print(x)
    elif train_log_backend == 'loguru':
        loguru.logger.info(x)
    elif train_log_backend == 'wandb':
        wandb.log(x, *args, **kargs)
    else:
        raise Exception('Unknown logging backend.')
    
    