import time
import json
import torch.nn.functional
from torch import optim
from comm.comm_utils import *
from modules.dist_gpt_pp_module import *
from data_parallel.dist_dp_utils import get_dp_module
from optimizer.optimizer import get_fp16_optimizer
import os
import cupy
import wandb
from transformers import get_linear_schedule_with_warmup

flag_profile = int(os.environ.get('FLAG_BENCHMARK', '0'))

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def create_optimizer(model, optimizer_type, weight_decay=0.01, learning_rate=2e-5,
                     adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-6):
    
    if optimizer_type == 'adamw' or optimizer_type == 'adam':
        from torch.optim import AdamW
        print('>>>>> using Adam')
    elif optimizer_type == '8bit-adam':
        from bitsandbytes.optim import Adam8bit as AdamW
        print('>>>>> using 8bit-Adam')
    else:
        assert False
    
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [
        name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
            "weight_decay": 0.0,
        }
    ]
    optimizer_cls = AdamW
    optimizer_kwargs = {
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }
    optimizer_kwargs["lr"] = learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


class GpipeAsync:
    r"""
    Async implementation of Gpipe.
    The current implementation leave the computation on the PyTorch default stream and the communication on a different
    stream, there is:
        a group of events to check if recv (from rank i-1) finishes in the forward propagation;
        a group of events to check if recv (from rank i+1) finishes in the backward propagation;
        a group of events to check if computation finishes in the forward propagation;
        a group of events to check if computation finishes in the backward propagation.
    """

    def __init__(self, args, config, device, use_dp=False,
                 _StageFull=GPTStageFull,
                 _StageFirst=GPTStageFirst,
                 _StageLast=GPTStageLast,
                 _StageMiddle=GPTStageMiddle):
        print("=======Initialize Gpipe.")
        if args.fp16:
            self.use_fp16 = True
            self.use_dynamic_scale = (args.loss_scale == 0)
            print("=======Gpipe use FP16")
        else:
            self.use_fp16 = False
            print("=======Gpipe use FP32")
        self.use_dp = use_dp
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        self.global_rank = args.rank
        self.pipeline_group_size = args.pipeline_group_size
        # Rank is the pipeline rank by default.
        self.pp_rank = get_pipeline_parallel_rank()
        if use_dp:
            self.dp_rank = get_data_parallel_rank()
        else:
            self.dp_rank = 0
        self.pre_node_rank = self.pp_rank - 1
        self.post_node_rank = self.pp_rank + \
            1 if self.pp_rank != self.pipeline_group_size - 1 else -1
        self.comm = get_pipeline_parallel_comm()
        self.gradient_accumulate_step = args.gradient_accumulate_step
        print("=======Gradient accumulate step: ",
              self.gradient_accumulate_step)

        assert (args.batch_size % args.micro_batch_size == 0)
        self.micro_batch_num = args.batch_size // args.micro_batch_size
        self.micro_batch_size = args.micro_batch_size
        self.seq_length = args.seq_length
        self.embedding_dim = args.embedding_dim
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_classes = config.num_labels

        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.device = device
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_recv_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_send_stream = torch.cuda.Stream(device=device, priority=-1)

        self.forward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]
        self.forward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                          for _ in range(self.micro_batch_num)]

        self.backward_recv_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]
        self.backward_comp_ready_events = [torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
                                           for _ in range(self.micro_batch_num)]

        if self.enable_tidy_profiling:
            self.profiling_log = []
            self.forward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                              for _ in range(self.micro_batch_num)]
            self.forward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                            for _ in range(self.micro_batch_num)]

            self.backward_recv_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_comp_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_start_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                               for _ in range(self.micro_batch_num)]
            self.backward_send_end_events = [torch.cuda.Event(enable_timing=True, blocking=False)
                                             for _ in range(self.micro_batch_num)]
            self.init_event = torch.cuda.Event(
                enable_timing=True, blocking=False)
            self.init_time_stamp = None
            self.optimizer_start_event = torch.cuda.Event(
                enable_timing=True, blocking=False)
            self.optimizer_end_event = torch.cuda.Event(
                enable_timing=True, blocking=False)

        self._compute_micro_batch_size()

        if hasattr(args, 'infer_only') and args.infer_only:
            do_train = False
        else:
            do_train = True

        if self.pp_rank == 0:
            self.input_micro_batches = None
        else:
            self.input_micro_batches = [
                torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                            requires_grad=do_train, device=self.device, dtype=self.dtype
                            ) for _ in range(self.micro_batch_num)
            ]

        if do_train:
            if self.pp_rank == self.pipeline_group_size - 1:

                if not hasattr(args, 'project_name'):
                    import re
                    args.project_name = "test-" + \
                        re.sub('[^a-zA-Z0-9 \n\.]', '_', args.task_name)

                wandb.init(
                    project=args.project_name, 
                    # entity='pipeline-activation-compression',
                    config=args,
                )

            if self.pp_rank == self.pipeline_group_size - 1:
                self.output_micro_batches_grad = None
            else:
                self.output_micro_batches_grad = [
                    torch.zeros((self.micro_batch_size, self.seq_length, self.embedding_dim),
                                requires_grad=False, device=self.device, dtype=self.dtype
                                ) for _ in range(self.micro_batch_num)
                ]

        if self.pipeline_group_size > 1:
            if self.pp_rank == 0:
                self.model = _StageFirst(args, config, device)
            elif self.pp_rank == self.pipeline_group_size - 1:
                self.model = _StageLast(args, config, device)
            else:
                self.model = _StageMiddle(args, config, device)
        else:
            self.model = _StageFull(args, config, device)

        if self.use_fp16:
            self.model.half()

        if do_train:
            if self.use_fp16:
                tmp_optimizer = create_optimizer(
                    self.model, optimizer_type=getattr(args, 'optimizer', 'adamw'), learning_rate=args.lr)
                self.optimizer = get_fp16_optimizer(
                    args, tmp_optimizer, device)
                self.scheduler = get_linear_schedule_with_warmup(
                    tmp_optimizer, args.warmup_steps, args.total_steps, )
            else:
                self.optimizer = create_optimizer(
                    self.model, optimizer_type=getattr(args, 'optimizer', 'adamw'), learning_rate=args.lr)
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, args.warmup_steps, args.total_steps, )

            # Notice that if we use fp16, gradients are aggregated in fp16, this may not be the default in Megatron.
            if use_dp:
                self.dp_optim = get_dp_module(
                    args, device, self.model, self.optimizer)

        self.global_step = 0

    def _compute_micro_batch_size(self):
        micro_batch_float_num = self.micro_batch_size * \
            self.seq_length * self.embedding_dim
        if self.use_fp16:
            print("=======Current micro-batch send/recv size: {} MB (fp16)"
                  .format(micro_batch_float_num * 2 // 1024 // 1024))
        else:
            print("=======Current micro-batch send/recv size: {} MB (fp32)"
                  .format(micro_batch_float_num*4//1024//1024))
        print("=======Number of micro-batches: {}.".format(self.micro_batch_num))

    def zero_input_grad(self):
        if self.input_micro_batches:
            for input_micro_batch in self.input_micro_batches:
                if input_micro_batch.grad is not None:
                    input_micro_batch.grad.zero_()

    def profile_mark_forward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(
                self.forward_comp_start_events[i])

    def profile_mark_forward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(
                self.forward_recv_start_events[i])

    def profile_mark_forward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(
                self.forward_send_start_events[i])

    def profile_mark_forward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(
                self.forward_send_end_events[i])

    def profile_mark_backward_comp_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_comp_stream.record_event(
                self.backward_comp_start_events[i])

    def profile_mark_backward_recv_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_recv_stream.record_event(
                self.backward_recv_start_events[i])

    def profile_mark_backward_send_start(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(
                self.backward_send_start_events[i])

    def profile_mark_backward_send_end(self, i):
        if self.enable_tidy_profiling:
            self.torch_send_stream.record_event(
                self.backward_send_end_events[i])

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def forward_stage(self, input_data=None, aux_input_data=None):
        # print("Forward stage start! rank-", self.rank)

        if aux_input_data is not None:
            for k in aux_input_data:
                aux_input_data[k] = torch.chunk(
                    aux_input_data[k], self.micro_batch_num, dim=0)
        else:
            aux_input_data = {}

        if self.pp_rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(
                input_data, self.micro_batch_num, dim=0)
        if self.pp_rank == self.pipeline_group_size - 1:
            if input_data is not None:
                input_ids_micro_batches = torch.chunk(
                    input_data, self.micro_batch_num, dim=0)
            else:
                input_ids_micro_batches = [None]*self.micro_batch_num
        output_micro_batches = []

        for i in range(self.micro_batch_num):
            if self.pipeline_group_size > 1:
                if self.pp_rank == 0:  # Only send output to next node, do not receive
                    with torch.cuda.stream(self.torch_comp_stream):
                        self.profile_mark_forward_comp_start(i)
                        current_micro_output = self.model(
                            self.input_micro_batches[i],
                            **{k: v[i] for k, v in aux_input_data.items()}
                        )
                        self.torch_comp_stream.record_event(
                            self.forward_comp_ready_events[i])
                    with torch.cuda.stream(self.torch_send_stream):
                        cupy_send_stream = cupy.cuda.ExternalStream(
                            self.torch_send_stream.cuda_stream)
                        self.torch_send_stream.wait_event(
                            self.forward_comp_ready_events[i])
                        self.profile_mark_forward_send_start(i)
                        self.comm.send(current_micro_output.data,
                                       dst=self.post_node_rank, stream=cupy_send_stream)
                        self.profile_mark_forward_send_end(i)
                elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                    with torch.cuda.stream(self.torch_recv_stream):
                        cupy_recv_stream = cupy.cuda.ExternalStream(
                            self.torch_recv_stream.cuda_stream)
                        self.profile_mark_forward_recv_start(i)
                        self.comm.recv(
                            self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                        self.torch_recv_stream.record_event(
                            self.forward_recv_ready_events[i])
                    with torch.cuda.stream(self.torch_comp_stream):
                        self.torch_comp_stream.wait_event(
                            self.forward_recv_ready_events[i])
                        self.profile_mark_forward_comp_start(i)
                        current_micro_output = self.model(
                            self.input_micro_batches[i], input_ids=input_ids_micro_batches[i],
                            **{k: v[i] for k, v in aux_input_data.items()}
                        )
                        self.torch_comp_stream.record_event(
                            self.forward_comp_ready_events[i])
                else:  # receive, compute, and send
                    with torch.cuda.stream(self.torch_recv_stream):
                        cupy_recv_stream = cupy.cuda.ExternalStream(
                            self.torch_recv_stream.cuda_stream)
                        self.profile_mark_forward_recv_start(i)
                        self.comm.recv(
                            self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                        self.torch_recv_stream.record_event(
                            self.forward_recv_ready_events[i])
                    with torch.cuda.stream(self.torch_comp_stream):
                        self.torch_comp_stream.wait_event(
                            self.forward_recv_ready_events[i])
                        self.profile_mark_forward_comp_start(i)
                        current_micro_output = self.model(
                            self.input_micro_batches[i],
                            **{k: v[i] for k, v in aux_input_data.items()}
                        )
                        self.torch_comp_stream.record_event(
                            self.forward_comp_ready_events[i])
                    with torch.cuda.stream(self.torch_send_stream):
                        cupy_send_stream = cupy.cuda.ExternalStream(
                            self.torch_send_stream.cuda_stream)
                        self.torch_send_stream.wait_event(
                            self.forward_comp_ready_events[i])
                        self.profile_mark_forward_send_start(i)
                        self.comm.send(current_micro_output.data,
                                       dst=self.post_node_rank, stream=cupy_send_stream)
                        self.profile_mark_forward_send_end(i)
            else:
                with torch.cuda.stream(self.torch_comp_stream):
                    self.profile_mark_forward_comp_start(i)
                    current_micro_output = self.model(
                        self.input_micro_batches[i],
                        **{k: v[i] for k, v in aux_input_data.items()}
                    )
                    self.torch_comp_stream.record_event(
                        self.forward_comp_ready_events[i])

            output_micro_batches.append(current_micro_output)

        return output_micro_batches

    def profiling_forward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.pp_rank != 0:
                recv_slot = self.forward_recv_start_events[i].elapsed_time(
                    self.forward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "1. forward-recv",
                            "ts": self.get_ts(self.forward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}  # cname is for color, a little silly.
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.forward_comp_start_events[i].elapsed_time(
                self.forward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "2. forward-compute",
                        "ts": self.get_ts(self.forward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)

            if self.pp_rank != self.pipeline_group_size - 1:
                send_slot = self.forward_send_start_events[i].elapsed_time(
                    self.forward_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "3. forward-send",
                            "ts": self.get_ts(self.forward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def backward_stage(self, cached_output_micro_batches: List[torch.Tensor], target=None,
                       loss_func=torch.nn.functional.cross_entropy):
        # print("Backward stage start! rank-", self.rank)
        if self.pp_rank == self.pipeline_group_size - 1:
            assert(target is not None)
            target_as_micro_batches = torch.chunk(
                target, self.micro_batch_num, dim=0)
        # else:
        #     assert(target is None)

        if self.pp_rank == self.pipeline_group_size - 1:
            tr_loss = []

        for i in range(self.micro_batch_num):
            if self.pipeline_group_size > 1:
                if self.pp_rank == self.pipeline_group_size - 1:  # only send grad back to last node, do not receive
                    with torch.cuda.stream(self.torch_comp_stream) as st:
                        self.profile_mark_backward_comp_start(i)
                        loss = loss_func(
                            input=cached_output_micro_batches[i], target=target_as_micro_batches[i])
                        if not flag_profile:
                            tr_loss.append(loss.item())
                        if self.use_fp16:
                            self.optimizer.scale(loss).backward()
                        else:
                            loss.backward()
                        self.torch_comp_stream.record_event(
                            self.backward_comp_ready_events[i])
                    with torch.cuda.stream(self.torch_send_stream):
                        cupy_send_stream = cupy.cuda.ExternalStream(
                            self.torch_send_stream.cuda_stream)
                        self.torch_send_stream.wait_event(
                            self.backward_comp_ready_events[i])
                        self.profile_mark_backward_send_start(i)
                        self.comm.send(
                            self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_send_stream)
                        self.profile_mark_backward_send_end(i)
                elif self.pp_rank == 0:  # only receive grad from previous node, do not send
                    with torch.cuda.stream(self.torch_recv_stream):
                        cupy_recv_stream = cupy.cuda.ExternalStream(
                            self.torch_recv_stream.cuda_stream)
                        self.profile_mark_backward_recv_start(i)
                        self.comm.recv(
                            self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_recv_stream)
                        self.torch_recv_stream.record_event(
                            self.backward_recv_ready_events[i])
                    with torch.cuda.stream(self.torch_comp_stream):
                        self.torch_comp_stream.wait_event(
                            self.backward_recv_ready_events[i])
                        self.profile_mark_backward_comp_start(i)
                        cached_output_micro_batches[i].backward(
                            gradient=self.output_micro_batches_grad[i])
                        self.torch_comp_stream.record_event(
                            self.backward_comp_ready_events[i])
                else:  # receive, compute and send
                    with torch.cuda.stream(self.torch_recv_stream):
                        cupy_recv_stream = cupy.cuda.ExternalStream(
                            self.torch_recv_stream.cuda_stream)
                        self.profile_mark_backward_recv_start(i)
                        self.comm.recv(
                            self.output_micro_batches_grad[i], src=self.post_node_rank, stream=cupy_recv_stream)
                        self.torch_recv_stream.record_event(
                            self.backward_recv_ready_events[i])
                    with torch.cuda.stream(self.torch_comp_stream):
                        self.torch_comp_stream.wait_event(
                            self.backward_recv_ready_events[i])
                        self.profile_mark_backward_comp_start(i)
                        cached_output_micro_batches[i].backward(
                            gradient=self.output_micro_batches_grad[i])
                        self.torch_comp_stream.record_event(
                            self.backward_comp_ready_events[i])
                    with torch.cuda.stream(self.torch_send_stream):
                        cupy_send_stream = cupy.cuda.ExternalStream(
                            self.torch_send_stream.cuda_stream)
                        self.torch_send_stream.wait_event(
                            self.backward_comp_ready_events[i])
                        self.profile_mark_backward_send_start(i)
                        self.comm.send(
                            self.input_micro_batches[i].grad, dst=self.pre_node_rank, stream=cupy_send_stream)
                        self.profile_mark_backward_send_end(i)
            else:
                
                with torch.cuda.stream(self.torch_comp_stream) as st:
                    self.profile_mark_backward_comp_start(i)
                    loss = loss_func(
                        input=cached_output_micro_batches[i], target=target_as_micro_batches[i])
                    if not flag_profile:
                        tr_loss.append(loss.item())
                    if self.use_fp16:
                        self.optimizer.scale(loss).backward()
                    else:
                        loss.backward()
                    self.torch_comp_stream.record_event(
                        self.backward_comp_ready_events[i])

        if not flag_profile:
            if self.pp_rank == self.pipeline_group_size - 1:
                wandb.log(
                    {
                        'loss': sum(tr_loss)/len(tr_loss),
                        'lr': self.scheduler.get_last_lr()[0],
                        #                     'scale': self.optimizer.get_loss_scale(), ##todo
                    }, step=self.global_step,
                )
                print("logging...")
                if hasattr(self, 'experiment'):
                    self.experiment.log_metrics({
                        'loss': sum(tr_loss)/len(tr_loss),
                        'lr': self.scheduler.get_last_lr()[0],
                    }, step=self.global_step)

    def profiling_backward_stage(self):
        torch.cuda.synchronize()
        for i in range(self.micro_batch_num):
            if self.pp_rank != self.pipeline_group_size - 1:
                recv_slot = self.backward_recv_start_events[i].elapsed_time(
                    self.backward_recv_ready_events[i]) * 1e+3
                recv_log = {"name": "recv", "ph": "X", "pid": self.global_rank, "tid": "4. backward-recv",
                            "ts": self.get_ts(self.backward_recv_start_events[i]), "dur": recv_slot,
                            "args": {"micro-batch": i}, "cname": "startup"}
                # print(recv_log)
                self.profiling_log.append(recv_log)

            comp_slot = self.backward_comp_start_events[i].elapsed_time(
                self.backward_comp_ready_events[i]) * 1e+3
            comp_log = {"name": "comp", "ph": "X", "pid": self.global_rank, "tid": "5. backward-compute",
                        "ts": self.get_ts(self.backward_comp_start_events[i]), "dur": comp_slot,
                        "args": {"micro-batch": i}, "cname": "good"}
            # print(comp_log)
            self.profiling_log.append(comp_log)
            if self.pp_rank != 0:
                send_slot = self.backward_send_start_events[i].elapsed_time(
                    self.backward_send_end_events[i]) * 1e+3
                send_log = {"name": "send", "ph": "X", "pid": self.global_rank, "tid": "6. backward-send",
                            "ts": self.get_ts(self.backward_send_start_events[i]), "dur": send_slot,
                            "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
                # print(send_log)
                self.profiling_log.append(send_log)

    def save_on_disk(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'pytorch_model.bin'))
        
    def optimizer_step(self):
        # hard code: grad clipping
        if not self.use_fp16:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if self.use_dp:
            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.record_event(
                    self.dp_optim.backward_ready_event)
            self.dp_optim.optimizer_step()
            self.scheduler.step()
        else:
            with torch.cuda.stream(self.torch_comp_stream):
                if self.enable_tidy_profiling:
                    self.optimizer_start_event.record()
                self.optimizer.step()
                self.scheduler.step()
                if self.enable_tidy_profiling:
                    self.optimizer_end_event.record()
        if self.enable_tidy_profiling:
            self.profiling_optimizer_step()

    def profiling_optimizer_step(self):
        torch.cuda.synchronize()
        if not self.use_dp:
            optimizer_slot = self.optimizer_start_event.elapsed_time(
                self.optimizer_end_event) * 1e+3
            optimizer_log = {"name": "opt", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-step",
                             "ts": self.get_ts(self.optimizer_start_event), "dur": optimizer_slot, "cname": "bad"}
            # print(optimizer_log)
            self.profiling_log.append(optimizer_log)
        else:
            self.profiling_log.extend(self.dp_optim.profiling_data_parallel(
                self.init_time_stamp, self.init_event))

    def export_profiling_result(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.profiling_log, outfile)

    def sgd_iter(self, input_=None, target=None,
                 aux_input_data=None, loss_func=torch.nn.functional.cross_entropy):
        
        
        if self.use_fp16 and self.use_dynamic_scale:
            scales_buffer = [torch.ones_like(self.optimizer.grad_scaler._scale) for _ in range(self.pipeline_group_size)]
            self.comm.all_gather(self.optimizer.grad_scaler._scale, scales_buffer)
            self.optimizer.grad_scaler._scale.data[:] = min([s.item() for s in scales_buffer])
        
        self.comm.barrier()
            
        start_time = time.time()
        if self.enable_tidy_profiling:
            torch.cuda.synchronize()
            self.init_time_stamp = time.time() * 1e+6
            self.init_event.record()

        step = self.global_step % self.gradient_accumulate_step
        self.zero_input_grad()
        if step == 0:
            self.optimizer.zero_grad(set_to_none=False)
            
        if step == self.gradient_accumulate_step - 1 and self.use_dp:
            if hasattr(self.dp_optim, 'pre_optimizer_step'):
                self.dp_optim.pre_optimizer_step()

        outputs = self.forward_stage(input_, aux_input_data=aux_input_data)
        forward_time = time.time()
        forward_slot = forward_time-start_time
        print("Rank {} node forward pass {}/{} takes {:3.2f}s"
              .format(self.global_rank, step, self.gradient_accumulate_step, forward_slot))
        
        # This is an educated guess that such barrier would make it fair TC (probably required)
        # self.comm.barrier()
        self.backward_stage(outputs, target, loss_func=loss_func)
        backward_time = time.time()
        print("Rank {} node backward pass {}/{} takes {:3.2f}s"
              .format(self.global_rank, step, self.gradient_accumulate_step, backward_time-forward_time))
        if step == self.gradient_accumulate_step - 1:
            optimizer_time = time.time()
            self.optimizer_step()
            torch.cuda.synchronize()
            
            if self.enable_tidy_profiling:
                self.profiling_forward_stage()
                self.profiling_backward_stage()
            
            print('after cuda sync', self.global_rank)
            self.comm.barrier()
            end_time = time.time()
            print("Rank {} node optimizer step takes {:3.2f}s".format(
                self.global_rank, end_time - optimizer_time))
        else:
            self.comm.barrier()
            end_time = time.time()
        iter_time = end_time - start_time
        print("Rank {} node whole iteration takes {:3.2f}s".format(
            self.global_rank, iter_time))
        print("-------------------------------------------")
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
        self.global_step += 1
        return iter_time
    
    
    def infer_stage(self, input_data=None, aux_input_data=None, 
                    labels=None, pred_func=None):
        
        if aux_input_data is not None:
            for k in aux_input_data:
                aux_input_data[k] = torch.chunk(aux_input_data[k], self.micro_batch_num, dim=0)
        else:
            aux_input_data = {}
        
        if self.pp_rank == 0:
            assert(input_data is not None)
            self.input_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
        if self.pp_rank == self.pipeline_group_size - 1:
            if input_data is not None:
                input_ids_micro_batches = torch.chunk(input_data, self.micro_batch_num, dim=0)
            else:
                input_ids_micro_batches = [None]*self.micro_batch_num
            if labels is not None:
                labels = torch.chunk(labels, self.micro_batch_num, dim=0)
            else:
                labels = [None]*self.micro_batch_num
                
        output_micro_batches = []

        for i in range(self.micro_batch_num):
            if self.pipeline_group_size > 1:
                if self.pp_rank == 0:  # Only send output to next node, do not receive
                    with torch.cuda.stream(self.torch_comp_stream):
                        current_micro_output = self.model(
                            self.input_micro_batches[i], 
                            **{k: v[i] for k, v in aux_input_data.items()},
                        )
                        self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                    with torch.cuda.stream(self.torch_send_stream):
                        cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                        self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                        self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
                elif self.pp_rank == self.pipeline_group_size - 1:  # Only receive input from last node, do not send
                    with torch.cuda.stream(self.torch_recv_stream):
                        cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                        self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                        self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                    with torch.cuda.stream(self.torch_comp_stream):
                        self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                        current_micro_output = self.model(
                            self.input_micro_batches[i], input_ids=input_ids_micro_batches[i],
                            **{k: v[i] for k, v in aux_input_data.items()},
                        )
                        current_micro_output = pred_func(current_micro_output, labels[i])
                        self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                else:  # receive, compute, and send
                    with torch.cuda.stream(self.torch_recv_stream):
                        cupy_recv_stream = cupy.cuda.ExternalStream(self.torch_recv_stream.cuda_stream)
                        self.comm.recv(self.input_micro_batches[i], src=self.pre_node_rank, stream=cupy_recv_stream)
                        self.torch_recv_stream.record_event(self.forward_recv_ready_events[i])
                    with torch.cuda.stream(self.torch_comp_stream):
                        self.torch_comp_stream.wait_event(self.forward_recv_ready_events[i])
                        current_micro_output = self.model(
                            self.input_micro_batches[i],
                            **{k: v[i] for k, v in aux_input_data.items()},
                        )
                        self.torch_comp_stream.record_event(self.forward_comp_ready_events[i])
                    with torch.cuda.stream(self.torch_send_stream):
                        cupy_send_stream = cupy.cuda.ExternalStream(self.torch_send_stream.cuda_stream)
                        self.torch_send_stream.wait_event(self.forward_comp_ready_events[i])
                        self.comm.send(current_micro_output.data, dst=self.post_node_rank, stream=cupy_send_stream)
            else:
                with torch.cuda.stream(self.torch_comp_stream):
                    current_micro_output = self.model(
                        self.input_micro_batches[i],
                        **{k: v[i] for k, v in aux_input_data.items()}
                    )
                    current_micro_output = pred_func(current_micro_output, labels[i])
                    self.torch_comp_stream.record_event(
                        self.forward_comp_ready_events[i])
                    
            output_micro_batches.append(current_micro_output)
            
        return output_micro_batches
    
    def infer_iter(self, input_=None, target=None, 
                   output_=None, 
                   aux_input_data=None, 
                   pred_func=None):
        # self.comm.barrier()
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = self.infer_stage(input_, 
                                       aux_input_data=aux_input_data,
                                       labels=target, pred_func=pred_func)
            if output_ is not None:
                outputs = torch.cat(outputs, 0).mean().item()
                print(outputs)
                output_.append(outputs)
        torch.cuda.synchronize()
        # self.comm.barrier()

