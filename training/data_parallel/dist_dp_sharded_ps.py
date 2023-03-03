import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from comm.comm_utils import *
from .flatten_utils import flatten_params


class ShardedPSDP:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None, flatten=True):
        self.flatten = flatten
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)

        self.module = module
        assert optimizer is not None
        self.optimizer = optimizer
        num_paras, element_size = self._compute_total_para_num()
        print("Total number of parameters: {}, element size: {}, total size {} MB."
              .format(num_paras, element_size, num_paras * element_size // 1024 // 1024))

        assert self.flatten
#         self.para = list(self.module.parameters())
        self.flatten_para = flatten_params(self.module.parameters(), self.dp_group_size)
        print("Flattened parameter number: {}, element size: {}."
              .format(self.flatten_para.data.numel(), self.flatten_para.data.element_size()))
        print("Flattened parameter grad number: {}, element size: {}."
              .format(self.flatten_para.grad.numel(), self.flatten_para.grad.element_size()))

        self.grad_buffer = self._declare_grad_buffer()

        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None

            assert self.flatten
            self.sync_gradients_start_event = torch.cuda.Event(enable_timing=True, blocking=False)

            self.optimizer_step_start_event = torch.cuda.Event(enable_timing=True, blocking=False)

    def _compute_total_para_num(self):
        total_count = 0
        element_size = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
            element_size = para.element_size()
        return total_count, element_size

    def _declare_grad_buffer(self):
        assert self.flatten_para.data.numel() % self.dp_group_size == 0
        chunk_size = self.flatten_para.data.numel() // self.dp_group_size
        grad_buffer = [torch.zeros(chunk_size, device=self.flatten_para.device, dtype=self.flatten_para.dtype)
                       for _ in range(self.dp_group_size)]
        return grad_buffer

    def profile_mark_sync_grad_start(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.sync_gradients_start_event)

    def profile_mark_allreduce_end(self):
        pass

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling:
            self.torch_optim_comp_stream.record_event(self.optimizer_step_start_event)

    def _sync_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            assert self.flatten
            self.profile_mark_sync_grad_start()
            self.dp_comm.all_reduce_opt(self.flatten_para.grad, self.grad_buffer, stream=cupy_dp_stream)
            self.profile_mark_allreduce_end()
            self.dp_comm_stream.record_event(self.sync_gradients_ready_event)

    def optimizer_step(self):
        self._sync_gradients()
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.sync_gradients_ready_event)
            self.profile_mark_optimizer_step_start()
            self.optimizer.step()
            self.torch_optim_comp_stream.record_event(self.optimizer_step_ready_event)

    def set_time_stamp(self, init_time_stamp, init_event):
        self.init_event = init_event
        self.init_time_stamp = init_time_stamp

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def profiling_data_parallel(self, init_time_stamp, init_event):
        self.set_time_stamp(init_time_stamp, init_event)
        profiling_log = []

        assert self.flatten
        allreduce_slot = self.sync_gradients_start_event.elapsed_time(self.sync_gradients_ready_event)*1e+3
        allreduce_log = {"name": "opt_shardedPS_sync", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                         "ts": self.get_ts(self.sync_gradients_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {"name": "opt_comp", "ph": "X", "pid": self.global_rank, "tid": "8. optimizer-comp",
                         "ts": self.get_ts(self.optimizer_step_start_event), "dur": optimizer_slot, "cname": "bad"}
        # print(optimizer_log)
        profiling_log.append(optimizer_log)
        return profiling_log
