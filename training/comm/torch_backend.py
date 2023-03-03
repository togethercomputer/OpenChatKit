import torch
import torch.distributed as dist
from typing import List

class TorchCommunicator:
        
    def __init__(self,
                 process_group,
                 to_global_rank=lambda rank: rank,
                 dp_rank=None,
                 comm_group_size=None,):
        self.process_group = process_group
        self.to_global_rank = to_global_rank
        self.dp_rank = dp_rank
        self.comm_group_size = comm_group_size

    # @staticmethod
    def barrier(self):
        dist.barrier(group=self.process_group)

    def send(self,
             tensor: torch.Tensor,
             dst: int,
             stream=None):
        # print("Send tensor of size:", torch.numel(tensor))
        if tensor.device == torch.device('cpu'):
            dist.send(tensor, self.to_global_rank(dst), group=self.process_group)
        else:
            dist.send(tensor.cpu(), self.to_global_rank(dst), group=self.process_group)
            
    def recv(self,
             tensor: torch.Tensor,
             src: int,
             stream=None):
        
        if tensor.device == torch.device('cpu'):
            dist.recv(tensor, self.to_global_rank(src), group=self.process_group)
        else:
            buffer = tensor.cpu()
            dist.recv(buffer, self.to_global_rank(src), group=self.process_group)
            tensor[:] = buffer.to(tensor.device)
    
    def isend(self,
             tensor: torch.Tensor,
             dst: int,
             stream=None):
        # print("Send tensor of size:", torch.numel(tensor))
        if tensor.device == torch.device('cpu'):
            handler = dist.isend(tensor, self.to_global_rank(dst), group=self.process_group)
        else:
            handler = dist.isend(tensor.cpu(), self.to_global_rank(dst), group=self.process_group)
        return handler

    def irecv(self,
             tensor: torch.Tensor,
             src: int,
             stream=None):
        if tensor.device == torch.device('cpu'):
            handler = dist.irecv(tensor, self.to_global_rank(src), group=self.process_group)
        else:
            assert False
            buffer = tensor.cpu()
            handler = dist.irecv(buffer, self.to_global_rank(src), group=self.process_group)
            tensor[:] = buffer.to(tensor.device)
        return handler

    def broadcast(self,
                  tensor: torch.Tensor,
                  src: int,
                  stream=None):
        if tensor.device == torch.device('cpu'):
            dist.broadcast(tensor, self.to_global_rank(src), group=self.process_group)
        else:
            buffer = tensor.cpu()
            dist.broadcast(buffer, self.to_global_rank(src), group=self.process_group)
            tensor[:] = buffer.to(tensor.device)

    def reduce(self,
               tensor: torch.Tensor,
               dst: int,
               stream=None,
               op=dist.ReduceOp.SUM):
        dist.reduce(tensor, self.to_global_rank(dst), group=self.process_group, op=op)

    def all_reduce(self,
                   tensor: torch.Tensor,
                   stream = None,
                   op=dist.ReduceOp.SUM):
        buffer = tensor.cpu()
        dist.all_reduce(buffer, group=self.process_group, op=op)
        tensor[:] = buffer.to(tensor.device)

    def gather(self,
               tensor: torch.Tensor,
               gather_list: List[torch.Tensor],
               dst: int,
               stream=None):
        dist.gather(tensor, gather_list, self.to_global_rank(dst), group=self.process_group)

    def all_to_all(self,
                   output_tensor_list: List[torch.Tensor],
                   input_tensor_list: List[torch.Tensor],
                   stream=None):
        dist.all_to_all(output_tensor_list, input_tensor_list, group=self.process_group)

    def all_gather(self,
                   tensor: torch.Tensor,
                   output_tensor_list: List[torch.Tensor],
                   stream=None):
        dist.all_gather(output_tensor_list, tensor, group=self.process_group)
        
