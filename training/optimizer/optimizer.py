import torch
from .grad_scalar import *

# This follows some implementation from Megatron


def _has_overflow_serial(grads):

    def _has_inf_or_nan(x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
                return True
            return False

    for p in grads:
        if _has_inf_or_nan(p):
            return torch.FloatTensor([1.0])

    return torch.FloatTensor([0.0])


# `x` is a torch.Tensor



def _zero_grad_group(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


'''
def _multi_tensor_copy_this_to_that(this, that):
    for this_, that_ in zip(this, that):
        that_.copy_(this_)
'''


class Fp16Optimizer:
    # If offload is set to true, the fp32 copy is stored on CPU.
    def __init__(self, optimizer, grad_scaler, device, offload=False):
        self.offload = offload
        if self.offload:
            self.cpu_to_gpu_stream = torch.cuda.Stream(device=device, priority=-1)
            self.gpu_to_cpu_stream = torch.cuda.Stream(device=device, priority=-1)
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0], device=device) if not self.offload else torch.FloatTensor([0.0])

        self._dummy_overflow_buf = torch.cuda.IntTensor([0], device=device) if not self.offload else torch.IntTensor([0])

        # Note that the model should first be cast to fp16 before passing to the optimizer.
        self.float16_groups = []
        self.fp32_from_float16_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    # float16 params:
                    assert param.type() == 'torch.cuda.HalfTensor'
                    float16_params_this_group.append(param)
                    # Create a copy
                    if self.offload:
                        optimizer_param = param.detach().clone().float().to(device='cpu')
                        assert optimizer_param.device == torch.device('cpu')
                        if optimizer_param.grad is None:
                            optimizer_param.grad = torch.zeros_like(optimizer_param.data)
                    else:
                        optimizer_param = param.detach().clone().float()
                    # Replace the optimizer params with the new fp32 copy.
                    param_group['params'][i] = optimizer_param
                    fp32_from_float16_params_this_group.append(optimizer_param)
                    # Reset existing state dict key to the new optimizer param.
                    if param in self.optimizer.state:
                        self.optimizer.state[optimizer_param] = self.optimizer.state.pop(param)

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def zero_grad(self, set_to_none=True):
        for group in self.float16_groups:
            _zero_grad_group(group, set_to_none)
        if not self.offload:
            for group in self.fp32_from_float16_groups:
                _zero_grad_group(group, set_to_none)

    def get_loss_scale(self):
        return self.grad_scaler.scale

    def _copy_model_grads_to_optimizer_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, optimizer_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, optimizer_param in zip(model_group, optimizer_group):
                if model_param.grad is not None:
                    if self.offload:
                        with torch.cuda.stream(self.gpu_to_cpu_stream):
                            optimizer_param.grad.copy_(model_param.grad, non_blocking=False)
                    else:
                        optimizer_param.grad = model_param.grad.float()
                # Safe to deallocate model's grad/optimizer_grad after copying.
                # (If using contiguous buffers, optimizer_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

    def _unscale_optimizer_grads_and_check_for_nan(self):
        optimizer_grads = []
        # fp32 params fromm float16 ones.
        for optimizer_group in self.fp32_from_float16_groups:
            for optimizer_param in optimizer_group:
                if optimizer_param.grad is not None:
                    optimizer_grads.append(optimizer_param.grad.data)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        print(optimizer_grads[0].device, self.found_inf.device, self.grad_scaler.inv_scale.device)
        if self.offload:
            self.found_inf = _has_overflow_serial(optimizer_grads)
        else:
            torch._amp_foreach_non_finite_check_and_unscale_(optimizer_grads, self.found_inf, self.grad_scaler.inv_scale)
        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)
        return found_inf_flag

    def _get_model_and_optimizer_params_data_float16_deprecated(self):
        model_data = []
        optimizer_data = []
        for model_group, optimizer_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, optimizer_param in zip(model_group, optimizer_group):
                model_data.append(model_param.data)
                optimizer_data.append(optimizer_param.data)
        return model_data, optimizer_data

    def _copy_optimizer_params_to_model_params(self):
        # Only needed for the float16 params.
        # model_data, optimizer_data = self._get_model_and_optimizer_params_data_float16_deprecated()
        # _multi_tensor_copy_this_to_that(this=optimizer_data, that=model_data)

        for model_group, optimizer_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, optimizer_param in zip(model_group, optimizer_group):
                if self.offload:
                    with torch.cuda.stream(self.cpu_to_gpu_stream):
                        model_param.data.copy_(optimizer_param.data, non_blocking=False)
                else:
                    model_param.data.copy_(optimizer_param.data)

    def _copy_model_params_to_optimizer_params(self):
        # Only needed for the float16 params.
        # model_data, optimizer_data = self._get_model_and_optimizer_params_data_float16_deprecated()
        # _multi_tensor_copy_this_to_that(this=model_data, that=optimizer_data)
        for model_group, optimizer_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, optimizer_param in zip(model_group, optimizer_group):
                if self.offload:
                    with torch.cuda.stream(self.gpu_to_cpu_stream):
                        optimizer_param.data.copy_(model_param.data, non_blocking=False)
                else:
                    optimizer_param.data.copy_(model_param.data)

    def reload_model_params(self):
        self._copy_model_params_to_optimizer_params()

    @torch.no_grad()
    def step(self):
        self._copy_model_grads_to_optimizer_grads()

        found_inf_flag = self._unscale_optimizer_grads_and_check_for_nan()
        self.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            print("!!! Warning: find inf in fp16 optimizer-step() !!!")
            return False
        
        for params in self.fp32_from_float16_groups:
            torch.nn.utils.clip_grad_norm_(params, 1.0)

        # Step the optimizer.
        self.optimizer.step()

        self._copy_optimizer_params_to_model_params()
        # Successful update.
        return True
    
    def scale(self, z):
        return z * self.grad_scaler.scale
    
    def unscale(self, z):
        return z * self.grad_scaler.inv_scale
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def get_fp16_optimizer(args, optimizer, device):
    assert args.fp16 is not None
    if args.loss_scale:
        print("fp16 uses ConstantGradScaler.")
        grad_scaler = ConstantGradScaler(args.loss_scale)
    else:
        print("fp16 uses DynamicGradScaler.")
        grad_scaler = DynamicGradScaler(
            initial_scale=args.initial_loss_scale,
            min_scale=args.min_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=args.loss_scale_window,
            hysteresis=args.hysteresis)
    return Fp16Optimizer(optimizer, grad_scaler, device, getattr(args, 'use_offload', False))

