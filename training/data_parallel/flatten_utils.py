import torch


def _assert_contiguous(tensors):
    data_ptr = None
    for t in tensors:
        if data_ptr is not None:
            assert t.data_ptr() == data_ptr
        data_ptr = t.data_ptr() + t.numel() * t.element_size()


def flatten_params(param_set, chunk=None):
    params = [p for p in param_set]
    weights = [p.data for p in params]
    grads = [p.grad.data if p.grad is not None else torch.zeros_like(p.data) for p in params]
    sizes = [p.numel() for p in params]
    total_size = sum(sizes)
    if chunk:
        total_size = ((total_size+chunk-1)//chunk)*chunk

    flatten_weights_tensor = torch.zeros(total_size, dtype=weights[0].dtype).to(weights[0].device)
    flatten_grads_tensor = torch.zeros(total_size, dtype=weights[0].dtype).to(weights[0].device)
    flatten_weights_storage = flatten_weights_tensor.storage()
    flatten_grads_storage = flatten_grads_tensor.storage()

    def set_storage(param, weight_storage, grad_storage, storage_offset):
        with torch.no_grad():
            z = torch.zeros_like(param.data)
            z.set_(weight_storage, storage_offset, param.shape)
            param.data = z

            t = torch.zeros_like(param.data)
            t.set_(grad_storage, storage_offset, param.shape)
            param.grad = t

    offset = 0
    for i in range(len(params)):
        flatten_weights_tensor[offset: offset + weights[i].numel()] = weights[i].reshape(-1)
        flatten_grads_tensor[offset: offset + grads[i].numel()] = grads[i].reshape(-1)
        set_storage(params[i], flatten_weights_storage, flatten_grads_storage, offset)
        offset += sizes[i]

    weight_tensors = [p.data for p in params]
    grad_tensors = [p.grad.data for p in params]

    _assert_contiguous(weight_tensors)
    _assert_contiguous(grad_tensors)

    with torch.no_grad():
        flatten_para = torch.nn.Parameter(flatten_weights_tensor, requires_grad=False)
        flatten_para.grad = flatten_grads_tensor
        return flatten_para
    

def flatten_tensors(tensor_set, chunk=None):
    tensors = [p for p in tensor_set]
    weights = [p.data for p in tensors]
    sizes = [p.numel() for p in tensors]
    total_size = sum(sizes)
    if chunk:
        total_size = ((total_size+chunk-1)//chunk)*chunk

    flatten_weights_tensor = torch.zeros(total_size, dtype=weights[0].dtype).to(weights[0].device)
    flatten_weights_storage = flatten_weights_tensor.storage()

    def set_storage(param, weight_storage, storage_offset):
        with torch.no_grad():
            z = torch.zeros_like(param.data)
            z.set_(weight_storage, storage_offset, param.shape)
            param.data = z

    offset = 0
    for i in range(len(tensors)):
        flatten_weights_tensor[offset: offset + weights[i].numel()] = weights[i].reshape(-1)
        set_storage(tensors[i], flatten_weights_storage, offset)
        offset += sizes[i]

    return flatten_weights_tensor
