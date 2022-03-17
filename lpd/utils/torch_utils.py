import torch

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def is_gpu_available(verbose=0):
    res = torch.cuda.is_available()
    if verbose: print(f"GPU Availability: {res}")
    return res

def how_many_gpus_available():
    amount = torch.cuda.device_count()
    print('GPUs amount: ', amount)
    return amount

def get_gpu_device_if_available(verbose=0):
    if is_gpu_available(verbose):
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def what_torch_version_is_currently_running():
    v = torch.__version__
    print(f'running torch {v}')
    return v

def get_lrs_from_optimizer(optimizer):
    return [group['lr'] for group in optimizer.param_groups]

def copy_model_weights(source_model, target_model):
    self.target_model.load_state_dict(self.source_model.state_dict())