import torch as T

def seed_torch(seed):
    T.manual_seed(seed)

def is_gpu_available(verbose = 1):
    res = T.cuda.is_available()
    if verbose: print(f"GPU Availability: {res}")
    return res

def how_many_gpus_available():
    amount = T.cuda.device_count()
    print('GPUs amount: ', amount)
    return amount

def get_gpu_device_if_available():
    if is_gpu_available():
        device = T.device('cuda:0')
    else:
        device = T.device('cpu')
    return device

def what_torch_version_is_currently_running():
    v = T.__version__
    print(f'running torch {v}')
    return v

def save_checkpoint(checkpoint_filepath, epoch, model, optimizer, scheduler, msg='', verbose=1):
    if verbose:
        print(f'{msg} - Saving checkpoint to {checkpoint_filepath}')
    checkpoint = {
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler':  scheduler.state_dict() if scheduler else None,
                  'epoch':epoch
                  }
    T.save(checkpoint, checkpoint_filepath)

def load_checkpoint(checkpoint_filepath, model, optimizer, scheduler, verbose=1):
    if verbose:
        print('Loading checkpoint')
    checkpoint = T.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch']
