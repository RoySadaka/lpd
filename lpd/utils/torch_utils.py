import torch as T

def seed_torch(seed):
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.deterministic = True

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

def save_checkpoint(dir_path, file_name, trainer, msg='', verbose=1):
    full_path = dir_path +file_name
    if verbose:
        print(f'{msg} - Saving checkpoint to {full_path}')
    checkpoint = {
                  'model': trainer.model.state_dict(),
                  'optimizer': trainer.optimizer.state_dict(),
                  'scheduler': trainer.scheduler.state_dict() if trainer.scheduler else None,
                  'epoch':trainer.epoch
                  }
    T.save(checkpoint, full_path)

def load_checkpoint(checkpoint_filepath, model, optimizer, scheduler, verbose=1):
    if verbose:
        print('Loading checkpoint')
    checkpoint = T.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch']

def get_lrs_from_optimizer(optimizer):
    return [group['lr'] for group in optimizer.param_groups]