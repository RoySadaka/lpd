import torch as T

def is_gpu_available(verbose = 1):
    res = T.cuda.is_available()
    if verbose: print(f"GPU Availablility: {res}")
    return res

def how_many_gpus_available():
    amount = T.cuda.device_count()
    print('GPUs amount: ', amount)
    return amount

def get_training_available_hardware():
    if is_gpu_available():
      device = T.device('cuda:0')
    else:
      device = T.device('cpu')
    return device

def assign_model_to_gpu(model):
    if is_gpu_available(verbose = 0):
      print('model is assigned to GPU')
    else:
      print('model is assigned to CPU')
    device = get_training_available_hardware()
    model.to(device)

def what_torch_version_is_currently_running():
    v = T.__version__
    print(f'running torch {v}')
    return v

def save_checkpoint(checkpoint_filepath, epoch, model, optimizer, scheduler, msg='', verbose=1):
    if verbose:
      print(msg + " - Saving checkpoint")
    checkpoint = {
                  'model': model.state_dict(), 
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'epoch':epoch
                  }
    T.save(checkpoint, checkpoint_filepath)

def load_checkpoint(checkpoint_filepath, model, optimizer, scheduler):
    print('Loading checkpoint')
    checkpoint = T.load(checkpoint_filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch']