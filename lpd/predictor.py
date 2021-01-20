import torch as T
from lpd.trainer import Trainer
from lpd.enums import Phase, State

class Predictor():
    """
        slim class to make predictions on trained model

        Args:
            model - your model (nn.Module)
            device - the device to send the inputs to
            callbacks - (optional) list of lpd.callbacks to apply during the predictions phases and states
                        callbacks will be executed by the order of the list 
            name - a friendly identifier

        Methods:
            from_trainer - for creating a new Predictor instance from given Trainer
            from_checkpoint - for creating a new Predictor instance from a saved Trainer checkpoint
            predict_sample - make prediction on single sample
            predict_batch - make prediction on single batch
            predict_data_loader - make prediction on data loader (DataLoader/Iterable/Generator)
    """

    def __init__(self, model,
                       device,
                       callbacks=None,
                       name='lpd predictor'):
        self._inner = Trainer(model=model, 
                             device=device, 
                             loss_func=None,
                             optimizer=None,
                             scheduler=None,
                             metrics=None,
                             train_data_loader=None,
                             val_data_loader=None,
                             train_steps=0,
                             val_steps=0,
                             callbacks=callbacks,
                             name=name)

    @staticmethod
    def from_trainer(trainer):
        print(f'[Predictor] - Loading from trainer {trainer.name}')

        predictor = Predictor(model=trainer.model, 
                              device=trainer.device, 
                              callbacks=trainer.callbacks,
                              name=f'Predictor-for-Trainer-{trainer.name}')

        return predictor

    @staticmethod
    def from_checkpoint(dir_path,
                        file_name,
                        model, 
                        device):
        full_path = dir_path + file_name
        checkpoint = T.load(full_path, map_location=device)
        print(f'[Predictor] - Loading from {full_path}')
        model.load_state_dict(checkpoint['model'])

        callbacks = None
        if 'callbacks' in checkpoint:
            callbacks = checkpoint['callbacks']

        name = None
        if 'name' in checkpoint:
            name = checkpoint['name']
            
        predictor = Predictor(model=model, 
                              device=device, 
                              callbacks=callbacks,
                              name=name)

        return predictor

    def predict_sample(self, inputs):
        return self._inner.predict_sample(inputs)

    def predict_batch(self, inputs):
        return self._inner.predict_batch(inputs)

    def predict_data_loader(self, inputs_data_loader, steps):
        return self._inner.predict_data_loader(inputs_data_loader, steps)



