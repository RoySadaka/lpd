# THE FOLLOWING EXAMPLE WAS CONSTRUCTED FROM
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# AND MIGRATED TO USE lpd FOR TRAINING
#IN THIS EXAMPLE WE WILL USE THE PYTORCH DATALOADER (AS OPPOSE TO PYTHON GENERATORS)

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lpd.trainer import Trainer
from lpd.extensions.custom_layers import Dense
from lpd.extensions.custom_metrics import binary_accuracy_with_logits
from lpd.callbacks import StatsPrint, EarlyStopping, SchedulerStep
from lpd.enums import CallbackPhase, TrainerState, MonitorType, MonitorMode, StatsType
import lpd.utils.general_utils as gu
import lpd.utils.torch_utils as tu


class MyDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

    def _generate_fake_sample(self, ID):
        # YOU CAN ALSO USE T.load('data/' + ID + '.pt') IF YOU HAVE FILES PER EACH ID
        id_as_number = int(ID[-1])
        return T.LongTensor([id_as_number]) #FOR 'id-1' RETURN [1,1,1,1], FOR 'id-2' RETURN [2,2,2,2] AND SO ON

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self._generate_fake_sample(ID)
        y = self.labels[ID]

        return X, y

# Parameters
params = {  'data_loader_params': { 
                                    'batch_size': 8,
                                    'shuffle': True,
                                    'num_workers': 6
                                   },
            'D_in': 4,
            'H': 128,
            'D_out': 1,
            'embedding_dim': 64,
            'num_epochs': 50}

# Datasets
partition = {
                #FOR SIMPLICITY, TRAIN/VAL/TEST WILL BE THE SAME DATA
                'train': ['id-1', 'id-2', 'id-3', 'id-4', 'id-5', 'id-6', 'id-7', 'id-8'], 
                'val':   ['id-1', 'id-2', 'id-3', 'id-4', 'id-5', 'id-6', 'id-7', 'id-8'],  
                'test':  ['id-1', 'id-2', 'id-3', 'id-4', 'id-5', 'id-6', 'id-7', 'id-8']   
            }

labels = {'id-1': 0., 'id-2': 1., 'id-3': 0., 'id-4': 1., 'id-5': 0., 'id-6': 1., 'id-7': 0., 'id-8': 1.}

# Generators
train_dataset = MyDataset(partition['train'], labels)
train_data_loader = DataLoader(train_dataset, **params['data_loader_params'])

val_dataset = MyDataset(partition['val'], labels)
val_data_loader = DataLoader(val_dataset, **params['data_loader_params'])

test_dataset = MyDataset(partition['test'], labels)
test_data_loader = DataLoader(test_dataset, **params['data_loader_params'])


class Model(nn.Module):
    def __init__(self, D_in, H, D_out, num_embeddings, embedding_dim):
        super(Model, self).__init__()

        #LAYERS

        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings + 1, 
                                            embedding_dim=embedding_dim)
        # nn.init.uniform_(self.embedding_layer.weight, a=-0.05, b=0.05) # I PREFER THE INIT THAT TensorFlow DO FOR Embedding

        self.dense = Dense(embedding_dim, H, use_bias=True, activation=F.relu)
        self.dense_out = Dense(H, D_out, use_bias=True, activation=None)

    def forward(self, x):               # (batch, D_in)
        x = self.embedding_layer(x)
        x = self.dense(x)               # (batch, H)
        x = self.dense_out(x)           # (batch, 1, 1)
        x = x.squeeze(2).squeeze(1)     # (batch, 1)
        return x #NOTICE! LOGITS OUT, NOT SIGMOID, THE SIGMOID WILL BE APPLIED IN THE LOSS HANDLER FOR THIS EXAMPLE

def get_trainer(params):

    device = tu.get_gpu_device_if_available()

    # Use the nn package to define our model and loss function.
    num_embeddings = len(train_dataset)
    model = Model(params['D_in'], params['H'], params['D_out'], num_embeddings, params['embedding_dim']).to(device)

    loss_func = nn.BCEWithLogitsLoss().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    metric_name_to_func = {"acc":binary_accuracy_with_logits}

    callbacks = [   
                    SchedulerStep(cb_phase=CallbackPhase.ON_BATCH_END, apply_on_states=TrainerState.TRAIN),
                    EarlyStopping(patience=3, monitor_type=MonitorType.LOSS, stats_type=StatsType.VAL, monitor_mode=MonitorMode.MIN),
                    StatsPrint(round_values_on_print_to=7)
                ]

    trainer = Trainer(model=model, 
                      device=device, 
                      loss_func=loss_func, 
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metric_name_to_func=metric_name_to_func, 
                      train_data_loader=train_data_loader, 
                      val_data_loader=val_data_loader,
                      train_steps=len(train_dataset),
                      val_steps=len(val_dataset),
                      num_epochs=params['num_epochs'],
                      callbacks=callbacks,
                      name='DataLoader-Example')
    return trainer


def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    trainer = get_trainer(params)
    
    trainer.summary()

    trainer.train()

    trainer.evaluate(test_data_loader, len(test_data_loader))