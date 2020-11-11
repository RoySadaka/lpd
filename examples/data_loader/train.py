# IN THIS EXAMPLE WE WILL USE THE PYTORCH DATALOADER (AS OPPOSE TO PYTHON GENERATORS)

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lpd.trainer import Trainer
from lpd.extensions.custom_layers import Dense
from lpd.metrics import BinaryAccuracyWithLogits
from lpd.callbacks import StatsPrint, EarlyStopping, SchedulerStep, LossOptimizerHandler, CallbackMonitor
from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
import lpd.utils.general_utils as gu
import lpd.utils.torch_utils as tu


num_embeddings = 10
num_samples_per_file = 100
def generate_file_data():
    # [(X,y), (X,y), (X,y), (X,y)] like [ (1,0), (5,1), (6,1) ...]
    return [(T.randint(1,num_embeddings,(1,1)).squeeze(), T.randint(0,2,(1,1)).squeeze().float()) for _ in range(num_samples_per_file)]


class MyDataset(Dataset):
    def __init__(self, file_ids):
        self.file_ids = file_ids
        self.idx_to_file_id = {idx:file_id for idx, file_id in enumerate(self.file_ids)}
        self.file_id_to_sample_idx = {i:0 for i in self.file_ids}

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = self.idx_to_file_id[index]
        
        data = raw_data[file_id]
        sample_idx = self.file_id_to_sample_idx[file_id]
        self.file_id_to_sample_idx[file_id] = (self.file_id_to_sample_idx[file_id] + 1) % num_samples_per_file

        data_at_sample_idx = data[sample_idx]

        X = data_at_sample_idx[0]
        y = data_at_sample_idx[1]

        return X, y

# Parameters
params = {  'data_loader_params': { 
                                    'batch_size': 8,
                                    'shuffle': True,
                                    'num_workers': 1
                                   },
            'H': 128,
            'D_out': 1,
            'embedding_dim': 64,
            'num_epochs': 80}

# Datasets
file_ids = [f'id-{i}' for i in range(16)]
partition = {
                #FOR SIMPLICITY, TRAIN/VAL/TEST WILL BE THE SAME DATA
                'train': file_ids, 
                'val':   file_ids,  
                'test':  file_ids   
            }

raw_data = {file_id:generate_file_data() for file_id in partition['train']}

# Generators
train_dataset = MyDataset(partition['train'])
train_data_loader = DataLoader(train_dataset, **params['data_loader_params'])

val_dataset = MyDataset(partition['val'])
val_data_loader = DataLoader(val_dataset, **params['data_loader_params'])

test_dataset = MyDataset(partition['test'])
test_data_loader = DataLoader(test_dataset, **params['data_loader_params'])


class Model(nn.Module):
    def __init__(self, H, D_out, num_embeddings, embedding_dim):
        super(Model, self).__init__()

        #LAYERS

        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings + 1, 
                                            embedding_dim=embedding_dim)

        self.dense = Dense(embedding_dim, H, use_bias=True, activation=nn.ReLU())
        self.dense_out = Dense(H, D_out, use_bias=True, activation=None)

    def forward(self, x):               
        x = self.embedding_layer(x)     # (batch, embedding_dim)
        x = self.dense(x)               # (batch, H)
        x = self.dense_out(x)           # (batch, 1)
        x = x.squeeze()                 # (batch)
        return x #NOTICE! LOGITS OUT, NOT SIGMOID, THE SIGMOID WILL BE APPLIED IN BCEWithLogitsLoss

def get_trainer(params):

    device = tu.get_gpu_device_if_available()

    # Use the nn package to define our model and loss function.
    model = Model(params['H'], params['D_out'], num_embeddings, params['embedding_dim']).to(device)

    loss_func = nn.BCEWithLogitsLoss().to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    metric_name_to_func = {"acc":BinaryAccuracyWithLogits()}

    callbacks = [  
                    LossOptimizerHandler(),
                    SchedulerStep(apply_on_phase=Phase.BATCH_END, apply_on_states=State.TRAIN),
                    EarlyStopping(callback_monitor=CallbackMonitor(patience=3, monitor_type=MonitorType.LOSS, stats_type=StatsType.VAL, monitor_mode=MonitorMode.MIN)),
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
                      callbacks=callbacks,
                      name='DataLoader-Example')
    return trainer


def run():
    gu.seed_all(42)  # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE

    trainer = get_trainer(params)
    
    trainer.summary()

    trainer.train(params['num_epochs'])

    evaluation = trainer.evaluate(test_data_loader, len(test_data_loader))

    print(evaluation)