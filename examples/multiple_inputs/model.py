import torch
import torch.nn as nn
import torch.optim as optim

from lpd.trainer import Trainer
from lpd.extensions.custom_layers import TransformerEncoderStack, Attention, MatMul2D
from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks import StatsPrint, ModelCheckPoint, Tensorboard, EarlyStopping, SchedulerStep, LossOptimizerHandler, CallbackMonitor
from lpd.metrics import BinaryAccuracyWithLogits, TruePositives, FalsePositives
from lpd.extensions.custom_schedulers import DoNothingToLR
import lpd.utils.torch_utils as tu


class TestModel(nn.Module):
    def __init__(self, config, num_embeddings):
        super(TestModel, self).__init__()
        self.config = config
        self.num_embeddings = num_embeddings

        #LAYERS
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings, 
                                            embedding_dim=config.EMBEDDINGS_SIZE)
        nn.init.uniform_(self.embedding_layer.weight, a=-0.05, b=0.05) # I PREFER THIS INIT

        self.transformer_encoder = TransformerEncoderStack(in_dim=config.EMBEDDINGS_SIZE, 
                                                            key_dim=config.TRANSFORMER_KEY_DIM,
                                                            out_dim=config.EMBEDDINGS_SIZE,
                                                            num_encoders=config.NUM_TRANSFORMER_ENCODERS,
                                                            num_heads=config.NUM_HEADS_PER_TRANSFORMER,
                                                            drop_out_proba=config.TRANSFORMER_DROP_OUT_PROBA,
                                                            ff_expansion_rate=config.TRANSFORMER_FF_EXPANSION_RATE)

        self.external_query_attention = Attention()
        self.norm = nn.LayerNorm(normalized_shape=config.EMBEDDINGS_SIZE) # WILL APPLY NORM OVER THE LAST DIMENTION ONLY
        self.mat_mul2d = MatMul2D(transpose_b=True)

    def forward(self, x1, x2, x3, index_select_aux):
        # x1   : sequence-Input  	(batch, num_elements)
        # x2   : some1-Input        (batch, 1)
        # x3   : some2-Input        (batch, 1)

        x1_emb = self.embedding_layer(x1)                                                     # (batch, num_elements, emb_size)
        x1_emb_transformed = self.transformer_encoder(x1_emb)                                 # (batch, num_elements, emb_size)
        
        x3_emb = self.embedding_layer(x3)                                                     # (batch, emb_size)
        x3_emb_unsqueeze = x3_emb.unsqueeze(1)                                               # (batch, 1, emb_size)

        x1_with_x3_reduced = torch.cat([x3_emb_unsqueeze, x1_emb_transformed], dim=1)          # (batch, num_elements+1, emb_size)

        x1_with_x3_reduced = self.external_query_attention(q=x1_with_x3_reduced, 
                                                           k=x1_with_x3_reduced, 
                                                           v=x1_with_x3_reduced)              # (batch, num_elements+1, emb_size)
        
        x1_with_x3_reduced = torch.index_select(x1_with_x3_reduced, dim=1, index=index_select_aux) # (batch, 1, emb_size)

        x1_with_x3_residual = self.norm(x1_with_x3_reduced + x3_emb_unsqueeze)     		  # (batch, 1, emb_size)

        x2_emb = self.embedding_layer(x2)                                                     # (batch, emb_size)

        dot_product = self.mat_mul2d(x2_emb.unsqueeze(1), x1_with_x3_residual)                  # (batch, 1, 1)
        dot_product = dot_product.squeeze(2).squeeze(1)  # safe on batch_size = 1             # (batch)
        return dot_product #NOTICE! LOGITS OUT, NOT SIGMOID, THE SIGMOID WILL BE APPLIED IN THE LOSS HANDLER FOR THIS EXAMPLE

def get_trainer(config, 
                num_embeddings,                         
                train_data_loader, 
                val_data_loader,
                train_steps,
                val_steps,
                checkpoint_dir,
                checkpoint_file_name,
                summary_writer_dir):
    device = tu.get_gpu_device_if_available()

    model = TestModel(config, num_embeddings).to(device)
   
    optimizer = optim.SGD(params=model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config.EARLY_STOPPING_PATIENCE // 2, verbose=True) # needs SchedulerStep callback WITH scheduler_parameters_func
    
    loss_func = nn.BCEWithLogitsLoss().to(device)

    metrics = [
                           BinaryAccuracyWithLogits(name='Accuracy'),
                           TruePositives(num_classes=2, threshold=0, name='TP'),
                           FalsePositives(num_classes=2, threshold=0, name='FP')
                        ]

    callbacks = [   
                    LossOptimizerHandler(),
                    SchedulerStep(scheduler_parameters_func=lambda callback_context: callback_context.val_stats.get_loss()),
                    
   
                    Tensorboard(summary_writer_dir=summary_writer_dir),
                    EarlyStopping(apply_on_phase=Phase.EPOCH_END, 
                                  apply_on_states=State.EXTERNAL,
                                  callback_monitor=CallbackMonitor(monitor_type=MonitorType.LOSS, 
                                                                   stats_type=StatsType.VAL, 
                                                                   monitor_mode=MonitorMode.MIN,
                                                                   patience=config.EARLY_STOPPING_PATIENCE)),
                    StatsPrint(apply_on_phase=Phase.EPOCH_END, 
                               round_values_on_print_to=7, 
                               print_confusion_matrix_normalized=True, 
                               train_best_confusion_matrix_monitor=CallbackMonitor(monitor_type=MonitorType.LOSS, 
                                                                                   stats_type=StatsType.TRAIN, 
                                                                                   monitor_mode=MonitorMode.MIN)),
                    ModelCheckPoint(checkpoint_dir=checkpoint_dir, 
                                    checkpoint_file_name=checkpoint_file_name, 
                                    callback_monitor=CallbackMonitor(monitor_type=MonitorType.LOSS, 
                                                                     stats_type=StatsType.VAL, 
                                                                     monitor_mode=MonitorMode.MIN),
                                    save_best_only=True, 
                                    round_values_on_print_to=7), # BETTER TO PUT ModelCheckPoint LAST (SO IN CASE IT SAVES, THE STATES OF ALL THE CALLBACKS WILL BE UP TO DATE)
                ]

    trainer = Trainer(model=model, 
                      device=device, 
                      loss_func=loss_func, 
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metrics=metrics, 
                      train_data_loader=train_data_loader, 
                      val_data_loader=val_data_loader,
                      train_steps=train_steps,
                      val_steps=val_steps,
                      callbacks=callbacks,
                      name='Multi-Input-Example')
    return trainer
