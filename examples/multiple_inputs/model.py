import torch as T
import torch.nn as nn
import torch.optim as optim

from lpd.trainer import Trainer
from lpd.extensions.custom_layers import TransformerEncoderStack, Attention, MatMul2D
from lpd.enums import Phase, State, MonitorType, MonitorMode, StatsType
from lpd.callbacks import StatsPrint, ModelCheckPoint, Tensorboard, EarlyStopping, SchedulerStep, LossOptimizerHandler, CallbackMonitor
from lpd.metrics import BinaryAccuracyWithLogits, TruePositives
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

        self.external_query_attention = Attention(key_dim=config.EMBEDDINGS_SIZE, use_query_dense=True)
        self.norm = nn.LayerNorm(normalized_shape=config.EMBEDDINGS_SIZE) # WILL APPLY NORM OVER THE LAST DIMENTION ONLY
        self.mat_mul2d = MatMul2D(transpose_b=True)

    def forward(self, x1, x2, x3):
        # x1   : sequence-Input  	(batch, num_elements)
        # x2   : some1-Input        (batch, 1)
        # x3   : some2-Input        (batch, 1)

        x1_emb = self.embedding_layer(x1)                                                     # (batch, num_elements, emb_size)
        x1_emb_transformed = self.transformer_encoder(x1_emb)                                 # (batch, num_elements, emb_size)
        
        x3_emb = self.embedding_layer(x3)                                                     # (batch, emb_size)
        x3_emb_unsqueesed = x3_emb.unsqueeze(1)                                               # (batch, 1, emb_size)

        x1_with_x3_reduced = self.external_query_attention(q=x3_emb_unsqueesed, 
                                                           k=x1_emb_transformed, 
                                                           v=x1_emb_transformed)              # (batch, 1, emb_size)
        

        x1_with_x3_residual = self.norm(x1_with_x3_reduced + x3_emb_unsqueesed)     		  # (batch, 1, emb_size)

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

    metric_name_to_func = {
                           "Accuracy":BinaryAccuracyWithLogits(),
                           "TP":TruePositives(2, threshold = 0)
                           }

    callbacks = [   
                    LossOptimizerHandler(),
                    SchedulerStep(scheduler_parameters_func=lambda callback_context: callback_context.val_stats.get_loss()),
                    
                    ModelCheckPoint(checkpoint_dir=checkpoint_dir, 
                                    checkpoint_file_name=checkpoint_file_name, 
                                    callback_monitor=CallbackMonitor(patience=-1,
                                                                     monitor_type=MonitorType.LOSS, 
                                                                     stats_type=StatsType.VAL, 
                                                                     monitor_mode=MonitorMode.MIN),
                                    save_best_only=True, 
                                    round_values_on_print_to=7), 
                    Tensorboard(summary_writer_dir=summary_writer_dir),
                    EarlyStopping(apply_on_phase=Phase.EPOCH_END, 
                                  apply_on_states=State.EXTERNAL,
                                  callback_monitor=CallbackMonitor(patience=config.EARLY_STOPPING_PATIENCE, 
                                                                    monitor_type=MonitorType.LOSS, 
                                                                    stats_type=StatsType.VAL, 
                                                                    monitor_mode=MonitorMode.MIN)),
                    StatsPrint(apply_on_phase=Phase.EPOCH_END, round_values_on_print_to=7, print_confusion_matrix_normalized=True) # BETTER TO PUT StatsPrint LAST (MAKES BETTER SENSE IN THE LOG PRINTS)
                ]

    trainer = Trainer(model=model, 
                      device=device, 
                      loss_func=loss_func, 
                      optimizer=optimizer,
                      scheduler=scheduler,
                      metric_name_to_func=metric_name_to_func, 
                      train_data_loader=train_data_loader, 
                      val_data_loader=val_data_loader,
                      train_steps=train_steps,
                      val_steps=val_steps,
                      callbacks=callbacks,
                      name='Multi-Input-Example')
    return trainer
