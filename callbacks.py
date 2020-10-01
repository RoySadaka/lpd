from torch.utils.tensorboard import SummaryWriter
import math

import utils.file_utils as fu
from utils.torch_utils import save_checkpoint


CB_ON_TRAIN_BEGIN   = 'on_train_begin'
CB_ON_TRAIN_END     = 'on_train_end'
CB_ON_EPOCH_BEGIN   = 'on_epoch_begin'
CB_ON_EPOCH_END     = 'on_epoch_end'
CB_ON_BATCH_BEGIN   = 'on_batch_begin'
CB_ON_BATCH_END     = 'on_batch_end'
CB_ON_LOSS_BEGIN    = 'on_loss_begin'
CB_ON_LOSS_END      = 'on_loss_end'


class CallbackContext():
    #REPRESENTS THE INPUT TO THE CALLBACK, NOTICE, SOME VALUES MIGHT BE NONE, DEPENDING ON THE PHASE OF THE CALLBACK
    def __init__(self, epoch, train_loss_stats, train_metric_name_to_stats, val_loss_stats, val_metric_name_to_stats, trainer):
        self.epoch = epoch
        self.train_loss_stats = train_loss_stats
        self.train_metric_name_to_stats = train_metric_name_to_stats
        self.val_loss_stats = val_loss_stats
        self.val_metric_name_to_stats = val_metric_name_to_stats
        self.trainer = trainer

class CallbackBase():
    def __init__(self):
        self.cb_phase = ''

class EpochEndStats(CallbackBase):
    def __init__(self):
        self.cb_phase = CB_ON_EPOCH_END
        self.prev_train_loss = math.inf
        self.prev_val_loss = math.inf
        self.lowest_train_loss = math.inf
        self.lowest_val_loss = math.inf
        self.YELLOW_PRINT_COLOR = "\033[93m"
        self.GREEN_PRINT_COLOR = "\033[92m"
        self.RED_PRINT_COLOR = "\033[91m"
        self.END_PRINT_COLOR = "\033[0m"

    def _get_current_lr(self, optimizer):
        #CURRENTLY WILL RETURN ONLY FOR param_groups[0]
        return optimizer.param_groups[0]['lr']

    def _get_loss_with_print_color(self, prev_loss, mean_loss):
        diff_loss = round(mean_loss - prev_loss, 7)

        if diff_loss < 0:
            return self.GREEN_PRINT_COLOR + str(diff_loss) + self.END_PRINT_COLOR
        if diff_loss > 0:
            return self.RED_PRINT_COLOR + str(diff_loss) + self.END_PRINT_COLOR
        return self.YELLOW_PRINT_COLOR + str(diff_loss) + self.END_PRINT_COLOR


    def _handle_stats(self, loss_stats, prev_loss, lowest_loss):
        curr_mean_loss = loss_stats.get_mean()
        diff_color_str = self._get_loss_with_print_color(prev_loss, curr_mean_loss)
        lowest_loss = min(lowest_loss, prev_loss, curr_mean_loss)
        return diff_color_str, curr_mean_loss, prev_loss, lowest_loss

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 
        current_lr = round(self._get_current_lr(c.trainer.optimizer), 7)

        train_metrics = {metric_name:stats.get_mean() for metric_name,stats in c.train_metric_name_to_stats.items()}
        t_diff_color_str, t_curr_mean_loss, t_prev_loss, t_lowest_loss = self._handle_stats(c.train_loss_stats, self.prev_train_loss, self.lowest_train_loss)
        self.prev_train_loss = t_curr_mean_loss
        self.lowest_train_loss = t_lowest_loss

        val_metrics = {metric_name:stats.get_mean() for metric_name,stats in c.val_metric_name_to_stats.items()}
        v_diff_color_str, v_curr_mean_loss, v_prev_loss, v_lowest_loss = self._handle_stats(c.val_loss_stats, self.prev_val_loss, self.lowest_val_loss)
        self.prev_val_loss = v_curr_mean_loss
        self.lowest_val_loss = v_lowest_loss

        print('') #EMPTY LINE SEPERATOR
        print('-------------------[EpochEndPrint]--------------------')
        print(f'| Stats                ') 
        print(f'|   |-- Epoch:{c.epoch}')
        print(f'|   |-- Learning rate:{current_lr}')
        print(f'|   |-- Train                ') 
        print(f'|   |     |-- loss')
        print(f'|   |     |     |-- curr:{t_curr_mean_loss}, prev:{t_prev_loss}, change:{t_diff_color_str}, lowest:{self.lowest_train_loss}')
        print(f'|   |     |-- metrics        ')
        print(f'|   |           |-- {train_metrics}')
        print(f'|   |                        ')
        print(f'|   |-- Validation           ')   
        print(f'|         |-- loss')
        print(f'|         |     |-- curr:{v_curr_mean_loss}, prev:{v_prev_loss}, change:{v_diff_color_str}, lowest:{self.lowest_val_loss}')
        print(f'|         |-- metrics        ') 
        print(f'|               |-- {val_metrics}')
        print('------------------------------------------------------')
        print('') #EMPTY LINE SEPERATOR

class ModelCheckPoint(CallbackBase):
    def __init__(self, model_weights_dir, model_weights_file_name, monitor='val_loss', save_best_only=False):
        self.cb_phase = CB_ON_EPOCH_END
        self.monitor = monitor  #CAN BE  val_loss/train_loss
        self.save_best_only = save_best_only
        self.global_min_loss = math.inf
        self.model_weights_dir = model_weights_dir
        self.model_weights_file_name = model_weights_file_name
        self._ensure_folder_created()

    def _ensure_folder_created(self):
        if not fu.is_folder_exists(self.model_weights_dir):
            fu.create_folder(self.model_weights_dir)

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 
        should_save = False

        if self.monitor == 'val_loss':
            loss_to_consider = c.val_loss_stats.get_mean()
        elif self.monitor == 'train_loss':
            loss_to_consider = c.train_loss_stats.get_mean()

        if loss_to_consider < self.global_min_loss:
            msg = f'[ModelCheckPoint] - {self.monitor} improved from {self.global_min_loss} to {loss_to_consider}'
            self.global_min_loss = loss_to_consider
            #SAVE
            if self.save_best_only:
                full_path = f'{self.model_weights_dir}{self.model_weights_file_name}_best_only'
            else:
                full_path = f'{self.model_weights_dir}{self.model_weights_file_name}_epoch_{c.epoch}'
            save_checkpoint(full_path, c.epoch, c.trainer.model, c.trainer.optimizer, c.trainer.scheduler, msg=msg)
        else:
            print(f'[ModelCheckPoint] - {self.monitor} did not improved.')

class Tensorboard(CallbackBase):
    def __init__(self, summary_writer_dir):
        self.cb_phase = CB_ON_EPOCH_END
        self.TRAIN_NAME = 'Train'
        self.VAL_NAME = 'Val'
        self.tensorboard_writer = SummaryWriter(summary_writer_dir + 'tensorboard_files')

    def _write_to_summary(self, phase_name ,epoch, loss_stats, metric_name_to_stats):
        self.tensorboard_writer.add_scalar(f'{phase_name} loss', loss_stats.get_mean(), global_step=epoch)
        for metric_name, stats in metric_name_to_stats.items():
            self.tensorboard_writer.add_scalar(metric_name, stats.get_mean(), global_step=epoch)

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 
        self._write_to_summary(self.TRAIN_NAME, c.epoch, c.train_loss_stats, c.train_metric_name_to_stats)
        self._write_to_summary(self.VAL_NAME, c.epoch, c.val_loss_stats, c.val_metric_name_to_stats)

class EarlyStopping(CallbackBase):
    def __init__(self, patience, monitor='val_loss', verbose=1):
        self.cb_phase = CB_ON_EPOCH_END
        self.patience = patience # HOW MANY EPOCHS TO WAIT
        self.patience_countdown = patience
        self.monitor = monitor # CAN BE 'val_loss', 'train_loss'
        self.global_min_loss = math.inf
        self.verbose = verbose # 0 = NO PRINT, 1 = PRINT ALL, 2 = PRINT SAVE ONLY

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 

        if self.monitor == 'val_loss':
            loss_to_consider = c.val_loss_stats.get_mean()
        elif self.monitor == 'train_loss':
            loss_to_consider = c.train_loss_stats.get_mean()

        if loss_to_consider < self.global_min_loss:
            self.global_min_loss = loss_to_consider
            self.patience_countdown = self.patience
        else:
            self.patience_countdown -= 1

        if self.patience_countdown == 0:
            if self.verbose > 0:
                print(f'[EarlyStopping] - stopping on epoch {c.epoch}')
            c.trainer.stop_training()
            return

        if self.verbose == 1:
            print(f'[EarlyStopping] - patience:{self.patience_countdown}')
