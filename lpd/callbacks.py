from lpd.utils.torch_utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import lpd.utils.file_utils as fu
import math

CB_ON_TRAIN_BEGIN   = 'on_train_begin'
CB_ON_TRAIN_END     = 'on_train_end'
CB_ON_EPOCH_BEGIN   = 'on_epoch_begin'
CB_ON_EPOCH_END     = 'on_epoch_end'
#TODO - ADD SUPPPORT FOR THESE, TAKE INTO CONSIDERATION CALLBACK IN VALIDATION MODE
# CB_ON_BATCH_BEGIN   = 'on_batch_begin'
# CB_ON_BATCH_END     = 'on_batch_end'

class CallbackContext():
    #REPRESENTS THE INPUT TO THE CALLBACK, NOTICE, SOME VALUES MIGHT BE NONE, DEPENDING ON THE PHASE OF THE CALLBACK
    def __init__(self, trainer):
        self.epoch = trainer._current_epoch
        self.train_stats = trainer.train_stats
        self.val_stats = trainer.val_stats
        self.trainer = trainer

class CallbackBase():
    def __init__(self, cb_phase = None):
        self.cb_phase = cb_phase
        if self.cb_phase is None:
            print('[CallbackBase][Error!] - No callback phase was provided')

class SchedulerStep(CallbackBase):
    """This callback will invoke a "step()" on the scheduler 
        Since some schedulers takes parameters in step(param1, param2...)
        And other schedulers step() are parametersless, provide:
        scheduler_parameters_func
        a function that except trainer and returns whatever information needed, 
        
        e.g. for scheduler that takes val_loss as parameter, initialize like this:
            SchedulerStep(scheduler_parameters_func=lambda trainer: trainer.val_stats.get_loss())

        if your scheduler step does not expect parameters, leave scheduler_parameters_func = None
    """
    def __init__(self, scheduler_parameters_func=None, cb_phase=CB_ON_EPOCH_END):
        super(SchedulerStep, self).__init__(cb_phase)
        self.scheduler_parameters_func = scheduler_parameters_func

    def __call__(self, callback_context):
        if self.scheduler_parameters_func:
            callback_context.trainer.scheduler.step(self.scheduler_parameters_func(callback_context.trainer))
        else:
            callback_context.trainer.scheduler.step()

class EpochEndStats(CallbackBase):
    """
        Informative summary at the trainer state, most likely at the end of the epoch, but you
        can change cb_phase if you need it on a different phase
        Arguments:
            cb_phase - the phase to invoke this callback
    """

    def __init__(self, cb_phase=CB_ON_EPOCH_END):
        super(EpochEndStats, self).__init__(cb_phase)
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

    def _handle_stats(self, stats, prev_loss, lowest_loss):
        curr_mean_loss = stats.get_loss()
        diff_color_str = self._get_loss_with_print_color(prev_loss, curr_mean_loss)
        lowest_loss = min(lowest_loss, prev_loss, curr_mean_loss)
        return diff_color_str, curr_mean_loss, prev_loss, lowest_loss

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 
        current_lr = round(self._get_current_lr(c.trainer.optimizer), 7)

        train_metrics = c.train_stats.get_metrics()
        t_diff_color_str, t_curr_mean_loss, t_prev_loss, t_lowest_loss = self._handle_stats(c.train_stats, self.prev_train_loss, self.lowest_train_loss)
        self.prev_train_loss = t_curr_mean_loss
        self.lowest_train_loss = t_lowest_loss

        val_metrics = c.val_stats.get_metrics()
        v_diff_color_str, v_curr_mean_loss, v_prev_loss, v_lowest_loss = self._handle_stats(c.val_stats, self.prev_val_loss, self.lowest_val_loss)
        self.prev_val_loss = v_curr_mean_loss
        self.lowest_val_loss = v_lowest_loss

        print('[EpochEndStats] - ')
        print('------------------------------------------------------')
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
    """
        Saving a checkpoint when a monitored loss has improved.
        Checkpoint will save the model, optimizer, scheduler and epoch number
        Arguments:
            checkpoint_dir - the folder to dave the model
            checkpoint_file_name - 
            monitor - can be 'val_loss', 'train_loss'
            save_best_only - if True, will override previouse best model, else, will keep both
            verbose - 0 = no print, 1 = print
            cb_phase - the phase to invoke this callback
    """

    def __init__(self, checkpoint_dir, checkpoint_file_name, monitor='val_loss', save_best_only=False, verbose=1, cb_phase=CB_ON_EPOCH_END):
        super(ModelCheckPoint, self).__init__(cb_phase)
        self.monitor = monitor  # CAN BE val_loss/train_loss
        self.save_best_only = save_best_only
        self.verbose = verbose  # VERBOSITY MODE, 0 OR 1.
        self.global_min_loss = math.inf
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file_name = checkpoint_file_name
        self._ensure_folder_created()

    def _ensure_folder_created(self):
        if not fu.is_folder_exists(self.checkpoint_dir):
            fu.create_folder(self.checkpoint_dir)

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 
        should_save = False

        if self.monitor == 'val_loss':
            loss_to_consider = c.val_stats.get_loss()
        elif self.monitor == 'train_loss':
            loss_to_consider = c.train_stats.get_loss()

        if loss_to_consider < self.global_min_loss:
            msg = f'[ModelCheckPoint] - {self.monitor} improved from {self.global_min_loss} to {loss_to_consider}'
            self.global_min_loss = loss_to_consider
            #SAVE
            if self.save_best_only:
                full_path = f'{self.checkpoint_dir}{self.checkpoint_file_name}_best_only'
            else:
                full_path = f'{self.checkpoint_dir}{self.checkpoint_file_name}_epoch_{c.epoch}'
            save_checkpoint(full_path, c.epoch, c.trainer.model, c.trainer.optimizer, c.trainer.scheduler, msg=msg, verbose=self.verbose)
        else:
            if self.verbose:
                print(f'[ModelCheckPoint] - {self.monitor} did not improved.')

class Tensorboard(CallbackBase):
    """Writes entries directly to event files in the summary_writer_dir to be
    consumed by TensorBoard.
    """

    def __init__(self, summary_writer_dir, cb_phase=CB_ON_EPOCH_END):
        super(Tensorboard, self).__init__(cb_phase)
        self.TRAIN_NAME = 'Train'
        self.VAL_NAME = 'Val'
        self.tensorboard_writer = SummaryWriter(summary_writer_dir + 'tensorboard_files')

    def _write_to_summary(self, phase_name ,epoch, stats):
        self.tensorboard_writer.add_scalar(f'{phase_name} loss', stats.get_loss(), global_step=epoch)
        for metric_name, value in stats.get_metrics().items():
            self.tensorboard_writer.add_scalar(metric_name, value, global_step=epoch)

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 
        self._write_to_summary(self.TRAIN_NAME, c.epoch, c.train_stats)
        self._write_to_summary(self.VAL_NAME, c.epoch, c.val_stats)

class EarlyStopping(CallbackBase):
    """
        Stop training when a monitored loss has stopped improving.
        Arguments:
            patience - how much epochs to wait until decide to stop
            monitor - can be 'val_loss', 'train_loss'
            cb_phase - the phase to invoke this callback
            verbose - 0 = no print, 1 = print all, 2 = print save only
    """

    def __init__(self, patience, monitor='val_loss', cb_phase=CB_ON_EPOCH_END, verbose=1):
        super(EarlyStopping, self).__init__(cb_phase)
        self.patience = patience # HOW MANY EPOCHS TO WAIT
        self.patience_countdown = patience
        self.monitor = monitor 
        self.global_min_loss = math.inf
        self.verbose = verbose

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD 

        if self.monitor == 'val_loss':
            loss_to_consider = c.val_stats.get_loss()
        elif self.monitor == 'train_loss':
            loss_to_consider = c.train_stats.get_loss()

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
            print(f'[EarlyStopping] - patience:{self.patience_countdown} epochs')
