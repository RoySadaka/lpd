from lpd.utils.torch_utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import lpd.utils.file_utils as fu
import lpd.enums as en

from math import inf


class CallbackContext():
    #REPRESENTS THE INPUT TO THE CALLBACK, NOTICE, SOME VALUES MIGHT BE NONE, DEPENDING ON THE PHASE OF THE CALLBACK
    def __init__(self, trainer):
        self.epoch = trainer._current_epoch
        self.train_stats = trainer.train_stats
        self.val_stats = trainer.val_stats
        self.trainer_state = trainer.state
        self.trainer = trainer

class CallbackBase():
    """
        Agrs:
            cb_phase - (lpd.enums.CallbackPhase) the phase to invoke this callback e.g 
            round_values_on_print_to - optional, it will round the numerical values in the prints
            apply_on_states - (lpd.enums.State) state or list of states to invoke this parameter (under the relevant cb_phase), None will invoke it on all states
    """

    def __init__(self, cb_phase = None, round_values_on_print_to = None, apply_on_states=None):
        self.cb_phase = cb_phase
        if self.cb_phase is None:
            print('[CallbackBase][Error!] - No callback phase was provided')

        self.apply_on_states = apply_on_states
        self.round_values_on_print_to = round_values_on_print_to

    def round_to(self, value):
        if self.round_values_on_print_to:
            return round(value, self.round_values_on_print_to)
        return value

    def should_apply_on_state(self, callback_context):
        if self.apply_on_states is None:
            return True

        if isinstance(self.apply_on_states, list):
            for state in self.apply_on_states:
                if isinstance(state, en.State):
                    if callback_context.trainer_state == state:
                        return True
            return False

        if isinstance(self.apply_on_states, en.State):
            return callback_context.trainer_state == self.apply_on_states

class SchedulerStep(CallbackBase):
    """This callback will invoke a "step()" on the scheduler.

        Agrs:
            scheduler_parameters_func - Since some schedulers takes parameters in step(param1, param2...)
                And other schedulers step() are parameterless, provide:
                a function (or lambda) that except trainer and returns whatever information needed,
                e.g. for scheduler that takes val_loss as parameter, initialize like this:
                    SchedulerStep(scheduler_parameters_func=lambda trainer: trainer.val_stats.get_loss())
                if your scheduler step does not expect parameters, leave scheduler_parameters_func = None
            cb_phase - see in CallbackBase
            apply_on_states - see in CallbackBase
    """
    def __init__(self, scheduler_parameters_func=None, 
                       cb_phase=en.CallbackPhase.ON_EPOCH_END, 
                       apply_on_states=None):
        super(SchedulerStep, self).__init__(cb_phase=cb_phase, apply_on_states=apply_on_states)
        self.scheduler_parameters_func = scheduler_parameters_func

    def __call__(self, callback_context):
        if not self.should_apply_on_state(callback_context):
            return

        if callback_context.trainer.scheduler is None:
            print('[SchedulerStep] - no scheduler defined in trainer')
            return
        if self.scheduler_parameters_func:
            callback_context.trainer.scheduler.step(self.scheduler_parameters_func(callback_context.trainer))
        else:
            callback_context.trainer.scheduler.step()

class EpochEndStats(CallbackBase):
    """
        Informative summary at the trainer state, most likely at the end of the epoch, 
        but you can change cb_phase if you need it on a different phase
        Args:
            cb_phase - see in CallbackBase
            round_values_on_print_to - see in CallbackBase
    """

    def __init__(self, cb_phase=en.CallbackPhase.ON_EPOCH_END, round_values_on_print_to = None):
        super(EpochEndStats, self).__init__(cb_phase, round_values_on_print_to)
        self.prev_train_loss = inf
        self.prev_val_loss = inf
        self.lowest_train_loss = inf
        self.lowest_val_loss = inf
        self.YELLOW_PRINT_COLOR = "\033[93m"
        self.GREEN_PRINT_COLOR = "\033[92m"
        self.RED_PRINT_COLOR = "\033[91m"
        self.END_PRINT_COLOR = "\033[0m"

    def _get_current_lr(self, optimizer):
        #CURRENTLY WILL RETURN ONLY FOR param_groups[0]
        return optimizer.param_groups[0]['lr']

    def _get_loss_with_print_color(self, prev_loss, mean_loss):
        diff_loss = self.round_to(mean_loss - prev_loss)

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

    def _round_metrics(self, metrics):
        if len(metrics) == 0:
            return 'no metrics found'
        if self.round_values_on_print_to:
            return {metric:self.round_to(value) for metric,value in metrics.items()}
        return metrics

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD
        r = self.round_to #READABILITY DOWN THE ROAD
        r_met = self._round_metrics #READABILITY DOWN THE ROAD
        current_lr = self._get_current_lr(c.trainer.optimizer)

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
        print(f'| Stats for Trainer: {c.trainer.name}')
        print(f'|   |-- Epoch:{c.epoch}')
        print(f'|   |-- Learning rate:{r(current_lr)}')
        print(f'|   |-- Train')
        print(f'|   |     |-- loss')
        print(f'|   |     |     |-- curr:{r(t_curr_mean_loss)}, prev:{r(t_prev_loss)}, change:{t_diff_color_str}, lowest:{r(self.lowest_train_loss)}')
        print(f'|   |     |-- metrics')
        print(f'|   |           |-- {r_met(train_metrics)}')
        print(f'|   |')
        print(f'|   |-- Validation')
        print(f'|         |-- loss')
        print(f'|         |     |-- curr:{r(v_curr_mean_loss)}, prev:{r(v_prev_loss)}, change:{v_diff_color_str}, lowest:{r(self.lowest_val_loss)}')
        print(f'|         |-- metrics')
        print(f'|               |-- {r_met(val_metrics)}')
        print('------------------------------------------------------')
        print('') #EMPTY LINE SEPARATOR

class ModelCheckPoint(CallbackBase):
    """
        Saving a checkpoint when a monitored loss has improved.
        Checkpoint will save the model, optimizer, scheduler and epoch number
        Args:
            checkpoint_dir - the folder to dave the model
            checkpoint_file_name -
            monitor - can be 'val_loss', 'train_loss'
            save_best_only - if True, will override previous best model, else, will keep both
            verbose - 0 = no print, 1 = print
            cb_phase - see in CallbackBase
            round_values_on_print_to - see in CallbackBase
    """

    def __init__(self, checkpoint_dir, 
                       checkpoint_file_name, 
                       monitor='val_loss', 
                       save_best_only=False, 
                       verbose=1, 
                       cb_phase=en.CallbackPhase.ON_EPOCH_END, 
                       round_values_on_print_to = None):
        super(ModelCheckPoint, self).__init__(cb_phase, round_values_on_print_to)
        self.monitor = monitor  # CAN BE val_loss/train_loss
        self.save_best_only = save_best_only
        self.verbose = verbose  # VERBOSITY MODE, 0 OR 1.
        self.global_min_loss = inf
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file_name = checkpoint_file_name
        self._ensure_folder_created()

    def _ensure_folder_created(self):
        if not fu.is_folder_exists(self.checkpoint_dir):
            fu.create_folder(self.checkpoint_dir)

    def __call__(self, callback_context):
        c = callback_context #READABILITY DOWN THE ROAD
        r = self.round_to #READABILITY DOWN THE ROAD

        if self.monitor == 'val_loss':
            loss_to_consider = c.val_stats.get_loss()
        elif self.monitor == 'train_loss':
            loss_to_consider = c.train_stats.get_loss()

        if loss_to_consider < self.global_min_loss:
            msg = f'[ModelCheckPoint] - {self.monitor} improved from {r(self.global_min_loss)} to {r(loss_to_consider)}'
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
    """ 
        Writes entries directly to event files in the summary_writer_dir to be
        consumed by TensorBoard.
        Args:
            summary_writer_dir - the folder path to save tensorboard output
            cb_phase - see in CallbackBase
    """

    def __init__(self, summary_writer_dir, cb_phase=en.CallbackPhase.ON_EPOCH_END):
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
        Args:
            patience - how much epochs to wait until decide to stop
            monitor - can be 'val_loss', 'train_loss'
            cb_phase - see in CallbackBase
            verbose - 0 = no print, 1 = print all, 2 = print save only
    """

    def __init__(self, patience, monitor='val_loss', cb_phase=en.CallbackPhase.ON_EPOCH_END, verbose=1):
        super(EarlyStopping, self).__init__(cb_phase)
        self.patience = patience # HOW MANY EPOCHS TO WAIT
        self.patience_countdown = patience
        self.monitor = monitor
        self.global_min_loss = inf
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
