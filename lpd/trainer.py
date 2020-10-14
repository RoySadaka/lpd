import torch as T
from tqdm import tqdm
from lpd.callbacks import CallbackContext
from lpd.enums import State, Phase
from lpd.trainer_stats import TrainerStats
import lpd.utils.file_utils as fu

class Trainer():
    """
        class that maintains all the participating objects and stats during training and evaluation

        Args:
            model - your model (nn.Module)
            device - the device to send the inputs to
            loss_func - the model's loss function
            optimizer - the model's optimizer
            scheduler - the model's scheduler (make sure you add SchedulerStep to your callbacks),
                        pass None if you dont need scheduler
            metric_name_to_func - a dictionary with string as key and metric function as value
                        e.g.   {"binary_accuracy":lpd.extensions.custom_metrics.binary_accuracy_with_logits}
            train_data_loader - an iterable or generator to get the next train data batch
            val_data_loader - an iterable or generator to get the next val data batch
            train_steps - total number of steps (batches) before declaring the epoch as finished
            val_steps - total number of steps (batches) before declaring the epoch as finished
            num_epochs - number of epochs to train the model
            callbacks - list of lpd.callbacks to apply during the differrent training phases
            name - just an identifier, in case you create multiple trainers
            optimizer_step_and_zero_grad_criteria - in case of special handling, pass a function that expect the trainer, else pass None

        Methods:
            summary - will print information about the trainer and the model
            stop - will indicate this trainer to stop, e.g. from a callback
            train - this is the training loop, it will invoke the training and validation phases, as well as callbacks and maintain stats
            evaluate - will run a forward pass on the test data
    """

    def __init__(self, model,
                       device,
                       loss_func,
                       optimizer,
                       scheduler,
                       metric_name_to_func,
                       train_data_loader,
                       val_data_loader,
                       train_steps,
                       val_steps,
                       num_epochs=50,
                       callbacks = [],
                       name = 'lpd',
                       optimizer_step_and_zero_grad_criteria=None):
        self.device = device
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_name_to_func = metric_name_to_func if metric_name_to_func else {}
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.num_epochs = num_epochs
        self.callbacks = callbacks
        self.name = name
        self.optimizer_step_and_zero_grad_criteria = optimizer_step_and_zero_grad_criteria or (lambda trainer: True)

        self.epoch = 0
        self.sample_count = 0
        self.sample_count_in_epoch = 0
        self.iteration = 0
        self.iteration_in_epoch = 0
        self._stopped = False

        self.state = State.EXTERNAL
        self.phase = Phase.IDLE
        self.train_stats = TrainerStats(self.metric_name_to_func)
        self.train_last_loss_object = None
        self.val_stats = TrainerStats(self.metric_name_to_func)
        self.val_last_loss_object = None
        self.test_stats = TrainerStats(self.metric_name_to_func)
        self.test_last_loss_object = None

    def _train_handler(self, loss, batch_size):
        self.sample_count += batch_size
        self.sample_count_in_epoch += batch_size
        self.iteration += 1
        self.iteration_in_epoch += 1
        self.train_last_loss_object = loss
        loss.backward()
        if self.optimizer_step_and_zero_grad_criteria(self):
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _val_handler(self, loss, batch_size):
        self.val_last_loss_object = loss

    def _test_handler(self, loss, batch_size):
        self.test_last_loss_object = loss

    def _labels_handler(self, labels):
        return labels.to(self.device)

    def _inputs_handler(self, inputs):
        if isinstance(inputs, list):
            # MULTIPLE INPUTS CONSTRUCTED IN A LIST
            return [x.to(self.device) for x in inputs]
        #SINGLE INPUT
        return [inputs.to(self.device)]

    def _get_tqdm_description(self):
        if self.state == State.TEST:
            return f'[{self.state}]'
        elif self.state == State.VAL:
            return f'[Val   epoch {self.epoch}/{self.num_epochs}]'
        else: #TRAIN
            return f'[Train epoch {self.epoch}/{self.num_epochs}]'

    def _fwd_pass_base(self, data_loader, steps, state_handler, stats):
        stats.reset()
        loop = tqdm(data_loader, total=steps-1)
        self.sample_count_in_epoch = 0  # CAN BE INVOKED ON ALL STATES
        self.iteration_in_epoch = 0     # CAN BE INVOKED ON ALL STATES
        for inputs,labels in loop:
            steps -= 1

            self.phase = Phase.BATCH_BEGIN
            self._invoke_callbacks()

            x = self._inputs_handler(inputs)
            y = self._labels_handler(labels)
            batch_size = len(y)
            outputs = self.model(*x)
            loss = self.loss_func(outputs, y)
            stats.add_loss(loss, batch_size)
            stats.add_metrics(outputs, y, batch_size)
            state_handler(loss, batch_size)

            self.phase = Phase.BATCH_END
            self._invoke_callbacks()

            loop.set_description(self._get_tqdm_description())
            loop.set_postfix(loss=stats.get_loss(), metrics=stats.get_metrics())

            if self._stopped:
                break

            if steps == 0:
                break

    def _fwd_pass_test(self, test_data_loader, test_steps):
        if self._stopped:
            return
            
        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            self._fwd_pass_base(test_data_loader, test_steps, self._test_handler, self.test_stats)

    def _fwd_pass_val(self):
        if self._stopped or self.val_data_loader is None or self.val_steps == 0:
            return

        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            self._fwd_pass_base(self.val_data_loader, self.val_steps, self._val_handler, self.val_stats)

    def _fwd_pass_train(self):
        if self._stopped:
            return
        self.model.train() #MARK STATUS AS TRAIN
        self._fwd_pass_base(self.train_data_loader, self.train_steps, self._train_handler, self.train_stats)

    def _invoke_callbacks(self):
        if self._stopped:
            return
        context = CallbackContext(self)
        for callback in self.callbacks:
            if callback.should_apply_on_phase(context) and \
               callback.should_apply_on_state(context):
                callback(context)

    def save_trainer(self, dir_path, file_name, msg='', verbose=1):
        full_path = dir_path + file_name
        if verbose:
            if msg:
                print(f'[Trainer {self.name}] - {msg}')
            else:
                print(f'[Trainer {self.name}] - Saving to {full_path}')

        if not fu.is_folder_exists(dir_path):
            fu.create_folder(dir_path)

        trainer_checkpoint = {
                            'model': self.model.state_dict(),
                            'loss_func': self.loss_func.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler':  self.scheduler.state_dict() if self.scheduler else None,
                            'metric_name_to_func': self.metric_name_to_func,
                            'callbacks': self.callbacks,
                            'name': self.name,
                            'epoch': self.epoch,
                            'num_epochs': self.num_epochs,
                            'iteration': self.iteration,
                            'train_stats': self.train_stats,
                            'val_stats': self.val_stats,
                            'test_stats': self.test_stats
                            }
        T.save(trainer_checkpoint, full_path)

    @staticmethod
    def load_trainer(dir_path,
                     file_name,
                     model, 
                     device,
                     loss_func, 
                     optimizer, 
                     scheduler, 
                     train_data_loader,
                     val_data_loader,
                     train_steps,
                     val_steps):
        full_path = dir_path + file_name
        checkpoint = T.load(full_path)
        print(f'[Trainer] - Loading from {full_path}')
        model.load_state_dict(checkpoint['model'])
        loss_func.load_state_dict(checkpoint['loss_func'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        trainer = Trainer(model=model, 
                       device=device, 
                       loss_func=loss_func, 
                       optimizer=optimizer,
                       scheduler=scheduler,
                       metric_name_to_func=checkpoint['metric_name_to_func'], 
                       train_data_loader=train_data_loader, 
                       val_data_loader=val_data_loader,
                       train_steps=train_steps,
                       val_steps=val_steps,
                       num_epochs=checkpoint['num_epochs'],
                       callbacks=checkpoint['callbacks'],
                       name=checkpoint['name'])
        
        trainer.epoch = checkpoint['epoch']
        trainer.iteration = checkpoint['iteration']
        trainer.train_stats = checkpoint['train_stats']
        trainer.val_stats = checkpoint['val_stats']
        trainer.test_stats = checkpoint['test_stats']
        
        return trainer

    def summary(self):
        print(f'Trainer - {self.name}')
        print('Model Summary - ')
        print(self.model)
        print('')

        print('defined callbacks:')
        for c in self.callbacks:
            print(c.get_description())
        print('')

        print("parameters name and device:")
        for p in self.model.named_parameters():
            print(f'name: {p[0]}, device: {p[1].device}')
            # print(p[1].data)

        print('')
        print('optimizer', type(self.optimizer))
        print('')
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_total_params_requires_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('pytorch_total_params_requires_grad', pytorch_total_params_requires_grad)

    def stop(self):
        self._stopped = True

    def train(self):
        self._stopped = False
        self.state = State.EXTERNAL
        self.phase = Phase.TRAIN_BEGIN
        self._invoke_callbacks()

        for _ in range(1, self.num_epochs + 1):
            self.epoch += 1

            self.phase = Phase.EPOCH_BEGIN
            self._invoke_callbacks()

            self.state = State.TRAIN
            self._fwd_pass_train()
            self.state = State.VAL
            self._fwd_pass_val()
            self.state = State.EXTERNAL

            self.phase = Phase.EPOCH_END
            self._invoke_callbacks()

            if self._stopped:
                break

        self.phase = Phase.TRAIN_END
        self._invoke_callbacks()
        self.phase = Phase.IDLE

    def evaluate(self, test_data_loader, test_steps):
        self._stopped = False
        self.phase = Phase.TEST_BEGIN
        self._invoke_callbacks()
        self.state = State.TEST
        self._fwd_pass_test(test_data_loader, test_steps)
        self.state = State.EXTERNAL
        self.phase = Phase.TEST_END
        self._invoke_callbacks()
        self.phase = Phase.IDLE



