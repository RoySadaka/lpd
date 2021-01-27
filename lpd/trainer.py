import torch as T
from tqdm import tqdm
from lpd.metrics import MetricBase
from lpd.callbacks import CallbackContext, CollectOutputs, LossOptimizerHandlerBase
from lpd.enums import State, Phase
from lpd.trainer_stats import TrainerStats, StatsResult
from lpd.input_output_label import InputOutputLabel
import lpd.utils.file_utils as fu
from lpd.extensions.custom_schedulers import DoNothingToLR

class Trainer():
    """
        class that maintains all the participating objects and stats during training and evaluation

        Args:
            model - your model (nn.Module)
            device - the device to send the inputs/labels to
            loss_func - the model's loss function
            optimizer - the model's optimizer
            scheduler - the model's scheduler (make sure you add SchedulerStep to your callbacks),
                        pass None if you dont need scheduler
            metrics - metric or a list of metrics
            train_data_loader - an iterable or generator to get the next train data batch
            val_data_loader - an iterable or generator to get the next val data batch
            train_steps - total number of steps (batches) before declaring the epoch as finished
            val_steps - total number of steps (batches) before declaring the epoch as finished
            callbacks - list of lpd.callbacks to apply during the differrent training phases
                        callbacks will be executed by the order of the list 
            name - a friendly identifier

        Methods:
            save_trainer - saving the full trainer state to a file
            load_trainer - for creating a new Trainer instance from a saved Trainer checkpoint
            summary - will print information about the trainer and the model
            stop - will indicate this trainer to stop, e.g. from a callback
            train - this is the training loop, it will invoke the training and validation phases, as well as callbacks and maintain stats
            evaluate - will run a forward pass on the test data
            predict_sample - make prediction on single sample
            predict_batch - make prediction on single batch
            predict_data_loader - make prediction on data loader (DataLoader/Iterable/Generator)
    """

    def __init__(self, model,
                       device,
                       loss_func,
                       optimizer,
                       scheduler,
                       metrics,
                       train_data_loader,
                       val_data_loader,
                       train_steps,
                       val_steps,
                       callbacks = None,
                       name = 'lpd'):
        self.device = device
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler else DoNothingToLR()
        self.metrics = metrics if metrics else []
        self._validate_metrics()
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.callbacks = callbacks if callbacks else []
        self.name = name

        self.epoch = 0
        self.sample_count = 0
        self.sample_count_in_epoch = 0
        self.iteration = 0
        self.iteration_in_epoch = 0

        self.state = State.EXTERNAL
        self.phase = Phase.IDLE
        self.train_stats = TrainerStats(self.metrics)
        self.train_last_loss = None
        self.val_stats = TrainerStats(self.metrics)
        self.val_last_loss = None
        self.test_stats = TrainerStats(self.metrics)
        self.test_last_loss = None

        self._stopped = False

        self._last_data = {s:InputOutputLabel() for s in State}

        self._total_num_epochs = 0

    def _validate_metrics(self):
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]
            
        for metric in self.metrics:
            if not isinstance(metric, MetricBase):
                raise ValueError(f'[Trainer] - one of the metrics is of type {type(metric)}, expected {MetricBase}')

    def _train_callbacks_validation(self):
        has_loss_optimizer_handler = False
        for cb in self.callbacks:
            if isinstance(cb, LossOptimizerHandlerBase):
                has_loss_optimizer_handler = True

        if not has_loss_optimizer_handler:
            raise ValueError('[Trainer][train] - callbacks not containing LossOptimizerHandlerBase, either use LossOptimizerHandler callback, or create your own callback derived from LossOptimizerHandlerBase')

    def _train_handler(self, loss, batch_size):
        self.sample_count += batch_size
        self.sample_count_in_epoch += batch_size
        self.iteration += 1
        self.iteration_in_epoch += 1
        self.train_last_loss = loss

    def _val_handler(self, loss, batch_size):
        self.val_last_loss = loss

    def _test_handler(self, loss, batch_size):
        self.test_last_loss = loss

    def _predict_handler(self, loss, batch_size):
        pass

    def _labels_handler(self, labels):
        return labels.to(self.device)

    def _inputs_handler(self, inputs):
        if isinstance(inputs, list):
            # MULTIPLE INPUTS CONSTRUCTED IN A LIST
            return [x.to(self.device) for x in inputs]
        #SINGLE INPUT
        return [inputs.to(self.device)]

    def _get_epoch_description(self):
        if self.state == State.TEST or self.state == State.PREDICT:
            desc = f'[{self.state}]'
        elif self.state == State.VAL:
            desc = f'[Val   epoch {self.epoch}/{self._total_num_epochs}]'
        else: #TRAIN
            desc = f'[Train epoch {self.epoch}/{self._total_num_epochs}]'
        return desc

    def _tqdm_description(self, loop, stats):
        desc = self._get_epoch_description()
        loop.set_description(desc)
        if self.state != State.PREDICT:
            stats_result = StatsResult('', stats)
            if stats_result.metrics:
                loop.set_postfix(loss=stats_result.loss, metrics=stats_result.metrics)
            else:
                loop.set_postfix(loss=stats_result.loss)
            
    def _print_verbos_2(self, stats, verbose):
        if verbose != 2:
            return
        desc = self._get_epoch_description()
        stats_result = StatsResult('', stats)
        if stats_result.metrics:
            metrics_str = f', metrics={stats_result.metrics}'
        else:
            metrics_str = ''
        print(f'{desc}, loss={stats_result.loss}{stats_result.metrics_str}')

    def _prepare_batch(self, batch):
        if self.state == State.PREDICT:
            inputs,labels = batch, T.zeros(len(batch)) #FAKE LABELS FOR CODE CONSISTENCY, NO ACTUAL USE TO THEM 
        else:
            inputs,labels = batch
        return inputs,labels

    def _fwd_pass_base(self, data_loader, steps, state_handler, stats, loss_f, last_data, verbose):
        stats.reset()
        loop = tqdm(data_loader, total=steps-1, disable=verbose-1) #verbose-1 maps 0,2=>True, 1=>False
        self.sample_count_in_epoch = 0
        self.iteration_in_epoch = 0
        for batch in loop:
            inputs,labels = self._prepare_batch(batch)
            steps -= 1

            self.phase = Phase.BATCH_BEGIN
            self._invoke_callbacks()

            x = self._inputs_handler(inputs)
            y = self._labels_handler(labels)
            batch_size = len(y)
            outputs = self.model(*x)

            last_data.update(x, outputs, y)
            self._last_data[State.EXTERNAL].update(x, outputs, y) #ALWAYS UPDATE EXTERNAL WITH LATEST DATA SO WE CAN CHOOSE TO USE THE LAST SAMPLE ONLY IN SPECIFIC PHASE/STATE
            
            loss = loss_f(outputs, y)
            stats.add_loss(loss, batch_size)
            stats.add_metrics(outputs, y, batch_size)
            state_handler(loss, batch_size)

            self.phase = Phase.BATCH_END
            self._invoke_callbacks()

            self._tqdm_description(loop, stats)

            if self._stopped:
                break

            if steps == 0:
                break

    def _fwd_pass_predict(self, predict_data_loader, predict_steps):
        if self._stopped:
            return
            
        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            self._fwd_pass_base(predict_data_loader, 
                                predict_steps, 
                                self._predict_handler, 
                                stats=TrainerStats({}), # NO STATS
                                loss_f=lambda outputs, y: T.Tensor([0]), # DO NOTHING LOSS
                                last_data=self._last_data[State.PREDICT],
                                verbose=0) 

    def _fwd_pass_test(self, test_data_loader, test_steps, verbose):
        if self._stopped:
            return
            
        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            self._fwd_pass_base(test_data_loader, 
                                test_steps, 
                                self._test_handler, 
                                self.test_stats, 
                                loss_f=self.loss_func,
                                last_data=self._last_data[State.TEST],
                                verbose=verbose)

    def _fwd_pass_val(self, verbose):
        if self._stopped or self.val_data_loader is None or self.val_steps == 0:
            return

        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            self._fwd_pass_base(self.val_data_loader, 
                                self.val_steps, 
                                self._val_handler, 
                                self.val_stats, 
                                loss_f=self.loss_func,
                                last_data=self._last_data[State.VAL],
                                verbose=verbose)

    def _fwd_pass_train(self, verbose):
        if self._stopped:
            return
            
        self.model.train() #MARK STATUS AS TRAIN
        self._fwd_pass_base(self.train_data_loader, 
                            self.train_steps, 
                            self._train_handler, 
                            self.train_stats, 
                            loss_f=self.loss_func,
                            last_data=self._last_data[State.TRAIN],
                            verbose=verbose)

        self._print_verbos_2(self.train_stats, verbose)

    def _invoke_callbacks(self):
        if self._stopped:
            return
        context = CallbackContext(self)
        for callback in self.callbacks:
            if callback.should_apply_on_phase(context) and \
               callback.should_apply_on_state(context):
                callback(context)

    def _predict(self, inputs_data_loader, steps):
        """
            return numpy array(s) of current trainer model's predictions.
        """

        # ADD COLLECT OUTPUTS CALLBACK 
        collect_outputs = CollectOutputs(apply_on_phase=Phase.BATCH_END, apply_on_states=State.PREDICT)
        self.callbacks.append(collect_outputs)

        self._stopped = False
        self.phase = Phase.PREDICT_BEGIN
        self._invoke_callbacks()
        self.state = State.PREDICT
        self._fwd_pass_predict(inputs_data_loader, steps)
        self.state = State.EXTERNAL
        self.phase = Phase.PREDICT_END
        self._invoke_callbacks()
        self.phase = Phase.IDLE

        # REMOVE COLLECT OUTPUTS CALLBACK 
        self.callbacks.pop()
        
        outputs = collect_outputs.get_outputs_for_state(State.PREDICT)
        return outputs


    # ---------------- PUBLIC ---------------- 


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
                            'metrics': self.metrics,
                            'callbacks': self.callbacks,
                            'name': self.name,
                            'epoch': self.epoch,
                            'iteration': self.iteration,
                            'sample_count': self.sample_count,
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
        checkpoint = T.load(full_path, map_location=device)
        print(f'[Trainer] - Loading from {full_path}')
        model.load_state_dict(checkpoint['model'])
        loss_func.load_state_dict(checkpoint['loss_func'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = scheduler if scheduler else DoNothingToLR()
        scheduler.load_state_dict(checkpoint['scheduler'])

        trainer = Trainer(model=model, 
                       device=device, 
                       loss_func=loss_func, 
                       optimizer=optimizer,
                       scheduler=scheduler,
                       metrics=checkpoint['metrics'], 
                       train_data_loader=train_data_loader, 
                       val_data_loader=val_data_loader,
                       train_steps=train_steps,
                       val_steps=val_steps,
                       callbacks=checkpoint['callbacks'],
                       name=checkpoint['name'])
        
        if 'epoch' in checkpoint:
            trainer.epoch = checkpoint['epoch']
        if 'iteration' in checkpoint:
            trainer.iteration = checkpoint['iteration']
        if 'sample_count' in checkpoint:
            trainer.sample_count = checkpoint['sample_count']
        if 'train_stats' in checkpoint:
            trainer.train_stats = checkpoint['train_stats']
        if 'val_stats' in checkpoint:
            trainer.val_stats = checkpoint['val_stats']
        if 'test_stats' in checkpoint:
            trainer.test_stats = checkpoint['test_stats'] 
        
        return trainer

    def summary(self):
        print(f'Trainer - {self.name}')

        print('Defined callbacks:')
        for c in self.callbacks:
            print(c.get_description())
        print('')

        print("Parameters name and device:")
        for p in self.model.named_parameters():
            print(f'name: {p[0]}, device: {p[1].device}')
            # print(p[1].data)

        print('')
        print('Optimizer', type(self.optimizer))
        
        print('')
        print('Loss', type(self.loss_func))
        
        print('')
        print('Model Summary - ')
        print(self.model)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        total_params_requires_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print('')
        print(f'Total params: {total_params}')
        print(f'Trainable params: {total_params_requires_grad}')
        print(f'Non-trainable params: {total_params_requires_grad-total_params}')

    def stop(self):
        self._stopped = True

    def train(self, num_epochs, verbose=1):
        """
            num_epochs - amount of epochs to run
            verbose: 0 = no progress bar, 1 = progress bar, 2 = one line per epoch
        """
        self._total_num_epochs = self.epoch + num_epochs
        self._train_callbacks_validation()
        self._stopped = False
        self.state = State.EXTERNAL
        self.phase = Phase.TRAIN_BEGIN
        self._invoke_callbacks()

        for _ in range(num_epochs):
            self.epoch += 1

            self.phase = Phase.EPOCH_BEGIN
            self._invoke_callbacks()

            self.state = State.TRAIN
            self._fwd_pass_train(verbose)
            self.state = State.VAL
            self._fwd_pass_val(verbose)
            self.state = State.EXTERNAL

            self.phase = Phase.EPOCH_END
            self._invoke_callbacks()

            if self._stopped:
                break

        self.phase = Phase.TRAIN_END
        self._invoke_callbacks()
        self.phase = Phase.IDLE

    def evaluate(self, test_data_loader, test_steps, verbose=1):
        """
            verbose: 0 = no progress bar, 1 = progress bar
        """
        self._stopped = False
        self.phase = Phase.TEST_BEGIN
        self._invoke_callbacks()
        self.state = State.TEST
        self._fwd_pass_test(test_data_loader, test_steps, verbose)
        self.state = State.EXTERNAL
        self.phase = Phase.TEST_END
        self._invoke_callbacks()
        self.phase = Phase.IDLE

        return StatsResult(self.name, self.test_stats)

    def predict_sample(self, inputs):
        # MAKE BATCH WITH 1 SAMPLE USING unsqueeze
        outputs = self.predict_batch(inputs.unsqueeze(0))
        return outputs[0]

    def predict_batch(self, inputs):
        # MAKE ITERATOR WITH 1 BATCH 1 STEP
        outputs = self.predict_data_loader([inputs], 1)
        return outputs[0]

    def predict_data_loader(self, inputs_data_loader, steps):
        return self._predict(inputs_data_loader, steps)
