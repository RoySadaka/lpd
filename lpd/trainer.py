import torch as T
from tqdm import tqdm

import lpd.callbacks as tc
from lpd.trainer_stats import TrainerStats

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
        
        Methods:
            summary - will print information abour the trainer and the model
            stop_training - will indicate this trainer to stop train (e.g. from a callback) after the current epoch is done
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
                       callbacks = []):
        self.device = device
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_name_to_func = metric_name_to_func
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.num_epochs = num_epochs
        self.callbacks = callbacks

        self._current_epoch = 0
        self._should_stop_train = False

        self.train_stats = TrainerStats(self.metric_name_to_func)
        self.train_last_loss_object = None
        self.val_stats = TrainerStats(self.metric_name_to_func)
        self.val_last_loss_object = None
        self.test_stats = TrainerStats(self.metric_name_to_func)
        self.test_last_loss_object = None


    def _train_loss_opt_handler(self, loss):
        self.train_last_loss_object = loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _val_loss_opt_handler(self, loss):
        self.val_last_loss_object = loss

    def _test_loss_opt_handler(self, loss):
        self.test_last_loss_object = loss

    def _fwd_pass_base(self, phase_description, data_loader, steps, loss_opt_handler, stats):
        stats.reset()
        loop = tqdm(data_loader, total=steps-1)
        for X_batch,y_batch in loop:
            steps -= 1
            inputs = []
            for x in X_batch:
                inputs.append(x.to(self.device))
            y = y_batch.to(self.device)
            outputs = self.model(*inputs)
            loss = self.loss_func(outputs, y)
            stats.add_loss(loss)
            stats.add_metrics(outputs, y)
            loss_opt_handler(loss)
            
            loop.set_description(phase_description)
            loop.set_postfix(loss=stats.get_loss(), acc=stats.get_metrics())
            
            if steps == 0:
                break

    def _fwd_pass_test(self, test_data_loader, test_steps):
        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            phase_description = f'[Test]'
            self._fwd_pass_base(phase_description, test_data_loader, test_steps, self._test_loss_opt_handler, self.test_stats)

    def _fwd_pass_val(self):
        if self.val_data_loader is None or self.val_steps == 0:
            return

        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            phase_description = f'[Val   epoch {self._current_epoch}/{self.num_epochs}]'
            self._fwd_pass_base(phase_description, self.val_data_loader, self.val_steps, self._val_loss_opt_handler, self.val_stats)

    def _fwd_pass_train(self):
        self.model.train() #MARK STATUS AS TRAIN
        phase_description = f'[Train epoch {self._current_epoch}/{self.num_epochs}]'
        self._fwd_pass_base(phase_description, self.train_data_loader, self.train_steps, self._train_loss_opt_handler, self.train_stats)

    def _invoke_callbacks(self, phase):
        context = tc.CallbackContext(self)
        for cb in self.callbacks:
            if cb.cb_phase == phase:
                cb(context)


    def summary(self):
        print('[Model Summary] - ')
        print(self.model)

        print("parameters name and device:")
        for p in self.model.named_parameters():
            print(f'name: {p[0]}, device: {p[1].device}')
            # print(p[1].data)

        print('optimizer', type(self.optimizer))
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        pytorch_total_params_requires_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('pytorch_total_params_requires_grad', pytorch_total_params_requires_grad)

    def stop_training(self):
        #MARKS THIS TRAINER AS DONE, MOST LIKELY DUE TO A CALLBACK (E.G. EARLY-STOPPING)
        self._should_stop_train = True

    def train(self):
        self._invoke_callbacks(tc.CB_ON_TRAIN_BEGIN)
        self._current_epoch = 0
        for epoch in range(1, self.num_epochs + 1):
            self._current_epoch = epoch
            self._invoke_callbacks(tc.CB_ON_EPOCH_BEGIN)

            self._fwd_pass_train()
            self._fwd_pass_val()

            self._invoke_callbacks(tc.CB_ON_EPOCH_END)
            
            if self._should_stop_train:
                break
        
        self._invoke_callbacks(tc.CB_ON_TRAIN_END)

    def evaluate(self, test_data_loader, test_steps):
        self._fwd_pass_test(test_data_loader, test_steps)

