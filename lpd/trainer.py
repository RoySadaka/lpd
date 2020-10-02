import torch as T
from statistics import mean 
from tqdm import tqdm

import lpd.callbacks as tc
from lpd.trainer_stats import TrainerStats

class Trainer():
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
                       print_round_values_to = None):
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
        self.callbacks = callbacks
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.should_stop_train = False
        self.print_round_values_to = print_round_values_to

        self.train_loss_stats = None
        self.train_metric_name_to_stats = None
        self.val_loss_stats = None
        self.val_metric_name_to_stats = None
        self.test_loss_stats = None
        self.test_metric_name_to_stats = None

    def _train_loss_opt_handler(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _val_test_loss_opt_handler(self, loss):
        pass

    def _metrics_handler_in_epoch(self, y_pred, y_true, metric_name_to_stats):
        for metric_name, f in self.metric_name_to_func.items():
            value = f(y_pred, y_true)
            metric_name_to_stats[metric_name].add_value(value.item())

    def _fwd_pass_base(self, phase_description, data_loader, steps, loss_opt_handler):
        loss_stats = TrainerStats(self.print_round_values_to)
        metric_name_to_stats = {metric_name:TrainerStats(self.print_round_values_to) for metric_name,_ in self.metric_name_to_func.items()}
        loop = tqdm(data_loader, total=steps-1)
        for X_batch,y_batch in loop:
            steps -= 1
            inputs = []
            for x in X_batch:
                inputs.append(x.to(self.device))
            y = y_batch.to(self.device)
            outputs = self.model(*inputs)
            loss = self.loss_func(outputs, y)
            loss_stats.add_value(loss.item())
            self._metrics_handler_in_epoch(outputs, y, metric_name_to_stats)
            loss_opt_handler(loss)
            
            loop.set_description(phase_description)
            loop.set_postfix(loss=loss_stats.get_mean(), acc={metric_name:stats.get_mean() for metric_name, stats in metric_name_to_stats.items()})
            
            if steps == 0:
                break

        return loss_stats, metric_name_to_stats

    def _fwd_pass_test(self, test_data_loader, test_steps):
        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            phase_description = f'[Test]'
            self.test_loss_stats, self.test_metric_name_to_stats = self._fwd_pass_base(phase_description, test_data_loader, test_steps, self._val_test_loss_opt_handler)

    def _fwd_pass_val(self):
        if self.val_data_loader is None or self.val_steps == 0:
            return

        with T.no_grad():
            self.model.eval()  #MARK STATUS AS EVAL
            phase_description = f'[Val   epoch {self.current_epoch}/{self.num_epochs}]'
            self.val_loss_stats, self.val_metric_name_to_stats = self._fwd_pass_base(phase_description, self.val_data_loader, self.val_steps, self._val_test_loss_opt_handler)

    def _fwd_pass_train(self):
        self.model.train() #MARK STATUS AS TRAIN
        phase_description = f'[Train epoch {self.current_epoch}/{self.num_epochs}]'
        self.train_loss_stats, self.train_metric_name_to_stats = self._fwd_pass_base(phase_description, self.train_data_loader, self.train_steps, self._train_loss_opt_handler)

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
        self.should_stop_train = True

    def train(self):
        self._invoke_callbacks(tc.CB_ON_TRAIN_BEGIN)
        self.current_epoch = 0
        for epoch in range(1, self.num_epochs + 1):
            self.current_epoch = epoch
            self._invoke_callbacks(tc.CB_ON_EPOCH_BEGIN)

            self._fwd_pass_train()
            self._fwd_pass_val()

            self.scheduler.step(metrics=self.val_loss_stats.get_mean())
            # self.scheduler.step()

            self._invoke_callbacks(tc.CB_ON_EPOCH_END)
            
            if self.should_stop_train:
                break
        
        self._invoke_callbacks(tc.CB_ON_TRAIN_END)

    def evaluate(self, test_data_loader, test_steps):
        self._fwd_pass_test(test_data_loader, test_steps)
        test_mean_loss = self.test_loss_stats.get_mean()
        test_metrics = {metric_name:stats.get_mean() for metric_name,stats in self.test_metric_name_to_stats.items()}
        print(f'[Test Results] - loss: {test_mean_loss}, metric: {test_metrics}')

