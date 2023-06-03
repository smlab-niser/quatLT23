import torch
import torch.nn as nn
from tqdm import trange
import wandb

from utils.training import one_epoch

class LR_Sched:
    def __init__(self, optimiser, lr_schedule):
        self.optimiser = optimiser
        self.lr_schedule = lr_schedule

    def step(self, epoch):
        if epoch in self.lr_schedule:
            self.optimiser.param_groups[0]["lr"] = self.lr_schedule[epoch]


class Trainer_img64:
    def __init__(
        self,
        model,  # model instance to train
        hparams,  # hyperparameters dictionary
        optimiser=torch.optim.SGD,  # optimiser class name
        loss_fn=nn.CrossEntropyLoss,  # loss function class name
        scheduler=LR_Sched,  # scheduler class name
        log=False,  # whether to log to wandb
        e_offset=0,  # epoch offset
    ):
        self.model = model
        self.hparams = hparams
        self.optimiser = optimiser(model.parameters(), lr=0.1, momentum=hparams["momentum"], weight_decay=hparams["weight_decay"], nesterov=True)
        self.loss_fn = loss_fn()
        self.scheduler = scheduler(
            self.optimiser,
            {
                0: 0.1,
                11: 0.01,
                13: 0.001,
            }
        )
        self.log = log
        self.e_offset = e_offset

    def one_iter(self, batch_x, batch_y, epoch):
        self.scheduler.step(epoch)
        loss = one_epoch(self.model, batch_x, batch_y, self.optimiser, self.loss_fn, self.GPU)
        if self.log:
            wandb.log({f"loss {self.model.name}": loss})

    def train(self, training_generator, desc = "Training", end=None, end_args=None, start=None, start_args=None, after_epoch=None, after_epoch_args=None):
        
        if start is not None:
            start(*start_args)

        for epoch in trange(self.e_offset, self.hparams["num_epochs"], desc=desc, unit="epoch"):
            for batch_x, batch_y in training_generator:
                self.one_iter(batch_x, batch_y, epoch)
                
            if after_epoch is not None:
                after_epoch(*after_epoch_args)

        if end is not None:    
            end(*end_args)