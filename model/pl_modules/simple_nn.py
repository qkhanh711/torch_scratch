
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class Net(pl.LightningModule):

    def __init__(self, in_dims, n_classes=10,
                 n_layer_1=128, n_layer_2=256, lr=1e-4):
        super().__init__()
        self.layer_1 = nn.Linear(np.prod(in_dims), n_layer_1)
        self.layer_2 = nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = nn.Linear(n_layer_2, n_classes)
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.valid_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.test_acc  = torchmetrics.Accuracy(num_classes=n_classes)

    def forward(self, x):
        batch_size, *dims = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, xs, ys):
        logits = self(xs)  
        loss = F.nll_loss(logits, ys)
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        self.train_acc(logits, ys)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss
    
    def train_dataloader(self):
        return self.train_loader
    
    def configure_callbacks(self):
        return super().configure_callbacks()
    
    def configure_optimizers(self):
        return super().configure_optimizers()


class Net2(pl.LightningModule):

    def __init__(self, in_dims, n_classes=10,
                 n_layer_1=128, n_layer_2=256, lr=1e-4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(in_dims), n_layer_1),
            nn.ReLU(),
            nn.Linear(n_layer_1, n_layer_2),
            nn.ReLU(),
            nn.Linear(n_layer_2, n_classes)
        )
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        batch_size, *dims = x.size()
        x = x.view(batch_size, -1)
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, xs, ys):
        logits = self(xs)  
        loss = F.nll_loss(logits, ys)
        return logits, loss