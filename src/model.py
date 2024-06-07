import numpy as np 


## ====== Torch imports ======
import torch
import pytorch_lightning as pl 
import torch.nn as nn 
import torch.optim as optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch.utils.data
from torch.nn import functional as F
import lightning.pytorch.loggers
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import F1Score
from sklearn.metrics import f1_score, confusion_matrix

class LModel(pl.LightningModule):
    def __init__(self,
                dim: int = None,
                output_dim: int = 8,
                batch_size: int = 20,
                lr: float = 1e-2,
                weight_decay: float=1e-3,
                ):
        super().__init__()
        self.init_dim = dim,
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay 
        self.output_dim = output_dim,
        #self.loss_fun = nn.CrossEntropyLoss()
        self.loss_fun = nn.BCELoss()
        self.f1 = F1Score(task='binary', average='micro')
        self.test_preds = None
        self.test_labels = None
        
        self.embedder = nn.Sequential(
            nn.Linear(dim,2**12),
            nn.LeakyReLU(),
            # nn.Linear(2**12, 2**10),
            # nn.LeakyReLU(),
            # nn.Linear(2**10, 2**8),
            # nn.LeakyReLU(),
            nn.Linear(2**12, 2**8),
            nn.LeakyReLU(),
            nn.Linear(2**8, 2**6),
            nn.LeakyReLU(),
            nn.Linear(2**6, output_dim),
            nn.LeakyReLU()
        )
        self.classifier = nn.Sequential(
            #nn.Linear(2**4,2**3),
            #nn.LeakyReLU(),
            nn.Linear(output_dim,1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.embedder(x)
        x = self.classifier(x)  
        return x

    def on_train_epoch_start(self) -> None:
        self.train_loss = 0
        
    def training_step(self, batch):
        X_batch, y_batch = batch
        y = y_batch.unsqueeze(1)
        #y_float = y.float()
        X_embedded = self.embedder(X_batch)
        y_pred = self.classifier(X_embedded)
        loss = self.loss_fun(y_pred, y)
        f1 = self.f1(y_pred, y)
        
        #loss = self.loss_fun(y_pred, y_batch)
        # loss = (self.loss_fun(y_pred,y_batch) + torch.sum(torch.cat([torch.flatten(torch.abs(x)) for x in self.embedder.parameters() ]))*.0016
        #         + torch.sum(torch.cat([torch.flatten(torch.square(x)) for x in self.embedder.parameters() ]))*.00255 )
        # acc = (y_pred.max(1).indices == y_batch.max(1).indices).sum().item()/y_pred.shape[0]
        self.log("train_loss", 
                loss, 
                prog_bar=False, 
                on_step=False, 
                on_epoch=True)
        self.log("train_acc", 
                f1, 
                prog_bar=False, 
                on_step=False, 
                on_epoch=True)
        

        return loss
        
    def validation_step(self, batch):
        print("Entered validation")
        X_batch, y_batch = batch
        y = y_batch.unsqueeze(1)
        #y_float = y.float()
        X_embedded = self.embedder(X_batch)
        y_pred = self.classifier(X_embedded)
        #loss = self.loss_fun(y_pred,y_batch)
        loss = self.loss_fun(y_pred, y)
        f1 = self.f1(y_pred, y)
        print(f1)
        self.log("val_loss",
                loss, 
                prog_bar=False,
                on_step=False,
                on_epoch=True)
        self.log("val_acc", 
                f1,
                prog_bar=False,
                on_step=False,
                on_epoch=True)
        return loss
    
    def configure_optimizers(self) ->   OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(list(self.embedder.parameters()) +
                                        list(self.classifier.parameters()),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay) 
        return optimizer
    
    def test_step(self, batch):
        X_batch,y_batch = batch
        y = y_batch.unsqueeze(1)
        #y_float = y.float()
        y_pred = self(X_batch)
        loss = self.loss_fun(y_pred, y)
        f1 = self.f1(y_pred, y)
        #acc = (y_pred.max(1).indices == y_batch.max(1).indices).sum().item()/y_pred.shape[0]
        #X_embedded = self.embedder(X)
        
        self.log("test_loss", 
                loss, 
                on_step=False,
                on_epoch=True)
        self.log("test_acc",
                f1,
                on_step=False, 
                on_epoch=True)
        
        return {'loss': loss, 
                'f1score': f1}