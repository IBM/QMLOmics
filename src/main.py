""" Quantum machine learning on neural network embeddings

    Returns:
        Performance metrics on neural network, support vector classifier, and quantum support vector classifier 
"""
### Author: Aritra Bose <a.bose@ibm.com>
### MIT license


### --- base class imports --- ###
import pandas as pd
import numpy as np
import argparse
import os
import copy
from time import strftime, gmtime
#import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('dark')

# ====== Torch imports ======
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl 
from torchmetrics import ConfusionMatrix, F1Score
# ====== Scikit-learn imports ======

from sklearn.svm import SVC
from sklearn.metrics import (
    auc,
    roc_curve,
    ConfusionMatrixDisplay,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# ====== Qiskit imports ======

from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC

# ====== Local imports ======
from model import LModel
from dataset import OmicsData


def parse_args(): 
    """Parse the input command line args using argparse 

    Returns:
        Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="quantum machine learning on multi-omics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str, 
        default=None, 
        help="Multi-omics data file"
    )
    parser.add_argument(
        "-cv",
        "--num_cv",
        type=int,
        default = 1, 
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "-e", "--epoch", 
        type=int, 
        default=100, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "-b", 
        "--batch_size", 
        type=int, 
        default=20, 
        help="Train/test batch size"
    )
    parser.add_argument(
        "-lr",
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate"
    )
    parser.add_argument(
        "-l2",
        "--weight_decay",
        type=float,
        default=1e-5,
        help="L2 regularization"
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=1,
        help="Number of iterations"
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=8,
        help="Number of dimensions for the neural network embedding"
    )
    parser.add_argument(
        "-c",
        "--C",
        type=int,
        default=1,
        help="Regularization parameter for SVC"
    )
    parser.add_argument(
        "-pq",
        "--pegasos",
        type=bool,
        default=False,
        help="Flag to use PegasosQSVC"
    )
    parser.add_argument(
        "-en",
        "--encoding",
        type=str, 
        default="ZZ", 
        choices=['ZZ', 'Z', 'P'],
        help="Econding for QML"
    )
    args = parser.parse_args()
    return args 

def validate_args(args):
    """Validate the arguments

    Args:
        args (dictionary): The argument dictionary as returned by parse_args(). 

    Raises:
        ValueError: Input file path error if incorrect path provided.
    """
    if args.file is None or os.path.exists(args.file) is None: 
        raise ValueError("Input file path error!")


def process_data(file):
    """Process the data file 

    Args:
        file (path): Path of the .csv file with the following column structure: 
                    [Sample ID, Genes..., label]
                    label should contain the header of y in the .csv file 

    Returns:
        numpy ndarrays pertaining to the splits of the training and held out test data. 
    """
    
    df = pd.read_csv(file)
    y = df['y'].values.astype(float)
    X = df[df.columns[1:-1]].values
    
    # held-out master split
    X_working, X_held_out, y_working, y_held_out = train_test_split(X,
                                                    y,
                                                    train_size=0.8,
                                                    shuffle=True)
    
    return X_working, y_working, X_held_out, y_held_out


# def compute_metrics(y_hat, y):
#     _, preds = torch.max(y_hat, 1)
#     f1_score = F1Score(y, preds, average='micro')
#     cm = ConfusionMatrix(y, preds)
    
#     return f1_score, cm       

def kfold_cross_validation(args, model, fname, X, y, k, early_stopping_patience, iter, **trainer_kwargs):
    """K Fold cross validation method to train the neural network model

    Args:
        args (dict): arguments dictionary with all the variables
        model (LModel): The model object of LModel class
        X (numpy ndarray): Training data 
        y (numpy array): Training labels
        k (int): Number of cross validation to be conducted
        early_stopping_patience (int): Patience for early stopping checks
        iter (int): number of iterations of the whole pipeline

    Returns:
        best_model_weights (numpy ndarray): best model weights after training and validation 
        best_train_index (list): train indices which led to best model  
    """
    kfold = KFold(n_splits=k, shuffle=True)
    best_model_weights = None
    best_train_index = None
    best_val_metric = float("-inf")  
    
    for fold, (train_index, val_index) in enumerate(kfold.split(X)): 
        print(f"Fold {fold+1}")
        print(len(train_index))
        print(len(val_index))
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        #create dataloaders 
        train_data = OmicsData(X_train,y_train)
        val_data = OmicsData(X_val, y_val)
        train_dataloader = DataLoader(train_data)
        val_dataloader = DataLoader(val_data)
        #rint(val_dataloader)
        
        checkpoint_callback = ModelCheckpoint(
                                        dirpath=f"checkpoints/{fname}/fold_{fold}",
                                        save_top_k=1, 
                                        monitor="val_loss",
                                        mode="min",
                                        )
        early_stopping = EarlyStopping(
                                    monitor="val_loss", 
                                    patience=early_stopping_patience,
                                    mode="min"
                                    )
        
        logger = TensorBoardLogger(save_dir="logs", name=f"{fname}_fold_{fold}")
        
        trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epoch,
        callbacks=[early_stopping, checkpoint_callback],
        accumulate_grad_batches=len(train_dataloader),
        check_val_every_n_epoch=10,
        logger=logger
        )
        
        trainer.fit(model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders= val_dataloader)
        
        val_metric = trainer.callback_metrics.get("val_acc")
        print(val_metric)
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_model_weights = model.state_dict()
            best_train_index = train_index.tolist()
            
    return best_model_weights, best_train_index

    
def training(args, fname, X, y, iter): 
    """Training method which calls the kfold cross validation code

    Args:
        args (dict): dictionary of arguments from input 
        fname (str): file name for storing checkpoints and embeddings
        X (numpy ndarray): Training data
        y (numpy array): Training labels
        iter (int): number of iterations to conduct

    Returns:
        embedded_train (numpy ndarray): Embedded training data of size samples x output dimension
        train_index (array): training indices 
        model (LModel): LModel object 
        model_weights (numpy ndarray): learned weights of the model
        
    """
    num_feats = X.shape[1]
    model = LModel(
        dim=num_feats, 
        output_dim = args.dim,
        batch_size=args.batch_size, 
        weight_decay=args.weight_decay,
        lr=args.lr
    )
    model_weights, train_index = kfold_cross_validation(args, 
                                                        model,
                                                        fname, 
                                                        X, 
                                                        y, 
                                                        args.num_cv, 
                                                        args.patience,
                                                        iter
                                                        )
    model.load_state_dict(model_weights)
    embedded_train = model.embedder(torch.tensor(X[train_index], dtype=torch.float32)).detach().numpy()
    #print(embedded_train.shape)
    
    return embedded_train, train_index, model, model_weights

def testing(X,y, model, model_weights):
    
    test_data = OmicsData(X, y)
    test_dataloader = DataLoader(test_data)
    model.load_state_dict(model_weights)
    X = torch.tensor(X, dtype=torch.float32) 
    embedded_test = model.embedder(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    print(embedded_test.shape)
    trainer = pl.Trainer()
    results = trainer.test(model=model, dataloaders=test_dataloader)
    
    return results, embedded_test

def compute_svc(X_train, y_train, X_test, y_test, c = 1):
    svc = SVC(C=c)
    # y_train = torch.argmax(torch.tensor(y_train, dtype=torch.float32),dim=1)
    # y_test = torch.argmax(torch.tensor(y_test, dtype=torch.float32),dim=1)
    svc_vanilla = svc.fit(X_train, y_train)
    labels_vanilla = svc_vanilla.predict(X_test)
    f1_svc = f1_score(y_test, labels_vanilla, average='micro')
    
    return f1_svc
    
def compute_QSVC(X_train, y_train, X_test, y_test, encoding='ZZ', c = 1, pegasos=False):
    
    service = QiskitRuntimeService(instance="accelerated-disc/internal/default") 
    backend = service.least_busy(simulator=False, operational=True)    
    # service = QiskitRuntimeService()    
    # backend = AerSimulator(method='statevector')
    algorithm_globals.random_seed = 12345

    feature_map = None

    if encoding == 'ZZ' :
        feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], 
                            reps=2, 
                            entanglement='linear')
    else: 
        if encoding == 'Z': 
            feature_map = ZFeatureMap(feature_dimension=X_train.shape[1], 
                            reps=2)
        if encoding == 'P': 
            feature_map = PauliFeatureMap(feature_dimension=X_train.shape[1], 
                            reps=2, entanglement='linear')

    sampler = Sampler(backend=backend, 
                    options={"shots": 1024}) 
    fidelity = ComputeUncompute(sampler=sampler)
    Qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    if pegasos == False: 
        qsvc = QSVC(quantum_kernel=Qkernel, C=c)
    else: 
        qsvc = PegasosQSVC(quantum_kernel=Qkernel, C=c)
    qsvc_model = qsvc.fit(X_train, y_train)
    labels_qsvc = qsvc_model.predict(X_test)
    f1_qsvc = f1_score(y_test, labels_qsvc, average='micro')

    return f1_qsvc

if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    file_name = os.path.basename(args.file).split('.')[0]
    results_iter = {}
    for i in range(args.iter):
        print("===== Iteration " + str(i+1) + " =====")
        #process data to obtain master split
        X_working,y_working,X_held_out,y_held_out = process_data(args.file)
        print("Training size: ", X_working.shape[0])
        print("Held out size: ", X_held_out.shape[0])
        
        fname = file_name + "_iter" + str(i)
        #get embedded training data and the best performing model weights using cross validation
        embedded_train, train_idx, model, model_weights = training(args,
                                                                fname,
                                                                X_working, 
                                                                y_working, 
                                                                i)
        fname_train = fname + "_train_embedding"
        np.save(f"checkpoints/{fname}/{fname_train}", embedded_train)
        fname_train_y = fname + "_train_target"
        np.save(f"checkpoints/{fname}/{fname_train_y}", y_working[train_idx])
        
        results_dict, embedded_test = testing(X_held_out, y_held_out, model, model_weights)
        results_nn = results_dict[0]
        print("NN results on held-out data:", results_nn['test_acc'])
        
        fname_test = fname + "_test_embedding"
        np.save(f"checkpoints/{fname}/{fname_test}", embedded_test)
        fname_test_y = fname + "_test_target"
        np.save(f"checkpoints/{fname}/{fname_test_y}", y_held_out)
        
        results_svc = compute_svc(
                                embedded_train, 
                                y_working[train_idx], 
                                embedded_test, 
                                y_held_out,
                                args.C
                                )

        print("SVC results on held-out data: " + str(results_svc))
        
        
        results_qsvc = compute_QSVC(
                                embedded_train, 
                                y_working[train_idx],
                                embedded_test,
                                y_held_out, 
                                args.encoding,
                                args.C
                                )     
        print("QSVC results on held-out data: " + str(results_qsvc))

        results_iter[i] = [results_nn['test_acc'], results_svc, results_qsvc]
    
    results_df = pd.DataFrame.from_dict(results_iter, orient='index')
    print(results_df)
    
    str_time = strftime("%Y-%m-%d-%H-%M", gmtime())
    of_name = file_name + "_" + str_time + "_Results.csv" 
    results_df.to_csv(of_name, index=False, header=['NN', 'SVC', 'QSVC'])
    max_memory_allocated = torch.cuda.max_memory_allocated()
    print(f"{max_memory_allocated/1024**3:.2f} GB of GPU memory allocated")
    
    