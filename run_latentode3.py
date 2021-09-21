########################################################################
# train models
#
#
########################################################################

from src.data.data_loader import MIMICDataset,import_data,collate_fn_padd
from src.utils import setup_logger
from src.training.training_nn import *
from src.models.models import ODERNN,LatentODE1,LatentODE2,NeuralODE,ODEGRU,ODELSTM,LSTM,LatentODE3,NormalOutputNN
from src.utils import seed_everything
from src.data.data_scaler import PreProcess
from data.feature_sets import all_features,glycaemic_features

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import os
import math
import copy
import matplotlib.pyplot as plt
import optuna
from datetime import datetime
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=5,help='Maximum number of epochs to train a model.')
parser.add_argument('--hypertune', type=int, default=0,help='Perform hyperparameter tuning. 0=no, 1=yes.')
parser.add_argument('--gpu', type=int, default=0,help='Use GPU if available. 0=no, 1=yes.')
parser.add_argument('--test', type=int, default=0,help='Evaluate on hold out test data. 0=no, 1=yes.')
parser.add_argument('--batchsize', type=int, default=64,help='Batch size for training data, int >= 1')
parser.add_argument('--lrpatience', type=int, default=3,help='Learning rate reduction patience')

args = parser.parse_args()

# cmd / globals
SEQUENCE_LEN = 100
BATCH_SIZE = args.batchsize
OPTIM_TRIALS = 10
N_EPOCHS = args.nepochs
HYPERTUNE = args.hypertune
#FEATURE_DIM = 4
LR_PATIENCE = args.lrpatience
OUTPUT_DIM = 1
EARLY_STOPPING = 5
if args.gpu == 1 & torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("using: ",DEVICE)
if args.test == 1:
    ONTEST = True
else:
    ONTEST = False

# logger/save dir configuration
log_dir = os.path.join('experiments','mimic')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
setup_logger('log_train', os.path.join(log_dir, 'train.log'))
setup_logger('log_test',os.path.join(log_dir, 'test.log'))
log_train = logging.getLogger('log_train')
log_test = logging.getLogger('log_test')

# seed
seed_everything(1)

def train_and_evaluate(model, dataloaders, optim, scheduler,n_epochs,early_stopping_rounds,logging):
    dl_train = dataloaders['train']
    dl_val = dataloaders['validation']

    best_val_loss = float('inf')
    best_val_err = float('inf')
    best_state = None
    best_epoch = 0
    early_stop = 0
    with tqdm(total=n_epochs) as t:
        for i in range(n_epochs):
            logging.info("EPOCH {}".format(i))
            loss = model.train_single_epoch(dl_train,optim)
            logging.info("Training loss {:05.4f}".format(loss))
            loss_val,error_val,y_preds,y_tests,msks = model.evaluate(dl_val)
            logging.info("Validation loss {:05.4f}".format(loss_val))
            logging.info("Validation RMSE {:05.4f}".format(error_val))
            scheduler.step(loss_val)
            is_best = loss_val <= best_val_loss
            if is_best:
                best_val_loss = loss_val
                best_val_err = error_val
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = i
                early_stop = 0
                logging.info("Found new best loss at {}".format(i))
            else:
                early_stop += 1
                logging.info("Current best loss at {}".format(best_epoch))
            t.set_postfix(loss_and_val_err='{:05.4f} and {:05.4f}'.format(
                loss_val, error_val))
            print('\n')
            t.update()
            if early_stop > early_stopping_rounds:
                logging.info("Stopping early")
                break
    return best_state, best_val_loss,best_val_err

def define_model(trial):

    hidden_dim = trial.suggest_int("hidden_dim", 2, 20)
    dropout_p = trial.suggest_float("dropout_p", 0.05, 0.75) 
    
    log_train.info("Hyperparameter tuning...")
    log_train.info("hidden dim:%s" % hidden_dim)
    log_train.info("dropout:%s" % dropout_p)

    model = NET(FEATURE_DIM, hidden_dim, dropout_p, OUTPUT_DIM,BATCH_SIZE,DEVICE).to(DEVICE)

    return model

def objective(trial):
    
    # model 
    model = define_model(trial)
    
    # optimisation
    optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    l2_penalty = trial.suggest_float("l2_penalty", 1e-8, 1e-1, log=True)    
    model_optim = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # log
    log_train.info("learning rate:%s" % lr)
    log_train.info("l2_penalty:%s" % l2_penalty)
    log_train.info("model_optim:\n:%s" % model_optim)
    log_train.info("model:\n:%s" % model)

    # fixed
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min',patience=LR_PATIENCE,verbose=True)

    # train and evaluate
    dl_train = dataloaders['train']
    dl_val = dataloaders['validation']
    
    best_val_loss = float('inf')
    best_val_err = float('inf')

    for epoch in range(N_EPOCHS):
        logging.info("EPOCH {}".format(epoch))
        loss = model.train_single_epoch(dl_train,model_optim,epoch=epoch)
        loss_val,error_val,y_preds,y_vals,msks = model.evaluate(dataloaders["validation"])
        log_train.info("Validation loss {:05.4f}".format(loss_val))
        log_train.info("Validation RMSE {:05.4f}".format(error_val))

        optim_scheduler.step(loss_val)
            
        trial.report(loss_val, epoch)
        
        is_best = loss_val <= best_val_loss
        if is_best:
            best_val_loss = loss_val
            best_val_err = error_val
            best_epoch = epoch
            early_stop = 0
            logging.info("Found new best loss at {}".format(epoch))
        else:
            early_stop += 1
            logging.info("Current best loss at {}".format(best_epoch))
        if early_stop > EARLY_STOPPING:
            logging.info("Stopping early")
            break
    
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return(loss_val)

def glc_transform(x):
    x = x.copy()
    x[x > 0] = np.log(x[x > 0]) - np.log(140)
    return x

def glc_invtransform(x):
    x = x.copy()
    x = np.exp(x + np.log(140))
    return x

ginv = glc_invtransform

if __name__ == '__main__':
    """
    Train
    """
    FEATURES = ['glc', 'input_short_injection', 'input_short_push', 'input_intermediate', 'input_long', 'input_hrs']
    FEATURE_DIM = len(FEATURES)
    print(FEATURES)
    print(FEATURE_DIM)

    # data configuration
    df = import_data('data/train.csv')
    df = df.iloc[0:5000,:]
    df = df.loc[~df.glc.isnull()].copy()
    train_ids, valid_ids = train_test_split(df.icustay_id.unique(),test_size=0.1)
    df_train = df.loc[df.icustay_id.isin(train_ids)]
    #df_valid = df.loc[df.icustay_id.isin(valid_ids)]
    df_valid = df_train
    preproc = PreProcess(FEATURES,QuantileTransformer())
    preproc.fit(df_train)
    df_train = preproc.transform(df_train)
    df_valid = preproc.transform(df_valid)
    dl_train = DataLoader(MIMICDataset(df_train,FEATURES,pad=SEQUENCE_LEN),
                          batch_size=BATCH_SIZE,collate_fn=collate_fn_padd)
    dl_valid = DataLoader(MIMICDataset(df_valid,FEATURES,pad=SEQUENCE_LEN),
                          batch_size=BATCH_SIZE,collate_fn=collate_fn_padd)
    dataloaders = {'train':dl_train,'validation':dl_valid}
    print("training size:",df_train.shape)
    print("validation size:",df_valid.shape)
    locf_rmse = math.sqrt(np.mean((ginv(df_valid.loc[df_valid.msk==0,'glc']) - ginv(df_valid.loc[df_valid.msk==0,'glc_dt']))**2))
    print("LOCF RMSE: {:05.4f}".format(locf_rmse))
    log_train.info("LOCF RMSE: {:05.4f}".format(locf_rmse))
    
    # choose hyperparameters
    model = LatentODE3(NormalOutputNN,(1,5),(8,4,4),0.1,1,BATCH_SIZE,DEVICE)
    
    # record experiment
    log_train.info("model:\n:%s" % model)

    # train configuration
    lr = 1e-2
    l2_penalty = 1e-5
    optimizer = "Adam"
    model_optim = getattr(optim, optimizer)(model.parameters(), lr=lr)
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min',patience=LR_PATIENCE,verbose=True)

    # train and evaluate
    best_state, best_val_loss,best_val_err = train_and_evaluate(model, dataloaders, 
                                                    model_optim,optim_scheduler, N_EPOCHS,EARLY_STOPPING,log_train)
    
    # save a model
    model_name = args.net + '.pth'
    state = {
        'state_dict': best_state
    }
    torch.save(state, os.path.join(log_dir, model_name))