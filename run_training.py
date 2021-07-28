########################################################################
# train models
#
#
########################################################################

from src.data.data_loader import MIMICDataset,import_data
from src.utils import setup_logger
from src.training.training_nn import *
from src.models.models import ODERNN,LatentODE1,LatentODE2,NeuralODE
from src.utils import seed_everything
from src.data.data_scaler import PreProcess
from data.feature_sets import all_features

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
parser.add_argument('--net', dest='net', choices=['ODERNN','LatentODE1','LatentODE2','NeuralODE'], default='NeuralODE',
                    help='Network architecture to use (see src/models/model.py)')
parser.add_argument('--hypertune', type=int, default=0,help='Perform hyperparameter tuning. 0=no, 1=yes.')
parser.add_argument('--gpu', type=int, default=0,help='Use GPU if available. 0=no, 1=yes.')
parser.add_argument('--test', type=int, default=0,help='Evaluate on hold out test data. 0=no, 1=yes.')
parser.add_argument('--batchsize', type=int, default=64,help='Batch size for training data, int >= 1')

args = parser.parse_args()

# cmd / globals
SEQUENCE_LEN = 100
BATCH_SIZE = args.batchsize
OPTIM_TRIALS = 10
N_EPOCHS = args.nepochs
HYPERTUNE = args.hypertune
#FEATURE_DIM = 4
OUTPUT_DIM = 1
EARLY_STOPPING = 3
nets = {'ODERNN': ODERNN,'LatentODE1':LatentODE1,'LatentODE2':LatentODE2,'NeuralODE':NeuralODE}
NET = nets[args.net]
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
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min',patience=0,verbose=True)

    # train and evaluate
    dl_train = dataloaders['train']
    dl_val = dataloaders['validation']
    
    best_val_loss = float('inf')
    best_val_err = float('inf')

    for epoch in range(N_EPOCHS):
        logging.info("EPOCH {}".format(epoch))
        loss = model.train_single_epoch(dl_train,model_optim)
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
    FEATURE_DIM = len(all_features())
    FEATURES = all_features()
    print(FEATURES)
    print(FEATURE_DIM)

    # data configuration
    df = pd.read_csv('data/train.csv')
    train_ids, valid_ids = train_test_split(df.icustay_id.unique(),test_size=0.1)
    df_train = df.loc[df.icustay_id.isin(train_ids)]
    df_valid = df.loc[df.icustay_id.isin(valid_ids)]
    preproc = PreProcess(FEATURES,QuantileTransformer())
    preproc.fit(df_train)
    df_train = preproc.transform(df_train)
    df_valid = preproc.transform(df_valid)
    dl_train = DataLoader(MIMICDataset(df_train,FEATURES,pad=SEQUENCE_LEN),batch_size=BATCH_SIZE,shuffle=True)
    dl_valid = DataLoader(MIMICDataset(df_valid,FEATURES,pad=SEQUENCE_LEN),batch_size=BATCH_SIZE)
    dataloaders = {'train':dl_train,'validation':dl_valid}
    print("training size:",df_train.shape)
    print("validation size:",df_valid.shape)
    locf_rmse = math.sqrt(np.mean((ginv(df_valid.loc[df_valid.msk==0,'glc']) - ginv(df_valid.loc[df_valid.msk==0,'glc_dt']))**2))
    print("LOCF RMSE: {:05.4f}".format(locf_rmse))
    log_train.info("LOCF RMSE: {:05.4f}".format(locf_rmse))
    
    # choose hyperparameters
    if HYPERTUNE == 1:
        study_name = datetime.now().strftime("%d_%m_%Y_%H%M")
        study = optuna.create_study(direction="minimize",study_name=study_name)
        study.optimize(objective, n_trials=OPTIM_TRIALS, timeout=60*60*3,catch=(RuntimeWarning,ValueError,))
        trial = study.best_trial

        # model configuration
        hidden_dim = trial.params['hidden_dim']
        dropout_p = trial.params['dropout_p']
        model = NET(FEATURE_DIM, hidden_dim, dropout_p, OUTPUT_DIM,BATCH_SIZE,DEVICE).to(DEVICE)
        lr = trial.params['lr']
        l2_penalty = trial.params['l2_penalty']
        optimizer = trial.params['optimizer']
    else:
        hidden_dim = 10
        dropout_p = 0.1
        # input_dim, hidden_dim, p, output_dim, device
        model = NET(FEATURE_DIM, hidden_dim, dropout_p, OUTPUT_DIM,BATCH_SIZE,DEVICE).to(DEVICE)
        lr = 1e-2
        l2_penalty = 1e-4
        optimizer = "RMSprop"

    # record experiment
    log_train.info("feature_dim:%s" % FEATURE_DIM)
    log_train.info("hidden_dim:%s" % hidden_dim)
    log_train.info("dropout:\n:%s" % dropout_p)
    log_train.info("optimiser:\n:%s" % optimizer)
    log_train.info("lr:\n:%s" % lr)
    log_train.info("l2_penalty:\n:%s" % l2_penalty)
    log_train.info("model:\n:%s" % model)

    # train configuration
    model_optim = getattr(optim, optimizer)(model.parameters(), lr=lr,weight_decay=l2_penalty)
    optim_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min',patience=0,verbose=True)

    # train and evaluate
    best_state, best_val_loss,best_val_err = train_and_evaluate(model, dataloaders, 
                                                    model_optim,optim_scheduler, N_EPOCHS,EARLY_STOPPING,log_train)
    
    # on test
    if ONTEST == True:
        df_test = pd.read_csv('data/test.csv')
        df_test = preproc.transform(df_test)
        d_test = MIMICDataset(df_test,FEATURES,pad=SEQUENCE_LEN)  
        dl_test = DataLoader(d_test,batch_size=BATCH_SIZE)
        loss_test,error_tests,y_preds,y_tests,msks = model.evaluate(dl_test)
        #example_plots(y_preds,y_tests,msks)
        #probabilistic_eval_plots(y_preds,y_tests,msks)
        log_test.info("feature_dim:%s" % FEATURE_DIM)
        log_test.info("hidden_dim:%s" % hidden_dim)
        log_test.info("model:\n:%s" % model)
        log_test.info("loss:\n:%s" % error_tests)
        log_test.info("error:\n:%s" % loss_test)

    # save a model
    model_name = args.net + '.pth'
    state = {
        'state_dict': best_state
    }
    torch.save(state, os.path.join(log_dir, model_name))