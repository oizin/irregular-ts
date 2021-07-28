import torch
import torch.nn as nn
import copy
import math
from tqdm import tqdm


def train_and_evaluate(model, dataloaders, optim, scheduler,loss_fn,n_epochs,device,early_stopping_rounds,logging):
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
            loss = train_single_epoch(model, dl_train, optim,loss_fn,device,logging)
            loss_val,error_val,y_preds,s_preds,y_tests, tsteps = evaluate(model, dl_val, loss_fn, 
                                                                          'validation',device,logging)
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
            t.set_postfix(loss_and_val_err='{:05.3f} and {:05.3f}'.format(
                loss_val, error_val))
            print('\n')
            t.update()
            if early_stop > early_stopping_rounds:
                logging.info("Stopping early")
                break
    return best_state, best_val_loss,best_val_err,y_preds,s_preds, y_tests, tsteps