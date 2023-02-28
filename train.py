## libraries ##
import time
# use lightning framework
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
# data related
import pandas as pd
from src.data.data_loader import MIMICDataModule
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# import models
from src.models.models import *
from src.models.output import *
from src.models.catboost import CatboostModel,catboost_feature_engineer
from src.models.linearmodel import LinearModel
import torch
from catboost import Pool
# evaluation
import src.metrics.metrics as metrics
# other
import os
# plotting
import matplotlib.pyplot as plt
from src.plotting.trajectories import *

## possible models ##
models = {'LinearModel': LinearModel
        ,'CatboostModel': CatboostModel
        ,'ODEGRUModel': ODEGRUModel
        ,'FlowGRUModel': FlowGRUModel
        ,'DecayGRUModel':DecayGRUModel
        ,'ODELSTMModel':ODELSTMModel
        ,'FlowLSTMModel':FlowLSTMModel
        ,'IMODE':IMODE
        ,'GRUModel':GRUModel
        ,'LSTMModel':LSTMModel
        }

## cmd args ##
parser = ArgumentParser()
# general housingkeeping;
parser.add_argument('--seed', dest='seed',default=76,type=int)
parser.add_argument('--logfolder', dest='logfolder',default='default',type=str)
parser.add_argument('--nfolds', dest='nfolds',default=1,type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--no-test', dest='test', action='store_false')
parser.set_defaults(test=True)
parser.add_argument('--features', dest='features',default='features',type=str)
parser.add_argument('--small_data', action='store_true')
parser.add_argument('--no-small_data', dest='small_data', action='store_false')
parser.set_defaults(store_true=False)

# which experiment are we running:
parser.add_argument('--data', dest='data',default='mimic',type=str)
parser.add_argument('--task',dest='task',
                choices=['conditional_expectation','gaussian','categorical'],
                default='gaussian',
                type=str)

# which model to use:
parser.add_argument('--model', dest='model',choices=list(models.keys()),type=str)
parser.add_argument('--lr', dest='lr',default=0.01,type=float)
parser.add_argument('--update_mixing', dest='update_mixing',default=0.001,type=float)
parser.add_argument('--merror', dest='merror',default=1e-3,type=float)
parser.add_argument('--niter', dest='niter',default=10000,type=int)

#parser.add_argument('--loss', dest='loss',default="LL",type=str)
parser.add_argument('--plot', dest='plot',default=False,type=bool)
parser = BaseModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

def predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,y_full,t_full,nsteps=10,ginv=lambda x: x,xlabel="Time (hours in ICU)",ylabel="Blood glucose (mg/dL)",title=""):
    preds = model.forward_trajectory(dt_j,(xt_j,x0_j,xi_j),nsteps=nsteps)
    ts_j = time_trajectories(dt_j.squeeze(0),nsteps+1)
    mu_tj,sigma_tj = join_trajectories_gaussian(preds)
    ys_j,t_j = obs_data(xt_j.squeeze(0),y_j,dt_j.squeeze(0))
    plot_trajectory_dist(t_j,ys_j,ts_j,mu_tj,sigma_tj,y_full,t_full,ginv=ginv,xlabel=xlabel,ylabel=ylabel,sim=False,maxtime=12)

def ginv(x):
    x = x.copy()
    x = np.exp(x + np.log(140))
    return x

def g(x):
    x = x.copy()
    x = np.log(x) - np.log(140)
    return x

## logger ##
logger = CSVLogger("experiments/mimic",name=args.logfolder)

## deep learning trainer ##
def train_test_deeplearner(df_train,df_test,features,task='gaussian',test=True):
    """"
    Function for training and testing a deep learning model given training and test data
    """

    # setup the data
    mimic = MIMICDataModule(features,df_train,df_test,batch_size=128,testing = False)
    print('setting up data...')
    mimic.setup()
    train_dataloader = mimic.train_dataloader()
    val_dataloader = mimic.val_dataloader()
    test_dataloader = mimic.test_dataloader()
    
    # match model output layers with task:
    if task == 'gaussian':
        outputNN = GaussianOutputNNKL
        eval_fn = metrics.gaussian_eval_fn
    elif task == 'conditional_expectation':
        outputNN = ConditionalExpectNN
        eval_fn = metrics.conditional_eval_fn
    elif task == 'categorical':
        outputNN = BinnedOutputNN
        eval_fn = metrics.categorical_eval_fn

    # setup the model
    model = models[args.model]
    model = model(dims,
                outputNN,
                ginv,
                eval_fn,
                learning_rate=args.lr,
                update_mixing=args.update_mixing,
                merror=args.merror)

    # training monitors
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)
    early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=20,min_delta=0.0)
    
    # pytorch lightning trainer
    trainer = pl.Trainer.from_argparse_args(args,
                        logger=logger,
                        val_check_interval=0.5,
                        log_every_n_steps=20,
                        auto_lr_find=True,
                        gradient_clip_val=1.0,
                        callbacks=[lr_monitor,early_stopping,checkpoint_callback])
    trainer.fit(model, train_dataloader,val_dataloader)

    # test model
    if test:
        print("testing...")
        trainer.test(model,test_dataloader,ckpt_path="best")
        print("predicting...")
        predictions = trainer.predict(model,test_dataloader,ckpt_path="best")
        predictions = torch.cat(predictions,dim=0).numpy()
        if task == "gaussian": 
            df_predictions = pd.DataFrame(predictions,columns=['rn','mu','sigma'])
        elif task == "conditional_expectation":
            df_predictions = pd.DataFrame(predictions,columns=['rn','mu'])
        df_predictions['rn'] = df_predictions.rn.astype(int)
        df_predictions['model'] = args.model
        df_predictions.to_csv(os.path.join(trainer.logger.log_dir,'predictions_' + str(i) + '.csv'),index=False)

    # plot some examples
    # if args.plot == True:
    #     if args.model in ['ODEGRUModel']:
    #             model.RNN.NeuralODE.backend = 'torchdiffeq'
    #             dl_test = mimic.val_dataloader()
    #             xt, x0, xi, y, msk, dt, _, id = next(iter(dl_test))
    #             #ids = [id_.item() for id_ in id]

    #             n_examples = 40
    #             for j in range(n_examples):
    #                 with torch.no_grad():
    #                     model.eval()
    #                     msk_j = ~msk[j].bool()
    #                     if sum(msk_j) > 1:
    #                         dt_j = dt[j][msk_j].unsqueeze(0)
    #                         xt_j = xt[j][msk_j].unsqueeze(0)
    #                         x0_j = x0[j].unsqueeze(0)
    #                         xi_j = xi[j][msk_j].unsqueeze(0)
    #                         y_j = y[j][msk_j]
    #                         df_j = df.loc[df.stay_id == id[j].item(),:]
    #                         predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,df_j.glc_dt,df_j.timer_dt,nsteps=30,ginv=ginv)
    #                         plt.savefig(os.path.join(trainer.logger.log_dir,'example_'+str(j)+'.png'),bbox_inches='tight', dpi=150)
    #                         plt.close()
    #                     else:
    #                         next

## catboost trainer ##
def train_test_catboost(df_train,df_test,features,task='gaussian',test=True,niter=10000):
    ## setup data ##
    # train data
    df_train,cat_vars = catboost_feature_engineer(df_train,features)
    input_features = cat_vars + features['timevarying'] + features['static'] + features['counts'] + features['time_vars']
    print(input_features)
    # train-validation split
    train_ids, valid_ids = train_test_split(df_train[features['id']].unique(),test_size=0.1)
    df_valid = df_train.loc[df_train[features['id']].isin(valid_ids)].copy()
    df_train = df_train.loc[df_train[features['id']].isin(train_ids)].copy()
    X_train = df_train.loc[df_train.msk == 0,input_features].to_numpy()
    X_valid = df_valid.loc[df_valid.msk == 0,input_features].to_numpy()
    y_train = g(df_train.loc[df_train.msk == 0,features['target']].to_numpy())
    y_valid = g(df_valid.loc[df_valid.msk == 0,features['target']].to_numpy())
    train_pool = Pool(X_train,y_train)
    valid_pool = Pool(X_valid,y_valid)
    # test data
    df_test,_ = catboost_feature_engineer(df_test,features)
    X_test = df_test.loc[df_test.msk == 0,input_features].to_numpy()
    y_test = g(df_test.loc[df_test.msk == 0,features['target']].to_numpy())
    test_pool = Pool(X_test) 
    ## setup model ##
    if task == 'gaussian':
        eval_fn = metrics.gaussian_eval_fn
    elif task == 'conditional_expectation':
        eval_fn = metrics.conditional_eval_fn
    elif task == 'categorical':
        eval_fn = metrics.categorical_eval_fn
    model = CatboostModel(niter,task,eval_fn)
    ## train model ##
    model.fit(train_pool,eval_set=valid_pool)
    ## test model ##
    if test:
        preds = model.predict(test_pool)
        if task == "gaussian": 
            preds[:,1] = np.sqrt(preds[:,1])
        eval_catboost = model.eval_fn(preds,y_test,ginv)
        print(eval_catboost)
        logger.log_metrics(eval_catboost)
        logger.save()
        # save predictions
        df_predictions = df_test.loc[df_test.msk == 0,['rn']]
        if task == "gaussian": 
            df_predictions.loc[:,'mu'] = preds[:,0]
            df_predictions.loc[:,'sigma'] = preds[:,1]
        elif task == "conditional_expectation":
            df_predictions.loc[:,'mu'] = preds
        df_predictions['model'] = 'Catboost'
        df_predictions.to_csv(os.path.join(logger.log_dir,'predictions_' + str(i) + '.csv'),index=False)

## linear trainer ##
def train_test_linear_model(df_train,df_test,features,task='gaussian',test=True):
    ## setup data ##
    # train data
    input_features = features['timevarying'] + features['static'] + features['counts'] + features['time_vars']
    print(input_features)
    # train-validation split
    X_train = df_train.loc[df_train.msk == 0,input_features].to_numpy()
    y_train = g(df_train.loc[df_train.msk == 0,features['target']].to_numpy())
    # test data
    X_test = df_test.loc[df_test.msk == 0,input_features].to_numpy()
    y_test = g(df_test.loc[df_test.msk == 0,features['target']].to_numpy())
    ## setup model ##
    if task == 'gaussian':
        eval_fn = metrics.gaussian_eval_fn
    elif task == 'conditional_expectation':
        eval_fn = metrics.conditional_eval_fn
    elif task == 'categorical':
        eval_fn = metrics.categorical_eval_fn
    model = LinearModel(task,eval_fn)
    ## train model ##
    model.fit(X_train,y_train)
    ## test model ##
    if test:
        preds = model.predict(X_test)
        print(preds)
        eval_linear = model.eval_fn(preds,y_test,ginv)
        print(eval_linear)
        logger.log_metrics(eval_linear)
        logger.save()
        # save predictions
        df_predictions = df_test.loc[df_test.msk == 0,['rn']]
        if task == "gaussian": 
            df_predictions.loc[:,'mu'] = preds[:,0]
            df_predictions.loc[:,'sigma'] = preds[:,1]
        elif task == "conditional_expectation":
            df_predictions.loc[:,'mu'] = preds
        df_predictions['model'] = 'LinearModel'
        df_predictions.to_csv(os.path.join(logger.log_dir,'predictions_' + str(i) + '.csv'),index=False)

def import_feature_sets():
    pass

## run program ##
if __name__ == '__main__':

    ## print some information ##
    print('Training model {} for task {} on dataset {}.'.format(args.model,args.task,args.data))
    print('Train/eval will use {} fold validation (where 1 indicates a single train/test split)'.format(args.nfolds))
    print('The full dataset is being used: {}'.format(~args.small_data))

    ## seed ##
    seed_everything(args.seed, workers=True)

    ## features ##
    with open('data/feature_sets.json', 'r') as f:
        feature_sets = json.load(f)
    features = feature_sets[args.data.split('_')[0]][args.features]
    if args.model in ['IMODE']:
        features['timevarying'] = [t for t in features['timevarying'] if t in features['timevarying'] and t not in features['intervention']]
    
    ## dimensions - for NNs ##
    dims = {'input_dim_t':len(features['timevarying']) + len(features['counts']),
             'input_dim_0':len(features['static']),
             'input_dim_i':len(features['intervention']),
             'hidden_dim_t':args.hidden_dim_t,
             'hidden_dim_0':None,
             'hidden_dim_i':4,
             'input_size_update':len(features['timevarying'])+len(features['static'])+len(features['counts'])}
    print(dims)

    ## data import ##
    df = pd.read_csv('data/'+ args.data +'.csv')
    df.sort_values(by=[features['id'],features['time_vars'][0]],inplace=True)
    if args.data.split('_')[0] == 'simulation':
        # only use "observed data" in the simulation
        df = df.loc[(df.obs == True) & ~(df.glucose_t_obs_next.isnull()),:] 
        print(df)
    df.reset_index(drop=True,inplace=True)
    if args.small_data:
        df = df.iloc[0:8000,]
    
    ## split data into train/test ##
    if args.nfolds == 1:
        splits = [train_test_split(df[features['id']].unique(),test_size=0.2)]
    else:
        kf = KFold(n_splits=args.nfolds)
        splits = kf.split(df[features['id']].unique())

    ## train/test loop ##
    for i,(train_ids, test_ids) in enumerate(splits):
        print('fold:',i)
        if args.nfolds == 1:
            df_test = df.loc[df[features['id']].isin(test_ids)].copy()
            df_train = df.loc[df[features['id']].isin(train_ids)].copy()
        else:
            ids_ = df[features['id']].unique()
            df_test = df.loc[df[features['id']].isin(ids_[test_ids])].copy()
            df_train = df.loc[df[features['id']].isin(ids_[train_ids])].copy()
        print('Data size: train {}; test {}'.format(df_train.shape[0],df_test.shape[0]))

        ## run model train/test ##
        start_time = time.time()
        if args.model in ['CatboostModel']:
            train_test_catboost(df_train,df_test,features,args.task,args.test,args.niter)
        elif args.model in ['LinearModel']:
            train_test_linear_model(df_train,df_test,features,args.task,args.test)
        else:
            train_test_deeplearner(df_train,df_test,features,args.task,args.test)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')

            
