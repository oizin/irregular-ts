# use lightning framework
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
# data related
from src.data.data_loader import SimulationsDataModule
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# import models
from src.models.models import *
from src.models.output import *
import torch
from src.models.catboost import CatboostModel,catboost_feature_engineer
from catboost import Pool
# other
import os
import json
# plotting
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting.trajectories import *
# evaluation
import src.metrics.metrics as metrics

# possible models
models = {'ODEGRUModel': ODEGRUModel}

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
parser.add_argument('--merror', dest='merror',default=0.01,type=float)
parser.add_argument('--niter', dest='niter',default=10000,type=int)

parser = BaseModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

def ginv(x):
    x = x.copy()
    x = np.exp(x + np.log(140))
    return x

def g(x):
    x = x.copy()
    x = np.log(x) - np.log(140)
    return x

def predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,y_full,t_full,nsteps=6,ginv=lambda x: x,xlabel="Time (hours in ICU)",ylabel="Blood glucose (mg/dL)",title=""):
    preds = model.forward_trajectory(dt_j,(xt_j,x0_j,xi_j),nsteps=nsteps)
    ts_j = time_trajectories(dt_j.squeeze(0),nsteps+1)
    mu_tj,sigma_tj = join_trajectories_gaussian(preds)
    ys_j,t_j = obs_data(xt_j.squeeze(0),y_j,dt_j.squeeze(0))
    plot_trajectory_dist(t_j*10,ys_j,ts_j*10,mu_tj,sigma_tj,y_full,t_full,ginv=ginv,xlabel=xlabel,ylabel=ylabel,sim=True)

## deep learning trainer ##
def train_test_deeplearner(df_train,df_test,features,task='gaussian',test=True):
    """"
    Function for training and testing a deep learning model given training and test data
    """

    # setup the data
    sim = SimulationsDataModule(df_train,df_test,features)
    sim.setup()
    train_dataloader = sim.train_dataloader()
    val_dataloader = sim.val_dataloader()
    test_dataloader = sim.test_dataloader()
    
    # match model output layers with task:
    if task == 'gaussian':
        outputNN = GaussianOutputNNKL
        eval_fn = metrics.gaussian_eval_fn
    elif task == 'conditional_expectation':
        outputNN = ConditionalExpectNN
        eval_fn = metrics.conditional_eval_fn

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

logger = CSVLogger("experiments/simulations",name=args.logfolder)

if __name__ == '__main__':
    # seed
    seed_everything(42, workers=True)
    
    # data
    # features
    with open('data/feature_sets.json', 'r') as f:
        feature_sets = json.load(f)
    features = feature_sets['simulation']['features']
    print(features)

    # read in and prepare simulated data
    df = pd.read_csv(args.data)
    # filter to "observed data"
    df = df.loc[df.obs == 1,:].copy()
    df.sort_values(by=['id','t'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    df.drop(columns=["obs"],inplace=True)
    # reshape timestamp -> t0,t1
    df['glucose_t_obs_next'] = df.groupby('id')['glucose_t_obs'].shift()
    df['t0'] = df.groupby('id')['t'].shift()
    df.rename(columns={'t':'t1'},inplace=True)
    # drop first
    df = df.loc[~df.glucose_t_obs_next.isnull(),:]
    # msk where y is NaN
    df['msk'] =  df.glucose_t_obs_next.isnull()
    df['msk0'] =  df.glucose_t_obs.isnull()
    if args.small_data:
        df = df.iloc[0:8000,]
    # dimensions
    dims = {'input_dim_t':len(features['timevarying']),
             'input_dim_0':len(features['static']),
             'input_dim_i':len(features['intervention']),
             'hidden_dim_t':args.hidden_dim_t,
             'hidden_dim_0':None,
             'hidden_dim_i':4,
             'input_size_update':len(features['timevarying'])+len(features['static'])}
    # input_dims = {'input_dim_t':len(features['timevarying']),
    #               'input_dim_0':0,
    #               'input_dim_i':len(features['intervention'])}
    # sim = SimulationsDataModule(features,dict_args['data_dir'])
    # sim.setup()
    
    # splits
    if args.nfolds == 1:
        splits = [train_test_split(df.id.unique(),test_size=0.2)]
    else:
        kf = KFold(n_splits=args.nfolds)
        splits = kf.split(df.id.unique())

    for i,(train_ids, test_ids) in enumerate(splits):
        print('fold:',i)
        if args.nfolds == 1:
            df_test = df.loc[df.id.isin(test_ids)].copy()
            df_train = df.loc[df.id.isin(train_ids)].copy()
        else:
            ids_ = df.id.unique()
            df_test = df.loc[df.id.isin(ids_[test_ids])].copy()
            df_train = df.loc[df.id.isin(ids_[train_ids])].copy()

        if args.model in ['CatboostModel']:
            train_test_catboost(df_train,df_test,features,args.task,args.test,args.niter)
        else:
            train_test_deeplearner(df_train,df_test,features,args.task,args.test)

    
    # # logging
    # logger = CSVLogger("experiments/simulations",name=dict_args['logfolder'])
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)

    # # save simulation settings
    # df_settings = pd.DataFrame({'N':[dict_args['N']],'loss':[dict_args['loss']],'sim_error':[dict_args['sim_error']],'stationary':[dict_args['stationary']]})
    # df_settings.to_csv(os.path.join(trainer.logger.log_dir,'settings.csv'))

    # # plot some examples
    # if dict_args['plot'] == True:
    #     if not dict_args['net'] in ['dtRNNModel','dtGRUModel','dtLSTMModel']:
    #         df = pd.read_csv(dict_args['data_dir'])
    #         dl_test = sim.val_dataloader()
    #         xt, x0, xi, y, msk, dt, _, id = next(iter(dl_test))
    #         ids = [id_.item() for id_ in id]

    #         n_examples = 10
    #         for i in range(n_examples):
    #             with torch.no_grad():
    #                 model.eval()
    #                 msk_j = ~msk[i].bool()
    #                 dt_j = dt[i][msk_j].unsqueeze(0)
    #                 xt_j = xt[i][msk_j].unsqueeze(0)
    #                 x0_j = x0[i][msk_j].unsqueeze(0)
    #                 xi_j = xi[i][msk_j].unsqueeze(0)
    #                 y_j = y[i][msk_j]
    #                 df_j = df.loc[df.id == id[i].item(),:]
    #                 predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,df_j.x_true,df_j.t,nsteps=20,ginv=ginv)
    #                 plt.savefig(os.path.join(trainer.logger.log_dir,'example_'+str(i)+'.png'),bbox_inches='tight', dpi=150)