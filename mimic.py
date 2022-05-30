# use lightning framework
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
# data related
from src.data.data_loader import MIMICDataModule
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# import models
from src.models.models import *
from src.models.output import *
from src.models.other.catboost import CatboostModel,catboost_feature_engineer
import torch
from catboost import Pool
# other
import os
# plotting
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting.trajectories import *

# possible models
models = {#'ctRNNModel': ctRNNModel, 
        'CatboostModel': CatboostModel
        ,'ODEGRUModel': ODEGRUModel
        ,'FlowGRUModel': FlowGRUModel
        #'ODEGRUBayes':ODEGRUBayes, 
        #'ODELSTMModel':ODELSTMModel,
        #'neuralJumpModel':neuralJumpModel, 
        #'resNeuralJumpModel':resNeuralJumpModel, 
        #'IMODE':IMODE,
        #'dtRNNModel':RNNModel, 
        ,'GRUModel':GRUModel
        #,'LSTMModel':LSTMModel
        }

# cmd args
parser = ArgumentParser()
parser.add_argument('--model', dest='model',choices=list(models.keys()),type=str)
parser.add_argument('--seed', dest='seed',default=42,type=int)
parser.add_argument('--lr', dest='lr',default=0.01,type=float)
parser.add_argument('--test', dest='test',default=False,type=bool)
parser.add_argument('--logfolder', dest='logfolder',default='default',type=str)
parser.add_argument('--update_loss', dest='update_loss',default=0.001,type=float)
parser.add_argument('--loss', dest='loss',default="LL",type=str)
parser.add_argument('--merror', dest='merror',default=0.01,type=float)
parser.add_argument('--nfolds', dest='nfolds',default=1,type=int)
parser.add_argument('--plot', dest='plot',default=True,type=bool)
parser = BaseModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)


if dict_args['model'] in []:
    deeplearner = True
elif dict_args['model'] in []:
    catboost = True


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

def train_test_deeplearner():

    mimic = MIMICDataModule(features,df_train,df_test,batch_size=128,testing = False)
    print('setting up data...')
    mimic.setup()
    
    # model
    if dict_args['loss'] == "KL":
        outputNN = GaussianOutputNNKL
    else:
        outputNN = GaussianOutputNNLL
    model = models[dict_args['model']]
    model = model(dims,
                outputNN,
                ginv,
                learning_rate=dict_args['lr'],
                update_loss=dict_args['update_loss'],
                merror=dict_args["merror"])

    # logging
    logger = CSVLogger("experiments/mimic",name=dict_args['logfolder'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)

    # train
    early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=10,min_delta=0.0)  # mostly defaults
    trainer = pl.Trainer.from_argparse_args(args,
                        logger=logger,
                        val_check_interval=0.5,
                        log_every_n_steps=20,
                        gradient_clip_val=2.0,
                        callbacks=[lr_monitor,early_stopping,checkpoint_callback])
    trainer.fit(model, mimic)

    # test
    if dict_args['test'] == True:
        trainer.test(model,mimic,ckpt_path="best")

def train_test_catboost(df_train,df_test):
    # train data
    df_train,cat_vars = catboost_feature_engineer(df_train,features)
    input_features = cat_vars + features['timevarying'] + features['static'] + features['counts'] + features['time_vars']
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
    # train 
    model = CatboostModel()
    model.fit(train_pool,eval_set=valid_pool)
    # evaluate
    preds = model.predict(test_pool)
    preds[:,1] = np.sqrt(preds[:,1])
    print(preds)
    print(model.eval_fn(preds,y_test,ginv))

def import_feature_sets():
    pass

if __name__ == '__main__':

    # seed
    seed_everything(dict_args['seed'], workers=True)

    # data
    # features
    #input_features = all_features_treat_dict()
    with open('data/feature_sets.json', 'r') as f:
        feature_sets = json.load(f)
    features = feature_sets['test_features']
    if dict_args['model'] in ['neuralJumpModel','resNeuralJumpModel']:
        features['intervention'] = features['intervention'] + features['timevarying'] 
    elif dict_args['model'] in ['IMODE']:
        features['timevarying'] = [t for t in features['timevarying'] if t in features['timevarying'] and t not in features['intervention']]
    # dimensions
    dims = {'input_dim_t':len(features['timevarying']),
             'input_dim_0':len(features['static']),
             'input_dim_i':len(features['intervention']),
             'hidden_dim_t':8,
             'hidden_dim_0':None,
             'hidden_dim_i':4,
             'input_size_update':len(features['timevarying'])+len(features['static'])}
    print(dims)
    # import
    df = pd.read_csv('data/mimic.csv')
    df.sort_values(by=['stay_id','timer'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    # splits
    if dict_args['nfolds'] == 1:
        splits = [train_test_split(df.stay_id.unique(),test_size=0.2)]
    else:
        kf = KFold(n_splits=dict_args['nfolds'])
        splits = kf.split(df.stay_id.unique())

    for i,(train_ids, test_ids) in enumerate(splits):
        print('fold:',i)
        if dict_args['nfolds'] == 1:
            df_test = df.loc[df.stay_id.isin(test_ids)].copy()
            df_train = df.loc[df.stay_id.isin(train_ids)].copy()
        else:
            ids_ = df.stay_id.unique()
            df_test = df.loc[df.stay_id.isin(ids_[test_ids])].copy()
            df_train = df.loc[df.stay_id.isin(ids_[train_ids])].copy()

        if dict_args['model'] in ['CatboostModel']:
            train_test_catboost(df_train,df_test)
        else:
            train_test_deeplearner()
        # mimic = MIMICDataModule(features,df_train,df_test,batch_size=128,testing = True)
        # print('setting up data...')
        # mimic.setup()
        
        # # model
        # if dict_args['loss'] == "KL":
        #     outputNN = GaussianOutputNNKL
        # else:
        #     outputNN = GaussianOutputNNLL
        # model = models[dict_args['model']]
        # # NN0 = nn.Identity()
        # # preNN = nn.Identity()
        # model = model(dims,
        #             outputNN,
        #             ginv,
        #             learning_rate=dict_args['lr'],
        #             update_loss=dict_args['update_loss'],
        #             merror=dict_args["merror"])

        # # logging
        # logger = CSVLogger("experiments/mimic",name=dict_args['logfolder'])
        # lr_monitor = LearningRateMonitor(logging_interval='step')
        # checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)

        # # train
        # early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=10,min_delta=0.0)  # mostly defaults
        # trainer = pl.Trainer.from_argparse_args(args,
        #                     logger=logger,
        #                     val_check_interval=1.0,
        #                     log_every_n_steps=20,
        #                     gradient_clip_val=2.0,
        #                     callbacks=[lr_monitor,early_stopping,checkpoint_callback])
        # trainer.fit(model, mimic)

        # # test
        # if dict_args['test'] == True:
        #     trainer.test(model,mimic,ckpt_path="best")
            
        # # plot some examples
        # if dict_args['plot'] == True:
        #     if not dict_args['model'] in ['dtRNNModel','dtGRUModel','dtLSTMModel']:
        #             dl_test = mimic.val_dataloader()
        #             xt, x0, xi, y, msk, dt, _, id = next(iter(dl_test))
        #             ids = [id_.item() for id_ in id]

        #             n_examples = 40
        #             for i in range(n_examples):
        #                 with torch.no_grad():
        #                     model.eval()
        #                     msk_j = ~msk[i].bool()
        #                     if sum(msk_j) > 1:
        #                         dt_j = dt[i][msk_j].unsqueeze(0)
        #                         xt_j = xt[i][msk_j].unsqueeze(0)
        #                         x0_j = x0[i].unsqueeze(0)
        #                         xi_j = xi[i][msk_j].unsqueeze(0)
        #                         y_j = y[i][msk_j]
        #                         df_j = df.loc[df.stay_id == id[i].item(),:]
        #                         predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,df_j.glc_dt,df_j.timer_dt,nsteps=30,ginv=ginv)
        #                         plt.savefig(os.path.join(trainer.logger.log_dir,'example_'+str(ids[i])+'.png'),bbox_inches='tight', dpi=150)
        #                         plt.close()
        #                     else:
        #                         next