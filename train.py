# use lightning framework
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
# data related
from src.data.data_loader import MIMIC3DataModule
from data.feature_sets import all_features_treat_dict,glycaemic_features_treat
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# import models
from src.models.models import *
from src.models.output import *
import torch
# other
import os
# plotting
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting.trajectories import *

# possible models
nets = {'ctRNNModel': ctRNNModel,
        'ctGRUModel': ctGRUModel,
        'ODEGRUBayes':ODEGRUBayes,
        'ctLSTMModel':ctLSTMModel,
        'neuralJumpModel':neuralJumpModel,
        'resNeuralJumpModel':resNeuralJumpModel,
        'IMODE':IMODE,
        'dtRNNModel':dtRNNModel,
        'dtGRUModel':dtGRUModel,
        'dtLSTMModel':dtLSTMModel}

# cmd args
parser = ArgumentParser()
parser.add_argument('--net', dest='net',choices=list(nets.keys()),type=str)
parser.add_argument('--seed', dest='seed',default=42,type=int)
parser.add_argument('--lr', dest='lr',default=0.01,type=float)
parser.add_argument('--test', dest='test',default=False,type=bool)
parser.add_argument('--logfolder', dest='logfolder',default='default',type=str)
parser.add_argument('--update_loss', dest='update_loss',default=0.1,type=float)
parser.add_argument('--loss', dest='loss',default="LL",type=str)
parser.add_argument('--merror', dest='merror',default=0.01,type=float)
parser.add_argument('--nfolds', dest='nfolds',default=1,type=int)
parser.add_argument('--plot', dest='plot',default=True,type=bool)
parser.add_argument('--dt_scaler', dest='dt_scaler',default=1/24,type=float)
parser = BaseModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)

def import_data(path,verbose=True):
    df = pd.read_csv(path)
    ids = df.icustay_id.unique()
    for id_ in ids:
        df_id = df.loc[df.icustay_id == id_,:]
        if (sum(df_id.msk) == df_id.shape[0]):
            df.drop(df.loc[df.icustay_id == id_,:].index,inplace=True)
            if verbose:
                print("excluding:",id_)
    return df

def predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,y_full,t_full,nsteps=10,ginv=lambda x: x,xlabel="Time (hours in ICU)",ylabel="Blood glucose (mg/dL)",title=""):
    preds = model.forward_trajectory(dt_j,(xt_j,x0_j,xi_j),nsteps=nsteps)
    ts_j = time_trajectories(dt_j.squeeze(0),nsteps+1)
    mu_tj,sigma_tj = join_trajectories_gaussian(preds)
    ys_j,t_j = obs_data(xt_j.squeeze(0),y_j,dt_j.squeeze(0))
    plot_trajectory_dist(t_j,ys_j,ts_j,mu_tj,sigma_tj,y_full,t_full,ginv=ginv,xlabel=xlabel,ylabel=ylabel,sim=False,maxtime=12)
    
if __name__ == '__main__':

    # seed
    seed_everything(dict_args['seed'], workers=True)

    # data
    # features
    #input_features = all_features_treat_dict()
    input_features = glycaemic_features_treat()
    if dict_args['net'] in ['neuralJumpModel','resNeuralJumpModel']:
        input_features['intervention'] = input_features['intervention'] + input_features['timevarying'] 
    elif dict_args['net'] in ['IMODE']:
        input_features['timevarying'] = [t for t in input_features['timevarying'] if t in input_features['timevarying'] and t not in input_features['intervention']]
    # dimensions
    input_dims = {'input_dim_t':len(input_features['timevarying']),
                  'input_dim_0':len(input_features['static']),
                  'input_dim_i':len(input_features['intervention'])}
    hidden_dims = {'hidden_dim_t':dict_args['hidden_dim_t'],
                   'hidden_dim_0':dict_args['hidden_dim_0'],
                    'hidden_dim_i':dict_args['hidden_dim_i']}
    print(input_dims)
    print(hidden_dims)
    # import
    df = import_data('data/analysis.csv',verbose=False)
    df.sort_values(by=['icustay_id','timer'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    # splits
    if dict_args['nfolds'] == 1:
        splits = [train_test_split(df.icustay_id.unique(),test_size=0.2)]
    else:
        kf = KFold(n_splits=dict_args['nfolds'])
        splits = kf.split(df.icustay_id.unique())

    for i,(train_ids, test_ids) in enumerate(splits):
        print('fold -----',i)
        if dict_args['nfolds'] == 1:
            df_test = df.loc[df.icustay_id.isin(test_ids)].copy(deep=True)
            df_train = df.loc[df.icustay_id.isin(train_ids)].copy(deep=True)
        else:
            ids_ = df.icustay_id.unique()
            df_test = df.loc[df.icustay_id.isin(ids_[test_ids])].copy(deep=True)
            df_train = df.loc[df.icustay_id.isin(ids_[train_ids])].copy(deep=True)

        mimic3 = MIMIC3DataModule(input_features,df_train,df_test,batch_size=128,testing = False)
        mimic3.setup()
        
        # model
        if dict_args['loss'] == "KL":
            outputNN = GaussianOutputNNKL
        else:
            outputNN = GaussianOutputNNLL
        net = nets[dict_args['net']]
        NN0 = nn.Sequential(
                    nn.Linear(input_dims['input_dim_0'],input_dims['input_dim_0'] // 2),
                    nn.Dropout(0.2),
                    nn.Tanh(),
                    nn.Linear(input_dims['input_dim_0'] // 2,dict_args['hidden_dim_0']),
                    nn.Dropout(0.2),
                    nn.Tanh())
        NN0 = None
        preNN = nn.Sequential(
                    nn.Linear(input_dims['input_dim_t']+dict_args['hidden_dim_0'],(input_dims['input_dim_t']+dict_args['hidden_dim_0']) // 2),
                    nn.Dropout(0.2),
                    nn.Tanh(),
                    nn.Linear((input_dims['input_dim_t']+dict_args['hidden_dim_0']) // 2,dict_args['hidden_dim_t']),
                    nn.Dropout(0.2),
                    nn.Tanh())
        preNN = None
        model = net(input_dims,hidden_dims,outputNN,preNN,NN0,learning_rate=dict_args['lr'],update_loss=dict_args['update_loss'],
                    merror=dict_args["merror"],dt_scaler=dict_args["dt_scaler"])

        # logging
        logger = CSVLogger("experiments/mimic3",name=dict_args['logfolder'])
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)

        # train
        early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=10,min_delta=0.0)  # mostly defaults
        trainer = pl.Trainer.from_argparse_args(args,
                            logger=logger,
                            val_check_interval=1.0,
                            log_every_n_steps=20,
                            gradient_clip_val=20.0,
                            callbacks=[lr_monitor,early_stopping,checkpoint_callback])
        trainer.fit(model, mimic3)

        # test
        if dict_args['test'] == True:
            trainer.test(model,mimic3,ckpt_path="best")
            
        # plot some examples
        if dict_args['plot'] == True:
            if not dict_args['net'] in ['dtRNNModel','dtGRUModel','dtLSTMModel']:
                    dl_test = mimic3.val_dataloader()
                    xt, x0, xi, y, msk, dt, _, id = next(iter(dl_test))
                    ids = [id_.item() for id_ in id]

                    n_examples = 40
                    for i in range(n_examples):
                        with torch.no_grad():
                            model.eval()
                            msk_j = ~msk[i].bool()
                            if sum(msk_j) > 1:
                                dt_j = dt[i][msk_j].unsqueeze(0)
                                xt_j = xt[i][msk_j].unsqueeze(0)
                                x0_j = x0[i].unsqueeze(0)
                                xi_j = xi[i][msk_j].unsqueeze(0)
                                y_j = y[i][msk_j]
                                df_j = df.loc[df.icustay_id == id[i].item(),:]
                                predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,df_j.glc_dt,df_j.timer_dt,nsteps=30,ginv=ginv)
                                plt.savefig(os.path.join(trainer.logger.log_dir,'example_'+str(ids[i])+'.png'),bbox_inches='tight', dpi=150)
                                plt.close()
                            else:
                                next