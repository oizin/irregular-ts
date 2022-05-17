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
parser.add_argument('--N', dest='N',type=int)
parser.add_argument('--sim_error', dest='sim_error',type=float)
parser.add_argument('--plot', dest='plot',default=True,type=bool)
parser.add_argument('--net', dest='net',choices=list(nets.keys()),default='ctRNNModel',type=str)
#parser.add_argument('--hidden_dim_t', dest='hidden_dim_t',default=10,type=int)
parser.add_argument('--seed', dest='seed',default=42,type=int)
parser.add_argument('--lr', dest='lr',default=0.01,type=float)
parser.add_argument('--test', dest='test',default=False,type=bool)
parser.add_argument('--logfolder', dest='logfolder',default='default',type=str)
parser.add_argument('--update_loss', dest='update_loss',default=0.1,type=float)
parser.add_argument('--data_dir', dest='data_dir',default="data/simulation.csv",type=str)
parser.add_argument('--loss', dest='loss',default="LL",type=str)
parser.add_argument('--merror', dest='merror',default=0.01,type=float)
parser.add_argument('--stationary', dest='stationary',default=1,type=int)
parser.add_argument('--dt_scaler', dest='dt_scaler',default=1.0,type=float)
parser = BaseModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)
        
def predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,y_full,t_full,nsteps=6,ginv=lambda x: x,xlabel="Time (hours in ICU)",ylabel="Blood glucose (mg/dL)",title=""):
    preds = model.forward_trajectory(dt_j,(xt_j,x0_j,xi_j),nsteps=nsteps)
    ts_j = time_trajectories(dt_j.squeeze(0),nsteps+1)
    mu_tj,sigma_tj = join_trajectories_gaussian(preds)
    ys_j,t_j = obs_data(xt_j.squeeze(0),y_j,dt_j.squeeze(0))
    plot_trajectory_dist(t_j*10,ys_j,ts_j*10,mu_tj,sigma_tj,y_full,t_full,ginv=ginv,xlabel=xlabel,ylabel=ylabel,sim=True)
        
if __name__ == '__main__':
    # seed
    seed_everything(42, workers=True)
    
    # data
    # features
    if dict_args['net'] in ['neuralJumpModel','resNeuralJumpModel']:
        features = {'timevarying':['x','m','g'],'intervention':['x','m','g']}
    elif dict_args['net'] in ['IMODE']:
        features = {'timevarying':['x'],'intervention':['m','g']}
    else:
        features = {'timevarying':['x','m','g'],'intervention':['m','g']}
    # dimensions
    input_dims = {'input_dim_t':len(features['timevarying']),
                  'input_dim_0':0,
                  'input_dim_i':len(features['intervention'])}
    sim = SimulationsDataModule(features,dict_args['data_dir'])
    sim.setup()
    
    # model
    if dict_args['loss'] == "KL":
        outputNN = GaussianOutputNNKL
    else:
        outputNN = GaussianOutputNNLL
    hidden_dims = {'hidden_dim_t':dict_args['hidden_dim_t'],'hidden_dim_0':0,'hidden_dim_x':12,'hidden_dim_i':3}
    print(input_dims)
    print(hidden_dims)
    net = nets[dict_args['net']]
    model = net(input_dims,hidden_dims,outputNN,learning_rate=dict_args['lr'],
                update_loss=dict_args['update_loss'],merror=dict_args["merror"],dt_scaler=dict_args["dt_scaler"])

    # logging
    logger = CSVLogger("experiments/simulations",name=dict_args['logfolder'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)

    # train
    early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=10,min_delta=0.0)  # mostly defaults
    trainer = pl.Trainer.from_argparse_args(args,
                        logger=logger,
                        val_check_interval=1.0,
                        log_every_n_steps=10,
                        callbacks=[lr_monitor,early_stopping,checkpoint_callback])
    trainer.fit(model, sim)

    # test
    test_res = trainer.test(model,sim,ckpt_path="best")
    
    # save simulation settings
    df_settings = pd.DataFrame({'N':[dict_args['N']],'loss':[dict_args['loss']],'sim_error':[dict_args['sim_error']],'stationary':[dict_args['stationary']]})
    df_settings.to_csv(os.path.join(trainer.logger.log_dir,'settings.csv'))

    # plot some examples
    if dict_args['plot'] == True:
        if not dict_args['net'] in ['dtRNNModel','dtGRUModel','dtLSTMModel']:
            df = pd.read_csv(dict_args['data_dir'])
            dl_test = sim.val_dataloader()
            xt, x0, xi, y, msk, dt, _, id = next(iter(dl_test))
            ids = [id_.item() for id_ in id]

            n_examples = 10
            for i in range(n_examples):
                with torch.no_grad():
                    model.eval()
                    msk_j = ~msk[i].bool()
                    dt_j = dt[i][msk_j].unsqueeze(0)
                    xt_j = xt[i][msk_j].unsqueeze(0)
                    x0_j = x0[i][msk_j].unsqueeze(0)
                    xi_j = xi[i][msk_j].unsqueeze(0)
                    y_j = y[i][msk_j]
                    df_j = df.loc[df.id == id[i].item(),:]
                    predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,xi_j,y_j,df_j.x_true,df_j.t,nsteps=20,ginv=ginv)
                    plt.savefig(os.path.join(trainer.logger.log_dir,'example_'+str(i)+'.png'),bbox_inches='tight', dpi=150)