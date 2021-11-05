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
from data.feature_sets import all_features_dict
# import models
from src.models.base import BaseModel
from src.models.base import BaseModel,BaseModelCT,BaseModelDT,BaseModelDecay
from src.models.output import *
from src.models.odenet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchctrnn
# other
import os
# plotting
import pandas as pd
import matplotlib.pyplot as plt
from src.plotting.trajectories import *

class FF1(nn.Module):
    """FF1: basic feedforward network (1)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim*2,50)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim*2,50), hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self,input,t,hidden):
        output = self.net(hidden)
        return output

class FF2(nn.Module):
    """FF2: basic feedforward network (2)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, max(feature_dim*2,50)),
            nn.Tanh(),
            nn.Linear(max(feature_dim*2,50), max(feature_dim*2,50)),
            nn.Tanh(),
            nn.Linear(max(feature_dim*2,50), hidden_dim),
            nn.Tanh(),
        )
                
    def forward(self,input,hidden):
        output = self.net(input)
        return output

class Model(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,learning_rate=0.1,update_loss=0.1):
        input_dim_t = input_dims['input_dim_t']
        input_dim_0 = input_dims['input_dim_0']
        hidden_dim_t = hidden_dims['hidden_dim_t']
        hidden_dim_0 = hidden_dims['hidden_dim_0']
        preNN = None
        NN0 = None
        odenet = FF1(hidden_dims['hidden_dim_t'],hidden_dims['hidden_dim_t'])
        jumpnn = FF2(hidden_dims['hidden_dim_t'],input_dims['input_dim_t'])
        odernn = torchctrnn.LatentJumpODECell(jumpnn,odenet,input_dim_t,tol={'atol':1e-5,'rtol':1e-5},method='dopri5')
        outputNN = ConditionalExpectNN(hidden_dims['hidden_dim_t'],ginv=lambda x: x * 4.0)
        super().__init__(odernn,outputNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror=None)
        self.save_hyperparameters({'net':'jumpModel'})
        
def predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,y_j,y_full,t_full,nsteps=6,ginv=lambda x: x):
    preds = model.forward_trajectory(dt_j,(xt_j,x0_j),nsteps=nsteps)
    ts_j = time_trajectories(dt_j.squeeze(0),nsteps+1)
    mu_tj = join_trajectories_point(preds)
    ys_j,t_j = obs_data(xt_j.squeeze(0),y_j,dt_j.squeeze(0))
    plot_trajectory_point(t_j,ys_j,ts_j,mu_tj,y_full,t_full,ginv=ginv)
        
if __name__ == '__main__':
    # seed
    seed_everything(42, workers=True)
    
    # data
    sim = SimulationsDataModule()
    sim.setup()
    # model
    input_dims = {'input_dim_t':1,'input_dim_0':0}
    hidden_dims = {'hidden_dim_t':10,'hidden_dim_0':0}
    print(input_dims)
    print(hidden_dims)
    model = Model(input_dims,hidden_dims,learning_rate=1e-2,update_loss=1.0)
    print(model)

    # logging
    logger = CSVLogger("experiments/simulations",name='tmp')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=1)

    # train
    early_stopping = EarlyStopping(monitor="val_loss",mode="min",verbose=True,patience=5,min_delta=0.0)  # mostly defaults
    trainer = pl.Trainer(max_epochs=100,
                        logger=logger,
                         #gpus=1,
                        val_check_interval=1.0,
                        log_every_n_steps=10,
                        callbacks=[lr_monitor,early_stopping,checkpoint_callback])
    trainer.fit(model, sim)

    # test
    test_res = trainer.test(model,sim,ckpt_path="best")

    # plot some examples
    df = pd.read_csv("./data/simulation_OU_1.csv")
    dl_test = sim.val_dataloader()
    xt, x0, y, msk, dt, _, id = next(iter(dl_test))
    ids = [id_.item() for id_ in id]
    
    def ginv(x):
        return 4*x

    n_examples = 10
    for i in range(n_examples):
        msk_j = ~msk[i].bool()
        dt_j = dt[i][msk_j].unsqueeze(0)
        xt_j = xt[i][msk_j].unsqueeze(0)
        x0_j = x0[i][msk_j].unsqueeze(0)
        y_j = y[i][msk_j]
        df_j = df.loc[df.id == id[i].item(),:]
        predict_and_plot_trajectory(model,dt_j,xt_j,x0_j,y_j,df_j.value,df_j.timestamp/10.0,nsteps=30,ginv=ginv)
        plt.savefig(os.path.join(trainer.logger.log_dir,'example_'+str(i)+'.png'),bbox_inches='tight', dpi=150)

    # how many ODE Net evals?
#     model.RNN.ODENet.track_calls = True
#     model.RNN.ODENet.calls = 0
#     print(dt_j.shape)
#     model.forward(dt_j,(xt_j,x0_j))
#     print(model.RNN.ODENet.calls)