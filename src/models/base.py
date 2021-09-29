import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

class BaseModel(pl.LightningModule):
    """BaseModel
    
    loss function a property of output NN
    
    to do:
        - add in hidden, output dims (entering twice no issue!)
        
    for forward methods see...
    """
    def __init__(self,RNN,OutputNN,hidden_dim,input_dim,learning_rate=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.RNN = RNN
        self.OutputNN = OutputNN
        self.loss_fn = OutputNN.loss_fn
        self.sse_fn = OutputNN.sse_fn
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModel")
        parser.add_argument("--hidden_dim", type=int, default=12)
        return parent_parser
    
    def training_step(self, batch, batch_idx):
        x, y, msk, dt,_ = batch
        msk = msk.bool()
        pred_step = self.forward(dt,x,training = True, p = 0.2)
        loss_step = self.loss_fn(pred_step[~msk],y[~msk])
        self.log("training_loss", loss_step)
        return loss_step
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optim_setting = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=1),
            "monitor": "val_loss"}}
        return optim_setting
    
    def validation_step(self, batch, batch_idx):
        x, y, msk, dt,_ = batch
        msk = msk.bool()
        pred_step = self.forward(dt,x)
        loss_step = self.loss_fn(pred_step[~msk],y[~msk])
        sse_step = self.sse_fn(pred_step[~msk],y[~msk])
        n_step = torch.sum(~msk.bool())
        return {'loss':loss_step,'sse':sse_step,'n':n_step}
    
    def validation_epoch_end(self, outputs):
        loss_epoch = torch.tensor([o['loss'] for o in outputs])
        sse_epoch = torch.tensor([o['sse'] for o in outputs])
        n_epoch = torch.tensor([o['n'] for o in outputs])
        rmse = torch.sqrt(torch.sum(sse_epoch)/torch.sum(n_epoch))
        loss = torch.sum(loss_epoch)/torch.sum(n_epoch)
        self.log("val_loss", loss)
        self.log("val_rmse", rmse)

    def test_step(self, batch, batch_idx):
        x, y, msk, dt,_ = batch
        msk = msk.bool()
        pred_step = self.forward(dt,x)
        loss_step = self.loss_fn(pred_step[~msk],y[~msk])
        sse_step = self.sse_fn(pred_step[~msk],y[~msk])
        n_step = torch.sum(~msk.bool())
        return {'loss':loss_step,'sse':sse_step,'n':n_step,'y':y[~msk].cpu(),'pred':pred_step[~msk].cpu()}

    def test_epoch_end(self, outputs):
        # extract data
        loss_epoch = torch.tensor([o['loss'] for o in outputs])
        sse_epoch = torch.tensor([o['sse'] for o in outputs])
        n_epoch = torch.tensor([o['n'] for o in outputs])
        y_epoch = np.concatenate([o['y'] for o in outputs])
        pred_epoch = np.concatenate([o['pred'] for o in outputs])
        # metrics
        rmse = torch.sqrt(torch.sum(sse_epoch)/torch.sum(n_epoch))
        loss = torch.sum(loss_epoch)/torch.sum(n_epoch)
        prob_eval = self.OutputNN.probabilistic_eval_fn(pred_epoch,y_epoch)
        crps = prob_eval['crps_mean']
        ig =  prob_eval['ig_mean']
        int_score = prob_eval['int_score_mean']
        var_pit = prob_eval['var_pit']
        int_coverage = prob_eval['int_coverage']
        int_av_width = prob_eval['int_av_width']
        int_med_width = prob_eval['int_med_width']
        # log
        self.log("test_loss", loss)
        self.log("test_rmse", rmse)
        self.log("test_var_pit", var_pit)
        self.log("test_crps", crps)
        self.log("test_ignorance", ig)
        self.log("test_int_score", int_score)
        self.log("int_coverage", int_coverage)
        self.log("int_med_width", int_med_width)
        self.log("int_av_width", int_av_width)

#         alpha = 0.05
#         alpha_q = scipy.stats.norm.ppf(1-alpha/2)
#         lower = mu_preds_fold - alpha_q*sigma_preds_fold
#         upper = mu_preds_fold + alpha_q*sigma_preds_fold
#         y_obs_fold = df_test.loc[df_test.msk==0,'glc_dt'].to_numpy()
#         y_obs.append(y_obs_fold)
#         pit_fold = scipy.stats.norm(mu_preds_fold, sigma_preds_fold).cdf(y_obs_fold)
#         var_pit_fold = np.var(pit_fold)
#         crps_fold = ps.crps_gaussian(y_obs_fold, mu=mu_preds_fold, sig=sigma_preds_fold)
#         ig_fold = scipy.stats.norm.logpdf(y_obs_fold,loc=mu_preds_fold, scale=sigma_preds_fold)
#         int_score_fold = (upper - lower) + 2/alpha*(lower - y_obs_fold)*(y_obs_fold < lower) + 2/alpha*(y_obs_fold - upper)*(y_obs_fold > upper)
#         int_coverage_fold = sum((lower < y_obs_fold) & (upper > y_obs_fold))/y_obs_fold.shape[0]
#         int_width_fold = np.mean(ginv(upper) - ginv(lower))
#         int_median_fold = np.median(ginv(upper) - ginv(lower))

class BaseModelCT(BaseModel):
    """BaseModelCT
    
    continuous time
    """
    def __init__(self,RNN,OutputNN,hidden_dim,input_dim,learning_rate=1e-2):
        super().__init__(RNN,OutputNN,hidden_dim,input_dim,learning_rate)

    def forward(self, dt, x, training = False, p = 0.0):
        
        T = x.size(1)
        batch_size = x.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim,device=self.device)
        for i in range(0,T):
            x_i = x[:,i,:]
            dt_i = dt[:,i,:]
            h_t = self.RNN(x_i,h_t,dt_i).squeeze(0)
            h_t = F.dropout(h_t,training=training,p=p)
            output[:,i,:] = self.OutputNN(h_t)
        return output
    
    def forward_trajectory(self, dt, x, nsteps=10):
        T = x.size(1)
        batch_size = x.size(0)
        outputs = []
        h_t = torch.zeros(batch_size, self.hidden_dim,device=self.device)
        for i in range(0,T):
            x_i = x[:,i,:]
            dt_i = dt[:,i,:]
            h_t = self.RNN(x_i,h_t,dt_i,n_intermediate=nsteps).squeeze(0)
            outputs_i = self.OutputNN(h_t)
            outputs.append(outputs_i)
            h_t = h_t[-1]
        return outputs

class BaseModelDT(BaseModel):
    """BaseModelDT
    
    discrete time
    """
    def __init__(self,RNN,OutputNN,hidden_dim,input_dim,learning_rate=1e-2):
        super().__init__(RNN,OutputNN,hidden_dim,input_dim,learning_rate)

    def forward(self, dt, x, training = False, p = 0.0):
        
        T = x.size(1)
        batch_size = x.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim,device=self.device)
        for i in range(0,T):
            x_i = x[:,i,:]
            dt_i = dt[:,i,:]
            h_t = self.RNN(x_i,h_t).squeeze(0)
            h_t = F.dropout(h_t,training=training,p=p)
            output[:,i,:] = self.OutputNN(h_t)
        return output

class BaseModelDecay(BaseModel):
    """BaseModelDecay
    
    dz/dt = linear
    known solution
    
    discrete time
    """
    def __init__(self,RNN,OutputNN,hidden_dim,input_dim,learning_rate=1e-2):
        super().__init__(RNN,OutputNN,hidden_dim,input_dim,learning_rate)

    def forward(self, dt, x, training = False, p = 0.0):
        raise AttributeError('Not implemented')
