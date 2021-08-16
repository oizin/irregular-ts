import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import math
from tqdm import tqdm
import numpy as np

DT_SCALER = 1 / 24
SEQUENCE_LENGTH = 100
LAMBDA = 1.0

COVAR_INDEX = [0] + list(range(7,25))
INSULIN_INDEX = list(range(1,6))

def nReLU(input):
    return torch.min(input,0)

class nReLU(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, input):
        return nReLU(input) 

class Baseline(nn.Module):
    """
    Base class for the models
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, device):
        super(Baseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.p = p
        self.device = device
        
    def train_single_epoch(self,dataloader,optim,epoch=0):
        """
        Method for model training
        """
        loss = 0.0
        n_batches = len(dataloader)
        print("number of batchs: {}".format(n_batches))
        for i, (x, y, msk, dt,msk0) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            y0 = x[:,:,0:1].to(self.device)
            dt = dt.to(self.device)
            msk = msk.bool().to(self.device)
            msk0 = msk0.bool().to(self.device)
            optim.zero_grad()
            preds,preds0 = self.forward(dt,x,epoch=epoch,training=True)
            pred_loss_step = self.loss_fn(preds,y,~msk.view(x.shape[0],-1))
            pred0_loss_step = self.loss_fn(preds0,preds,~msk0.view(x.shape[0],-1))
            loss_step = pred_loss_step + LAMBDA*pred0_loss_step
            loss_step.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optim.step()
            loss += loss_step.item()
            if i % int(n_batches/4) == 0:
                print("Batch number: {}".format(i))
                print("BATCH_loss : {:05.3f}".format(loss_step.item()))
        loss /= (i + 1)
        print("EPOCH_loss : {:05.3f}".format(loss))
        
        return loss
        
    def evaluate(self,dataloader,p=0.0):
        """
        Method for model evaluation
        """
        rmse, loss = 0., 0.
        N = 0
        y_preds = []
        y_tests = []
        msks = []
        #dts = []
        with tqdm(total=len(dataloader)) as t:
            for i, (x, y, msk, dt, _) in enumerate(dataloader):
                N += sum(sum(msk == 0)).item()
                x = x.to(self.device)
                y = y.to(self.device)
                dt = dt.to(self.device)
                # model prediction
                y_ = self.forward(dt,x)
                y_preds.append([yc.detach().cpu().numpy() for yc in y_]) 
                y_tests.append(y.cpu().numpy())
                msk = msk.bool().to(self.device)
                rmse += self.get_sse(y_,y,~msk.view(x.shape[0],-1)).item()
                loss += self.loss_fn(y_,y,~msk.view(x.shape[0],-1)).item()
                msks.append(msk.cpu().numpy())
                t.update()
        rmse /= N
        loss /= N
        rmse = math.sqrt(rmse)
        print("_rmse : {:05.3f}".format(rmse))
        print("_loss : {:05.3f}".format(loss))
        return loss,rmse, y_preds, y_tests, msks

    def get_sse(self,y_,y,msk):
        """
        Method for calculation of the sum of squared errors
        """
        if type(y_) == tuple:
            y_ = y_[0]
        y_ = y_
        c = torch.log(torch.tensor(140.0))
        rmse = torch.sum((torch.exp(y_[msk] + c) - torch.exp(y[msk] + c))**2)
        return rmse
    
    def predict(self,dataloader):
        """
        Predictions that drop masked and concatenate
        """
        mu_preds = []
        sigma_preds = []
        for i, (x, y, msk, dt, _) in enumerate(dataloader):
            x = x.to(self.device)
            dt = dt.to(self.device)
            # model prediction
            mu_,sig_ = self.forward(dt,x)
            msk = msk.bool().to(self.device)
            mu_preds.append((mu_.squeeze(2)[~msk.bool()]).detach().cpu().numpy())
            sigma_preds.append((sig_.squeeze(2)[~msk.bool()]).detach().cpu().numpy())
        mu_preds = np.concatenate(mu_preds)
        sigma_preds = np.concatenate(sigma_preds)
        return mu_preds, sigma_preds

#-------------------------------------------------------------------------------------------------

class cODERNN_(nn.Module):
    """
    In an ODE-RNN the hidden state h_t of the RNN evolves according to
    an ODE. This ODE is a neural network, i.e. dh/dt = ODEFunc(h,x).
    """
    def __init__(self,input_dim,hidden_dim,batch_size,device):
        super(cODERNN_, self).__init__()

        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim),
            #nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, dt, y):
        return self.net(y)*(self.dt*DT_SCALER)
    
    def solve_ode(self, z0, t, x):
        self.x = x  # overwrites
        self.dt = t
        #outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),method='euler',options=dict(step_size=0.1))[1]
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),rtol=1e-3,atol=1e-3)[1]
        return outputs

class cODERNN(Baseline):
    """
    ODE-RNN
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # ODE-RNN
        self.rnn1 = nn.RNNCell(19, hidden_dim)
        nn.init.constant_(self.rnn1.bias_hh, val=0)
        nn.init.constant_(self.rnn1.bias_hh, val=0)
        nn.init.normal_(self.rnn1.weight_hh, mean=0, std=0.1)
        nn.init.normal_(self.rnn1.weight_ih, mean=0, std=0.1)
        
        self.rnn2 = nn.RNNCell(5+hidden_dim, hidden_dim)
        nn.init.constant_(self.rnn2.bias_hh, val=0)
        nn.init.constant_(self.rnn2.bias_hh, val=0)
        nn.init.normal_(self.rnn2.weight_hh, mean=0, std=0.1)
        nn.init.normal_(self.rnn2.weight_ih, mean=0, std=0.1)

        self.func = cODERNN_(input_dim,hidden_dim,batch_size,device)
        # N(mu,sigma)
        # mu
        self.output_net_covar = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        self.output_net_insulin = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        ).to(device)


    def forward(self, dt, x, p=0.0,epoch=0,training=False):
        
        T = x.size(1)
        batch_size = x.size(0)
        
        x_insulin = x[:,:,INSULIN_INDEX]
        x_covariate = x[:,:,COVAR_INDEX]
        
        # tempporary solution - put elsewhere...
        tt = x_insulin > 0
        tt = torch.sum(tt,axis=2)
        tt = tt > 0
        tt = tt.long()
        treatment = tt.unsqueeze(2)

        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        mu0_out = torch.zeros(batch_size,T,1,device = self.device)
        h_covariate_t = torch.zeros(batch_size, self.rnn1.hidden_size,device=self.device)
        h_insulin_t = torch.zeros(batch_size, self.rnn2.hidden_size,device=self.device)

        for i in range(0,T):
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
            treatment_i = treatment[:,i,:]

            x_covariate_i = x_covariate[:,i:(i+1),:]
            x_insulin_i = x_insulin[:,i:(i+1),:]
            
            h_covariate_t = h_covariate_t + self.rnn1(x_covariate_i.squeeze(1),h_covariate_t)     
            h_insulin_t = h_insulin_t + self.rnn2(torch.cat((x_insulin_i.squeeze(1),h_covariate_t),1),h_insulin_t)     

            if training==True:
                # immediate forward pass for prediction ('filtering') at t0
                mu0_tmp = self.output_net_covar(h_covariate_t) - treatment_i * self.output_net_insulin(h_insulin_t)
                mu0_out[:,i:(i+1),:] = mu0_tmp.unsqueeze(1)
            
            # forward pass for prediction at t0+
            h_covariate_t = self.func.solve_ode(h_covariate_t,dt_i,x_covariate_i)

            h_insulin_t = self.func.solve_ode(h_insulin_t,dt_i,x_insulin_i)
            mu_tmp = self.output_net_covar(h_covariate_t) - treatment_i * self.output_net_insulin(h_insulin_t)
            mu_out[:,i:(i+1),:] = mu_tmp.unsqueeze(1)
            
        if training == True:
            return mu_out.squeeze(2),mu0_out.squeeze(2)
        else:
            return mu_out.squeeze(2)
    
    def loss_fn(self,y_,y,msk):
        assert y_.shape == y.shape
        assert y_.shape == msk.shape
        return torch.mean((y_[msk] - y[msk])**2)
