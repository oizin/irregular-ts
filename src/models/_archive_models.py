import torch 
import torch.nn as nn
import torch.nn.functional as F
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

import math

from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

import math

# Base ----------------------------------------------------------------------------

import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, p, output_dim, device):
        super(Baseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.p = p
        self.device = device
        
    def train_single_epoch(self,dataloader,optim):
        loss = 0.0
        for i, (x, y, msk, dt) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.view(-1,1).to(self.device)
            dt = dt.to(self.device)
            msk = msk.bool().to(self.device)
            optim.zero_grad()
            preds = self.forward(dt,x)
            loss_step = self.loss_fn(preds,y,~msk.squeeze(0))
            loss_step.backward()
            #torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optim.step()
            loss += loss_step.item()
            if i % 1e3 == 0:
                print("BATCH_loss : {:05.3f}".format(loss_step.item()))
        loss /= (i + 1)
        print("EPOCH_loss : {:05.3f}".format(loss))
        
        return loss
        
    def evaluate(self,dataloader,p=0.0,verbose=True):
        rmse, loss = 0., 0.
        N = 0
        y_preds = []
        y_tests = []
        msks = []
        #dts = []
        with tqdm(total=len(dataloader),disable=(not verbose)) as t:
            for i, (x, y, msk, dt) in enumerate(dataloader):
                N += sum((msk == 0).squeeze(0)).item()
                x = x.to(self.device)
                y = y.view(-1,1).to(self.device)
                dt = dt.to(self.device)
                # model prediction
                y_ = self.forward(dt,x,p)
                y_preds.append([yc.detach().cpu().numpy() for yc in y_]) 
                y_tests.append(y.cpu().numpy())
                msk = msk.bool().to(self.device)
                rmse += self.get_sse(y_,y,~msk.squeeze(0)).item()
                loss += self.loss_fn(y_,y,~msk.squeeze(0)).item()
                msks.append(msk.cpu().numpy())
                t.update()
        rmse /= N
        loss /= (i + 1)
        rmse = math.sqrt(rmse)
        print("eval_rmse : {:05.3f}".format(rmse))
        print("eval_loss : {:05.3f}".format(loss))
        return loss,rmse, y_preds, y_tests, msks

    def get_sse(self,y_,y,msk):
        """
        SSE: sum of squared errors
        """
        if type(y_) == tuple:
            y_ = y_[0]
        c = torch.log(torch.tensor(140.0))
        rmse = torch.sum((torch.exp(y_[msk] + c) - torch.exp(y[msk] + c))**2)
        return rmse
    
# ODE-RNN model --------------------------------------------------------------------

class ODEFunc(nn.Module):
    """
    In an ODE-RNN the hidden state h_t of the RNN evolves according to
    an ODE. This ODE is a neural network, i.e. dh/dt = ODEFunc(h,x).
    """
    def __init__(self,hidden_dim):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim+4, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, dt, y):
        return self.net(y)

class ODERNN(Baseline):
    """
    ODE-RNN
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # ODE-RNN
        self.rnn1 = nn.RNNCell(input_dim, hidden_dim)
        self.func = ODEFunc(hidden_dim)
        # N(mu,sigma)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)
        self.l2 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_presigma = nn.Linear(hidden_dim//2, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x, dt, p=0.0):
        
        x = x.squeeze(0)
        dt = dt.squeeze(0)
        T = x.size(0)
        
        mu_out,s_out= torch.zeros(T,1,device = self.device),torch.zeros(T,1,device = self.device) 
        h_t = torch.zeros(1, self.rnn1.hidden_size,device=self.device)
        
        for i in range(0,T):
            if dt[i] == 0:
                h_t = h_t
            else:
                h_t = odeint(self.func,h_t,torch.Tensor([dt[i]]).to(self.device),
                             method='euler',atol=1e-1,rtol=1e-1)[0]
            h_t = F.dropout(h_t,training=True,p=p)
            h_t = self.rnn1(x[i].view(1,self.rnn1.input_size),h_t)
            h_t = F.dropout(h_t,training=True,p=p)
            mu = self.l1(h_t)
            mu = F.dropout(mu,training=True,p=p)
            mu = self.relu(mu)
            mu = self.distribution_mu(mu)
            pre_sigma = self.l2(h_t)
            pre_sigma = F.dropout(pre_sigma,training=True,p=p)
            pre_sigma = self.relu(pre_sigma)
            pre_sigma = self.distribution_presigma(pre_sigma)
            sigma = self.distribution_sigma(pre_sigma) 
            mu_out[i] = mu
            s_out[i] = sigma

        return (mu_out,s_out)
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk])
        return -torch.mean(likelihood)
    
# GRU-Decay model -------------------------------------------------------

class GRU(Baseline):
    """
    GRU
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # GRU
        self.gru = nn.GRU(input_dim+1, hidden_dim, 1, batch_first=True)
        # N(mu,sigma)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)
        self.l2 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_presigma = nn.Linear(hidden_dim//2, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x, dt, p=0.0):
        
        x = torch.cat((x,dt.unsqueeze(2)),2).to(self.device)
        h_t,_ = self.gru(x)
        h_t = F.dropout(h_t,training=True,p=p)
        mu = self.l1(h_t)
        mu = F.dropout(mu,training=True,p=p)
        mu = self.relu(mu)
        mu = self.distribution_mu(mu)
        pre_sigma = self.l2(h_t)
        pre_sigma = F.dropout(pre_sigma,training=True,p=p)
        pre_sigma = self.relu(pre_sigma)
        pre_sigma = self.distribution_presigma(pre_sigma)
        sigma = self.distribution_sigma(pre_sigma) 

        return (mu.squeeze(0),sigma.squeeze(0))
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk])
        return -torch.mean(likelihood)
    
    
# Latent-ODE model -------------------------------------------------------

class LODEFunc(nn.Module):
    """
    dglucose/dt = NN(glucose,insulin)
    """
    def __init__(self,input_dim,hidden_dim):
        super(LODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim+hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim),
            nn.Tanh()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, z, x):
        zx = torch.cat((z,x),1)
        return self.net(zx)
    
class LatentODE(Baseline):

    def __init__(self, input_dim, hidden_dim, p, output_dim, device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        self.device = device
        self.func = LODEFunc(input_dim,hidden_dim).to(device)
        # sigma
        self.ls1 = nn.Linear(hidden_dim, hidden_dim)
        self.as1 = nn.Tanh()
        self.ls2 = nn.Linear(hidden_dim, 1)
        self.os1 = nn.Softplus()
        # mu
        self.lm1 = nn.Linear(hidden_dim, hidden_dim)
        self.am1 = nn.Tanh()
        self.lm2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, dt, x, p=0.0):
        
        if p > 0.0:
            Training = True
        else:
            Training = False
        x = x.squeeze(0)
        dt = dt.squeeze(0)
        T = x.size(0)
        
        mu_out,sigma_out= torch.zeros(T,1,device = self.device),torch.zeros(T,1,device = self.device) 
        z_t = torch.zeros(1,self.hidden_dim,device = self.device)
        for i in range(0,T):
            
            # embedding layer
            # ....
            
            # encode - ODE
            x_i = x[i].unsqueeze(0)
            z_t = self.euler(self.func,z_t.clone(),x_i,dt[i],h=0.1)
            z_t = F.dropout(z_t,p=self.p,training=Training)
            
            # output layer 
            # sigma
            pre_sigma = self.ls1(z_t.clone())
            pre_sigma = self.as1(pre_sigma) 
            #pre_sigma = F.dropout(pre_sigma,p=self.p,training=True)
            pre_sigma = self.ls2(pre_sigma)
            sigma = self.os1(pre_sigma)
            # mu
            pre_mu = self.lm1(z_t.clone())
            pre_mu = self.as1(pre_mu) 
            #pre_mu = F.dropout(pre_mu,self.p,training=True)
            mu = self.ls2(pre_mu)
            # concat
            sigma_out[i] = sigma
            mu_out[i] = mu
        
        return (mu_out,sigma_out)
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk])
        return -torch.mean(likelihood)

    def euler(self,func,y0,x,t,h):
        if (t[1]-t[0]) == 0:
            return y0
        else :
            tsteps = torch.linspace(t[0],t[1],int(2+(t[1]-t[0]) // h))
            hs = torch.diff(tsteps)
            #y = copy.deepcopy(y0)
            y = y0
            for h in hs:
                y += h*func(y,x)
        return y 
