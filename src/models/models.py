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
            preds = self.forward(dt,x,epoch=epoch,training=True)
            pred_loss_step = self.loss_fn(preds,y,~msk.view(x.shape[0],-1))
            #pred0_loss_step = self.loss0_fn(preds0,preds,~msk0.view(x.shape[0],-1))
            loss_step = pred_loss_step #+ LAMBDA*pred0_loss_step
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
        y_ = y_.squeeze(2)
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

class NeuralODE_(nn.Module):
    """
    dglucose/dt = NN(glucose,insulin)
    """
    def __init__(self,input_dim,batch_size,device):
        super(NeuralODE_, self).__init__()
        
        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        xz = torch.cat((z,self.x),2)
        return self.net(xz)*(self.dt*DT_SCALER) # -> scale by timestep
    
    def solve_ode(self, x0, t, x):
        self.x = x  # overwrites
        self.dt = t
        #outputs = odeint(self, x0, torch.tensor([0,1.0]).to(self.device),method='euler',options=dict(step_size=0.1))[1]
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),rtol=1e-2,atol=1e-3)[1]

        return outputs
    
class NeuralODE(Baseline):

    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        self.device = device
        self.func = NeuralODE_(input_dim,batch_size,device).to(device)
        
    def forward(self, dt, x, p=0.0):
        
        #x = x.squeeze(0)
        #dt = dt.squeeze(0)
        T = x.size(1)
        
        # ODE
        mu_out = torch.zeros(x.size(0),T,1,device = self.device)
        for i in range(0,T):
            y0 = x[:,i:(i+1),0:1]
            x_i = x[:,i:(i+1),1:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1).unsqueeze(1)
            mu_out[:,i:(i+1),:] = self.func.solve_ode(y0,dt_i,x_i)

        return mu_out
    
#     def loss_fn(self,mu_s_,y,msk):
#         y_, s_ = mu_s_
#         distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
#         likelihood = distribution.log_prob(y[msk])
#         return -torch.mean(likelihood)

    def loss_fn(self,y_,y,msk):
        return torch.sum((y_[msk] - y[msk])**2)

#-------------------------------------------------------------------------------------------------


class LatentODE1_(nn.Module):
    """
    dglucose/dt = NN(glucose,insulin)
    """
    def __init__(self,hidden_dim,input_dim,batch_size,device):
        super(LatentODE1_, self).__init__()
        
        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim+hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        xz = torch.cat((z,self.x),2)
        return self.net(xz)*(self.dt*DT_SCALER) # -> scale by timestep
    
    def solve_ode(self, z0, t, x, method,**kwargs):
        self.x = x  # overwrites
        self.dt = t
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),method=method,options=kwargs)[1]
        return outputs
    
class LatentODE1(Baseline):

    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        self.device = device
        self.func = LatentODE1_(hidden_dim,input_dim,batch_size,device).to(device)
        self.mu_net = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        self.sigma_net = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        ).to(device)
#         self.encode_z0 = nn.Sequential(
#             #nn.BatchNorm1d(input_dim),
#             nn.Linear(input_dim, 50),
#             nn.Tanh(),
#             nn.Linear(50, hidden_dim),
#             nn.Tanh(),
#         ).to(device)
        self.jumpNN = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim),
            nn.Tanh(),
        ).to(device)
        
#         for m in self.encode_z0.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.constant_(m.bias, val=0)
        
    def forward(self, dt, x, p=0.0,epoch=0):
        
        #x = x.squeeze(0)
        batch_size = x.size(0)
        T = x.size(1)
        
        # ODE
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        sigma_out = torch.zeros(batch_size,T,1,device = self.device)
        #z0 = 0.01 * torch.randn(batch_size,1,self.hidden_dim,device = self.device)
        #z0 = self.encode_z0(x[:,0,:]).unsqueeze(1)
        #z0 = torch.zeros(batch_size,1,self.hidden_dim,device = self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1).unsqueeze(1)
            z0 = self.jumpNN(x_i)
            z0 = self.func.solve_ode(z0,dt_i,x_i,'euler',step_size=0.1)
            mu_out[:,i:(i+1),:] = self.mu_net(z0.squeeze(1)).unsqueeze(1)
            sigma_out[:,i:(i+1),:] = self.sigma_net(z0.squeeze(1)).unsqueeze(1)
            
        return mu_out,sigma_out
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk].unsqueeze(1))
        return -torch.sum(likelihood)


#     def loss_fn(self,y_,y,msk):
#         return torch.sum((y_[msk] - y[msk])**2)
        
#-------------------------------------------------------------------------------------------------

class ODERNN_(nn.Module):
    """
    In an ODE-RNN the hidden state h_t of the RNN evolves according to
    an ODE. This ODE is a neural network, i.e. dh/dt = ODEFunc(h,x).
    """
    def __init__(self,input_dim,hidden_dim,batch_size,device):
        super(ODERNN_, self).__init__()

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

class ODERNN(Baseline):
    """
    ODE-RNN
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # ODE-RNN
        self.rnn = nn.RNNCell(input_dim, hidden_dim)
        nn.init.constant_(self.rnn.bias_hh, val=0)
        nn.init.constant_(self.rnn.bias_hh, val=0)
        nn.init.normal_(self.rnn.weight_hh, mean=0, std=0.1)
        nn.init.normal_(self.rnn.weight_ih, mean=0, std=0.1)
        self.func = ODERNN_(input_dim,hidden_dim,batch_size,device)
#         self.rnn = nn.RNNCell(1, hidden_dim)
#         self.func = ODERNN_(input_dim-1,hidden_dim,batch_size,device)
        # N(mu,sigma)
        # mu
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)
        self.relu = nn.ReLU()
        self.sigma_net = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        ).to(device)


    def forward(self, dt, x, p=0.0,epoch=0,training=False):
        
        T = x.size(1)

        batch_size = x.size(0)
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        mu0_out = torch.zeros(batch_size,T,1,device = self.device)
        sigma_out = torch.zeros(batch_size,T,1,device = self.device)
        h_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
#             h_t = self.rnn(x_i.squeeze(1),h_t)
#             x0_i = x[:,i:(i+1),0:1]
#             x1_i = x[:,i:(i+1),1:]
#             dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
            h_t = h_t.clone() + self.rnn(x_i.squeeze(1),h_t.clone())
            if training==True:
                mu0 = self.l1(h_t)
                mu0 = F.dropout(mu0,training=training,p=p)
                mu0 = self.relu(mu0)
                mu0_out[:,i:(i+1),:] = self.distribution_mu(mu0).unsqueeze(1)
            h_t = self.func.solve_ode(h_t,dt_i,x_i)
            mu = self.l1(h_t)
            mu = F.dropout(mu,training=training,p=p)
            mu = self.relu(mu)
            mu_out[:,i:(i+1),:] = self.distribution_mu(mu).unsqueeze(1)
            sigma_out[:,i:(i+1),:] = self.sigma_net(h_t.squeeze(1)).unsqueeze(1)
        if training == True:
            return (mu_out,sigma_out),mu0_out
        else:
            return (mu_out,sigma_out)
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk].unsqueeze(1))
        return -torch.sum(likelihood)
    
    def loss0_fn(self,mu0,y0,msk):
#         print(y0[0].shape)
#         print(mu0.shape)
#         print(msk.shape)
        distribution = torch.distributions.normal.Normal(mu0.squeeze(2)[msk], 1.0)
        likelihood = distribution.log_prob(y0[0].squeeze(2)[msk])
#         print(distribution)
        return -torch.sum(likelihood)#torch.mean((mu0.squeeze(2)[msk])**2)#

#     def loss_fn(self,y_,y,msk):
#         return torch.mean((y_[msk] - y[msk])**2)


#-------------------------------------------------------------------------------------------------

class ODEGRU_(nn.Module):
    """
    In an ODE-GRU the hidden state h_t of the GRU evolves according to
    an ODE. This ODE is a neural network, i.e. dh/dt = ODEFunc(h,x).
    """
    def __init__(self,input_dim,hidden_dim,batch_size,device):
        super(ODEGRU_, self).__init__()

        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim),
            nn.Tanh(),
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
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),method='euler',options=dict(step_size=0.1))[1]
        return outputs

class ODEGRU(Baseline):
    """
    ODE-GRU
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # ODE-GRU
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.func = ODEGRU_(input_dim,hidden_dim,batch_size,device)
        # N(mu,sigma)
        # mu
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)
        self.relu = nn.ReLU()
        self.sigma_net = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        ).to(device)


    def forward(self, dt, x, p=0.0,epoch=0,training=False):
        
        T = x.size(1)

        batch_size = x.size(0)
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        sigma_out = torch.zeros(batch_size,T,1,device = self.device)
        h_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
            h_t = self.rnn(x_i.squeeze(1),h_t)
            h_t = self.func.solve_ode(h_t,dt_i,x_i)
            mu = self.l1(h_t)
            mu = F.dropout(mu,training=True,p=p)
            mu = self.relu(mu)
            mu_out[:,i:(i+1),:] = self.distribution_mu(mu).unsqueeze(1)
            sigma_out[:,i:(i+1),:] = self.sigma_net(h_t.squeeze(1)).unsqueeze(1)
        return (mu_out,sigma_out)
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk].unsqueeze(1))
        return -torch.sum(likelihood)

#     def loss_fn(self,y_,y,msk):
#         return torch.mean((y_[msk] - y[msk])**2)

#-------------------------------------------------------------------------------------------------

class ODELSTM_(nn.Module):
    """
    In an ODE-LSTM the hidden state h_t of the LSTM evolves according to
    an ODE. This ODE is a neural network, i.e. dh/dt = ODEFunc(h,x).
    """
    def __init__(self,input_dim,hidden_dim,batch_size,device):
        super(ODELSTM_, self).__init__()

        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_dim),
            nn.Tanh(),
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
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),method='euler',options=dict(step_size=0.1))[1]
        return outputs

class ODELSTM(Baseline):
    """
    ODE-LSTM
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # ODE-LSTM
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        #self.cfunc = ODELSTM_(input_dim,hidden_dim,batch_size,device)
        self.hfunc = ODELSTM_(input_dim,hidden_dim,batch_size,device)
        # N(mu,sigma)
        # mu
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)
        self.relu = nn.ReLU()
        self.sigma_net = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        ).to(device)


    def forward(self, dt, x, p=0.0,epoch=0,training=False):
        
        T = x.size(1)

        batch_size = x.size(0)
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        sigma_out = torch.zeros(batch_size,T,1,device = self.device)
        h_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        c_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
            h_t,c_t = self.rnn(x_i.squeeze(1),(h_t,c_t))
            #c_t = self.cfunc.solve_ode(c_t,dt_i,x_i)
            h_t = self.hfunc.solve_ode(h_t,dt_i,x_i)
            mu = self.l1(h_t)
            mu = F.dropout(mu,training=True,p=p)
            mu = self.relu(mu)
            mu_out[:,i:(i+1),:] = self.distribution_mu(mu).unsqueeze(1)
            sigma_out[:,i:(i+1),:] = self.sigma_net(h_t.squeeze(1)).unsqueeze(1)
        return (mu_out,sigma_out)
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk].unsqueeze(1))
        return -torch.sum(likelihood)

#     def loss_fn(self,y_,y,msk):
#         return torch.mean((y_[msk] - y[msk])**2)

class LSTM(Baseline):
    """
    LSTM
    """
    def __init__(self, input_dim, hidden_dim, p, output_dim, batch_size,device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        # LSTM
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        # N(mu,sigma)
        # mu
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)
        self.relu = nn.ReLU()
        self.sigma_net = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        ).to(device)


    def forward(self, dt, x, p=0.0,epoch=0,training=False):
        
        T = x.size(1)

        batch_size = x.size(0)
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        sigma_out = torch.zeros(batch_size,T,1,device = self.device)
        h_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        c_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
            h_t,c_t = self.rnn(x_i.squeeze(1),(h_t,c_t))
            mu = self.l1(h_t)
            mu = F.dropout(mu,training=True,p=p)
            mu = self.relu(mu)
            mu_out[:,i:(i+1),:] = self.distribution_mu(mu).unsqueeze(1)
            sigma_out[:,i:(i+1),:] = self.sigma_net(h_t.squeeze(1)).unsqueeze(1)
        return (mu_out,sigma_out)
    
    def loss_fn(self,mu_s_,y,msk):
        y_, s_ = mu_s_
        distribution = torch.distributions.normal.Normal(y_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk].unsqueeze(1))
        return -torch.sum(likelihood)

#-------------------------------------------------------------------------------------------------

class LatentODE2_(nn.Module):
    """
    (dglucose/dt,dh/dt) = (NN(glucose,h),NN(h,x))
    """
    def __init__(self,hidden_dim,input_dim,batch_size,device):
        super(LatentODE2_, self).__init__()

        self.x = torch.zeros(batch_size,SEQUENCE_LENGTH,input_dim).to(device)
        self.dt = torch.zeros(batch_size,SEQUENCE_LENGTH,1).to(device)
        self.device = device
        # dh/dt
        self.dh = nn.Sequential(
            nn.Linear(hidden_dim+input_dim, min((hidden_dim+input_dim)*2,50)),
            nn.Tanh(),
            nn.Linear(min((hidden_dim+input_dim)*2,50), hidden_dim),
            nn.Tanh(),
        ).to(device)
        # dy/dt
#         self.dy_rnn = nn.RNNCell(input_dim, hidden_dim)
#         self.dy_lin = nn.Linear(hidden_dim, 1)
#         self.dy_tanh = nn.Tanh()
        self.dy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Tanh(),
            nn.Linear(hidden_dim*2, 1)
        ).to(device)

        for m in self.dh.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        for m in self.dy.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
#         nn.init.normal_(self.dy_rnn.weight_ih,mean=0, std=0.1)
#         nn.init.normal_(self.dy_rnn.weight_hh,mean=0, std=0.1)
#         nn.init.constant_(self.dy_rnn.bias_ih, val=0)
#         nn.init.constant_(self.dy_rnn.bias_ih, val=0)
#         nn.init.normal_(self.dy_lin.weight,mean=0, std=0.1)
#         nn.init.constant_(self.dy_lin.bias, val=0)

    def forward(self, t, z):
        # dh/dt
        #dh = self.dh(z[:,:,1:])*self.dt
        dh = self.dh(torch.cat((self.x,z[:,:,1:]),2)) * (self.dt * DT_SCALER)
        # dy/dt
        #dy = self.dy_rnn(self.x.squeeze(1),z[:,:,1:].squeeze(1)) # hidden should be shape (1,batch,hidden)
        #dy = self.dy_lin(dy.unsqueeze(1))
        dy = self.dy(z[:,:,1:]) * (self.dt * DT_SCALER)
        return torch.cat((dy,dh),2)

    def solve_ode(self, z0, t, x):
        self.x = x  # overwrites
        self.dt = t
        outputs = odeint(self, z0, torch.tensor([0,1.0]).to(self.device),method='euler',options=dict(step_size=0.1))[1]
        return outputs    
    
class LatentODE2(Baseline):

    def __init__(self, input_dim, hidden_dim, p, output_dim,batch_size, device):
        Baseline.__init__(self,input_dim, hidden_dim, p, output_dim, device)
        self.device = device
        self.batch_size = batch_size
        self.func = LatentODE2_(hidden_dim,input_dim,batch_size,device).to(device)
        
    def forward(self, dt, x):
        
        #x = x.squeeze(0)
        batch_size = x.size(0)
        T = x.size(1)

        # ODE stuff
        mu_out = torch.zeros(batch_size,T,1,device = self.device)
        z0 = torch.zeros(batch_size,1,self.hidden_dim+1,device = self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            y0 = x[:,i:(i+1),0:1]
            z0[:,:,0:1] = y0
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1).unsqueeze(1)
            z0 = self.func.solve_ode(z0,dt_i,x_i)
            mu_out[:,i:(i+1),:] = z0[:,:,0:1]

        return mu_out
    
    def loss_fn(self,y_,y,msk):
        return torch.sum((y_[msk] - y[msk].unsqueeze(1))**2)