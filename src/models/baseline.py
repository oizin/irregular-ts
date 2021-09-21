# base classes
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import numpy as np

class NeuralODE(nn.Module):
    """
    Neural ODE layer
        Maps from hidden_dim -> hidden_dim
        Requires neural network (ODENet) as input that takes input (t,x)
        Integrate from t0 to t1
    """
    def __init__(self, hidden_dim, ODENet):
        super(NeuralODE, self).__init__()
        self.ODENet = ODENet(hidden_dim)

    def forward(self, x, ts):
        """
        Returns xs
        """
        x_ts = odeint(self.ODENet, x, ts,rtol=1e-2,atol=1e-3)
        return x_ts
    
    
class f(nn.Module):
    """
    dy/dt = f()
    """
    def __init__(self,input_ode_size,hidden_size):
        super(f, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_ode_size+hidden_size, 50),
            nn.Tanh(),
            nn.Linear(50, hidden_size),
        )
        
    def forward(self,t,hidden,x):
        result = hidden
        output = self.net(result)*self.time_gaps
        return output
    
class ODENetBase(nn.Module):
    def __init__(self,ODENet,input_ode_size,hidden_size):
        super(f, self).__init__()
        
        self.net = ODENet(input_ode_size,hidden_size)
                
    def forward(self,t,x):
        result = hidden
        if use_t:
            result = torch.cat((result,t*self.time_gaps),1)
        if input_ode_size > 0:
            result = torch.cat((result,self.input_ode),1)
        output = self.net(result)*self.time_gaps
        return output

    
class ODERNNBase(nn.Module):
    """
    Base class for continuous time recurrent neural network (RNN) (e.g. vanilla RNNs, Jump NNs, GRUs and LSTMs)
    Args:
        output_size: dimension of output
        input_update_size: dimension of update features (can be larger than output)
        input_ode_size: dimension of features you wish to pass to ODENet
        hidden_size: dimension of hidden state
        ODENet: nn.Module
        UpdateNN: nn.Module
        use_t: will ODENet use time, default=False
    """
    def __init__(self,
                 updateNN,
                 ODENet,
                 output_size, 
                 input_update_size, 
                 input_ode_size, 
                 hidden_size, 
                 use_t=False):
        super(ODERNNBase,self).__init__()
        
        self.ODENet = ODENetBase(ODENet,input_ode_size,hidden_size)
        self.updateNN = updateNN(input_update_size,hidden_size)
        
        # update ODENet forward to account for 
        # 1. time non-alignment across batches
        # 2. passing of exogenous/control variables
        
    def forward(self,input_update,h_0,times,input_ode=None):   
        """ 
        forward
        """
        # discrete update/jump as new information receieved
        hidden = self.forward_update(input_update,h_0)
        # use ODENet to 'evolve' state to next timestep
        output = self.forward_ode(self.ODENet,hidden,times,input_ode)[1]
        return output
    
    def forward_update(input_update,h_0):
        """
        forward_update
        """
        output = self.updateNN(input_update,h_0)
        return output

    def forward_ode(hidden,times,input_ode=None):
        """
        forward_ode
        -----> use for predicting a trajectory
        """
        output = solve_ode(self.ODENet,input_ode,hidden,times)[1]
        return output
    
    def solve_ode(self,input_ode,h_0,times):
        """
        solve_ode
        """
        # enable input and time_gaps to be passed to ODENet.forward
        self.input_ode = input_ode
        self.time_gaps = times[:,i,1] - times[:,i,0]
        # numerical integration until next time step
        output = odeint(self.ODENet, h_0, torch.tensor([0,1.0]).to(self.device))
        return output    

    
    
class Baseline(nn.Module):
    """
    Wrapper class for training the networks used in this project
    """
    def __init__(self):
        super(Baseline, self).__init__()
                
    def train_single_epoch(self,dataloader,optim):
        """
        Method for model training
        """
        loss = 0.0
        n_batches = len(dataloader)
        for i, (t, x, y) in enumerate(dataloader):
            optim.zero_grad()
            preds = self.forward(t,x)
            loss_step = self.loss_fn(preds,y)
            loss_step.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
            optim.step()
            loss += loss_step.item()
        loss /= (i + 1)
        
        return loss
    
class Model(Baseline):

    def __init__(self, input_dim, hidden_dim, output_dim, odefunc):
        Baseline.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.neural_ode = NeuralODE(2,odefunc) ###! the 2

    def forward(self, t, x):
            
        T = x.size(1)
        outputs = torch.zeros(x.size(0),T,1)

        # ODE
        for i in range(0,T):
            t0 = t[0,i,0]  # fix
            t1 = t[0,i,1]
            y0 = x[:,i:(i+1),:]
            outputs[:,i:(i+1),:] = self.neural_ode(y0,torch.tensor([t0, t1]))[1]

        return outputs


    def forward_trajectory(self, t, x, nsteps=10):
        T = x.size(1)
        outputs = []
        # ODE
        for i in range(0,T):
            t0 = t[0,i,0]
            t1 = t[0,i,1]
            ts = torch.linspace(t0,t1,nsteps)
            y0 = x[:,i:(i+1),:]
            y_ = self.neural_ode(y0,ts)[0:nsteps].reshape(nsteps).detach()
            outputs_i = torch.cat((ts.reshape(-1,1),y_.reshape(-1,1)),1)
            outputs.append(outputs_i)

        return outputs

    def loss_fn(self,y_,y):
        return torch.sum((y_ - y)**2)

