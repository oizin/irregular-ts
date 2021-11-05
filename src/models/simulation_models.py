from .base import BaseModel,BaseModelCT,BaseModelDT,BaseModelDecay
from .output import *
from .odenet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchctrnn

ginv = lambda x: x * 4.0

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
            nn.Linear(max(feature_dim*2,50), hidden_dim),
            nn.Tanh(),
        )
                
    def forward(self,input,hidden):
        output = self.net(input)
        return output
    
class neuralJumpModel(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,learning_rate=0.1,update_loss=0.1):
        input_dim_t = input_dims['input_dim_t']
        input_dim_0 = input_dims['input_dim_0']
        hidden_dim_t = hidden_dims['hidden_dim_t']
        hidden_dim_0 = hidden_dims['hidden_dim_0']
        preNN = None
        NN0 = None
        odenet = FF1(hidden_dim_t,hidden_dim_t)
        jumpnn = FF2(hidden_dim_t,input_dim_t)
        odernn = torchctrnn.LatentJumpODECell(jumpnn,odenet,input_dim_t,tol={'atol':1e-5,'rtol':1e-5},method='dopri5')
        outputNN = ConditionalExpectNN(hidden_dim_t,ginv=ginv)
        super().__init__(odernn,outputNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror=None)
        self.save_hyperparameters({'net':'neuralJumpModel'})  
        
class ctRNNModel(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,learning_rate=0.1,update_loss=1.0):
        input_dim_t = input_dims['input_dim_t']
        input_dim_0 = input_dims['input_dim_0']
        hidden_dim_t = hidden_dims['hidden_dim_t']
        hidden_dim_0 = hidden_dims['hidden_dim_0']
        preNN = None
        NN0 = None
        odenet = FF1(hidden_dim_t,hidden_dim_t)
        odernn = torchctrnn.ODERNNCell(odenet,input_dim_t,tol={'atol':1e-5,'rtol':1e-5},method='dopri5')
        outputNN = ConditionalExpectNN(hidden_dim_t,ginv=ginv)
        super().__init__(odernn,outputNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss)
        self.save_hyperparameters({'net':'ctRNNModel'})
        
class ctGRUModel(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,learning_rate=0.1,update_loss=1.0):
        input_dim_t = input_dims['input_dim_t']
        input_dim_0 = input_dims['input_dim_0']
        hidden_dim_t = hidden_dims['hidden_dim_t']
        hidden_dim_0 = hidden_dims['hidden_dim_0']
        preNN = None
        NN0 = None
        odenet = FF1(hidden_dim_t,hidden_dim_t)
        odernn = torchctrnn.ODEGRUCell(odenet,input_dim_t,tol={'atol':1e-5,'rtol':1e-5},method='dopri5')
        outputNN = ConditionalExpectNN(hidden_dim_t,ginv=ginv)
        super().__init__(odernn,outputNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss)
        self.save_hyperparameters({'net':'ctGRUModel'})

class ctLSTMModel(BaseModelCT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1,update_loss=1.0):
        input_dim_t = input_dims['input_dim_t']
        input_dim_0 = input_dims['input_dim_0']
        hidden_dim_t = hidden_dims['hidden_dim_t']
        hidden_dim_0 = hidden_dims['hidden_dim_0']
        preNN = None
        NN0 = None
        odenet = FF1(hidden_dim,input_dim)
        odernn = torchctrnn.ODELSTMCell(odenet,hidden_dim,tol={'atol':1e-5,'rtol':1e-5},method='dopri5')
        outputNN = ConditionalExpectNN(hidden_dim,ginv=ginv)
        super().__init__(odernn,outputNN,hidden_dim,input_dim,learning_rate,update_loss)
        self.save_hyperparameters({'net':'ctLSTMModel'})
        
    def forward(self, dt, x, training = False, p = 0.0, include_update=False):
    
        T = x.size(1)
        batch_size = x.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = (torch.zeros(batch_size, self.hidden_dim,device=self.device),
               torch.zeros(batch_size, self.hidden_dim,device=self.device))

        for i in range(0,T):
            x_i = x[:,i,:]
            z_i = self.preNN(x_i)
            dt_i = dt[:,i,:]
            if (include_update == True):
                h_t_update = self.RNN.forward_update(z_i,h_t)
                o_t,c_t = h_t_update[0].squeeze(0),h_t_update[1]
                o_t1 = self.RNN.forward_ode(o_t,dt_i).squeeze(0)
                output_update[:,i,:] = self.OutputNN(o_t)
                output[:,i,:] = self.OutputNN(o_t1)
                h_t = (o_t1,c_t)
            else:
                h_t = self.RNN(z_i,h_t,dt_i)
                h_t = (h_t[0].squeeze(0),h_t[1]) # extra dimension gets tacked on by odeint
                output[:,i,:] = self.OutputNN(h_t[0])
        if (include_update == True):
            return output,output_update
        else:
            return output

    
    def forward_trajectory(self, dt, x, nsteps=10):
        T = x.size(1)
        batch_size = x.size(0)
        outputs = []
        h_t = (torch.zeros(batch_size, self.hidden_dim,device=self.device),
               torch.zeros(batch_size, self.hidden_dim,device=self.device))
        for i in range(0,T):
            x_i = x[:,i,:]
            z_i = self.preNN(x_i)
            dt_i = dt[:,i,:]
            h_t,c_t = self.RNN(z_i,h_t,dt_i,n_intermediate=nsteps)
            h_t = h_t.squeeze(0)
            outputs_i = self.OutputNN(h_t)
            outputs.append(outputs_i)
            h_t = h_t[-1]
            h_t = (h_t,c_t)
        return outputs

class dtRNNModel(BaseModelDT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1,update_loss=None):
        rnn = nn.RNNCell(hidden_dim,hidden_dim)
        outputNN = ConditionalExpectNN(hidden_dim,ginv=ginv)
        super().__init__(rnn,outputNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'dtRNNModel'})
        
class dtGRUModel(BaseModelDT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1,update_loss=None):
        rnn = nn.GRUCell(hidden_dim,hidden_dim)
        outputNN = ConditionalExpectNN(hidden_dim,ginv=ginv)
        super().__init__(rnn,outputNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'dtGRUModel'})
        
class dtLSTMModel(BaseModelDT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1,update_loss=None):
        rnn = nn.LSTMCell(hidden_dim,hidden_dim)
        outputNN = ConditionalExpectNN(hidden_dim,ginv=ginv)
        super().__init__(rnn,outputNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'dtLSTMModel'})
        
    def forward(self, dt, x, training = False, p = 0.0):
        
        T = x.size(1)
        batch_size = x.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = (torch.zeros(batch_size, self.hidden_dim,device=self.device),
               torch.zeros(batch_size, self.hidden_dim,device=self.device))
        for i in range(0,T):
            x_i = x[:,i,:]
            dt_i = dt[:,i,:]
            z_i = self.preNN(x_i)
            h_t = self.RNN(z_i,h_t)
            o_t = F.dropout(h_t[0],training=training,p=p)
            output[:,i,:] = self.OutputNN(o_t)
        return output
