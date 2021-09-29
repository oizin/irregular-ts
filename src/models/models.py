from .base import BaseModel,BaseModelCT,BaseModelDT,BaseModelDecay
from .output import *
from .odenet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchctrnn

class ctRNNModel(BaseModelCT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1):
        odenet = FF1(hidden_dim,input_dim)
        odernn = torchctrnn.ODERNNCell(odenet,input_dim)
        gaussianNN = GaussianOutputNN(hidden_dim)
        super().__init__(odernn,gaussianNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'ctRNNModel'})
        
class ctGRUModel(BaseModelCT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1):
        odenet = FF1(hidden_dim,input_dim)
        odernn = torchctrnn.ODEGRUCell(odenet,input_dim)
        gaussianNN = GaussianOutputNN(hidden_dim)
        super().__init__(odernn,gaussianNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'ctGRUModel'})
        
class ctLSTMModel(BaseModelCT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1):
        odenet = FF1(hidden_dim,input_dim)
        odernn = torchctrnn.ODELSTMCell(odenet,input_dim)
        gaussianNN = GaussianOutputNN(hidden_dim)
        super().__init__(odernn,gaussianNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'ctLSTMModel'})
        
    def forward(self, dt, x, training = False, p = 0.0):
        
        T = x.size(1)
        batch_size = x.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = (torch.zeros(batch_size, self.hidden_dim,device=self.device),
               torch.zeros(batch_size, self.hidden_dim,device=self.device))
        for i in range(0,T):
            x_i = x[:,i,:]
            dt_i = dt[:,i,:]
            h_t = self.RNN(x_i,h_t,dt_i)
            h_t = (h_t[0].squeeze(0),h_t[1])  # extra dimension gets tacked on by odeint
            o_t = F.dropout(h_t[0],training=training,p=p)
            output[:,i,:] = self.OutputNN(o_t)
        return output
    
    def forward_trajectory(self, dt, x, nsteps=10):
        T = x.size(1)
        batch_size = x.size(0)
        outputs = []
        h_t = (torch.zeros(batch_size, self.hidden_dim,device=self.device),
               torch.zeros(batch_size, self.hidden_dim,device=self.device))
        for i in range(0,T):
            x_i = x[:,i,:]
            dt_i = dt[:,i,:]
            h_t,c_t = self.RNN(x_i,h_t,dt_i,n_intermediate=nsteps)
            h_t = h_t.squeeze(0)
            outputs_i = self.OutputNN(h_t)
            outputs.append(outputs_i)
            h_t = h_t[-1]
            h_t = (h_t,c_t)
        return outputs


class latentJumpModel(BaseModelCT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1):
        odenet = FF1(hidden_dim,input_dim)
        jumpnn = FF2(hidden_dim,input_dim)
        ctjumpnn = torchctrnn.LatentJumpODECell(jumpnn,odenet)
        gaussianNN = GaussianOutputNN(hidden_dim)
        super().__init__(ctjumpnn,gaussianNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'latentJumpModel'})

class dtRNNModel(BaseModelDT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1):
        rnn = nn.RNNCell(input_dim,hidden_dim)
        gaussianNN = GaussianOutputNN(hidden_dim)
        super().__init__(rnn,gaussianNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'dtRNNModel'})
        
class dtGRUModel(BaseModelDT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1):
        rnn = nn.GRUCell(input_dim,hidden_dim)
        gaussianNN = GaussianOutputNN(hidden_dim)
        super().__init__(rnn,gaussianNN,hidden_dim,input_dim,learning_rate)
        self.save_hyperparameters({'net':'dtGRUModel'})
        
class dtLSTMModel(BaseModelDT):

    def __init__(self,input_dim,hidden_dim,learning_rate=0.1):
        rnn = nn.LSTMCell(input_dim,hidden_dim)
        gaussianNN = GaussianOutputNN(hidden_dim)
        super().__init__(rnn,gaussianNN,hidden_dim,input_dim,learning_rate)
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
            h_t = self.RNN(x_i,h_t)
            o_t = F.dropout(h_t[0],training=training,p=p)
            output[:,i,:] = self.OutputNN(o_t)
        return output
