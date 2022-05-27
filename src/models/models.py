from .base import BaseModel,BaseModelCT,BaseModelDT,BaseModelDecay
from .output import *
from .odenet import *
from .jumpnn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchctrnn as ct

#ode_tol = {'atol':1e-2,'rtol':1e-2}

def ginv(x):
    x = x.copy()
    x = np.exp(x + np.log(140))
    return x
def g(x):
    x = x.copy()
    x = np.log(x) - np.log(140)
    return x

# class ctRNNModel(BaseModelCT):

#     def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
#         if preNN is None:
#             input_size_update = input_dims['input_dim_t']
#         else:
#             input_size_update = hidden_dims['hidden_dim_t']
#         odenet = ODENetHI(hidden_dims['hidden_dim_t'],input_dims['input_dim_i'])
#         odernn = torchctrnn.ODERNNCell(odenet,input_size_update,
#                                        tol={'atol':1e-2,'rtol':1e-2},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
#         outNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
#         super().__init__(odernn,outNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
#         self.save_hyperparameters({'net':'ctRNNModel'})


class ctGRUModel(BaseModelCT):

    def __init__(self,dims,outputNN,preNN=nn.Identity(),NN0=nn.Identity(),learning_rate=1e-3,update_loss=0.1,merror=1e-2):
        func = nn.Sequential(
            nn.Linear(dims['hidden_dim_t'], 50),
            nn.Tanh(),
            nn.Linear(50, dims['hidden_dim_t']),
            nn.Tanh()
        )
        odenet = ct.NeuralODE(func,time_func=lambda x : x / 100.0,time_dependent=False,data_dependent=False,
                            solver='euler',solver_options={'step_size':1e-1})
        odernn = ct.ODEGRUCell(odenet,dims['input_size_update'],dims['hidden_dim_t'])
        outNN = outputNN(dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(odernn,outNN,preNN,NN0,dims,learning_rate,update_loss,merror)
        self.save_hyperparameters({'net':'ctGRUModel'})

class ODEGRUBayes(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
        if preNN is None:
            input_size_update = input_dims['input_dim_t']
        else:
            input_size_update = hidden_dims['hidden_dim_t']
        odenet = GRUNet(hidden_dims['hidden_dim_t'],input_dims['input_dim_i'])
        odernn = torchctrnn.ODEGRUCell(odenet,input_size_update,
                                       tol={'atol':1e-8,'rtol':1e-8},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
        outNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(odernn,outNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
        self.save_hyperparameters({'net':'ODEGRUBayes'})

class ctLSTMModel(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
        if preNN is None:
            input_size_update = input_dims['input_dim_t']
        else:
            input_size_update = hidden_dims['hidden_dim_t']
        odenet = ODENetHI(hidden_dims['hidden_dim_t'],input_dims['input_dim_i'])
        odernn = torchctrnn.ODELSTMCell(odenet,input_size_update,
                                        tol={'atol':1e-2,'rtol':1e-2},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
        outNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(odernn,outNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
        self.save_hyperparameters({'net':'ctLSTMModel'})
        
    def forward(self, dt, x, training = False, p = 0.0, include_update=False):
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = (torch.zeros(batch_size, self.hidden_dim_t,device=self.device),
               torch.zeros(batch_size, self.hidden_dim_t,device=self.device))
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            
            if (include_update == True):
                h_t_update = self.RNN.forward_update(xt_i,h_t)
                o_t_update,c_t = h_t_update[0].squeeze(0),h_t_update[1]
                o_t1 = self.RNN.forward_ode(o_t_update,dt_i,xi_i).squeeze(0)
                output_update[:,i,:] = self.OutputNN(o_t_update)
                output[:,i,:] = self.OutputNN(o_t1)
                h_t = (o_t1,c_t)
            else:
                h_t_update = self.RNN.forward_update(xt_i,h_t)
                o_t_update,c_t = h_t_update[0].squeeze(0),h_t_update[1]
                o_t1 = self.RNN.forward_ode(o_t_update,dt_i,xi_i).squeeze(0)
                output[:,i,:] = self.OutputNN(o_t1)
                h_t = (o_t1,c_t)
        if (include_update == True):
            return output,output_update
        else:
            return output
        
    def forward_trajectory(self, dt, x, nsteps=10):
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        outputs = []
        h_t = (torch.zeros(batch_size, self.hidden_dim_t,device=self.device),
               torch.zeros(batch_size, self.hidden_dim_t,device=self.device))
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            h_t_update = self.RNN.forward_update(xt_i,h_t)
            o_t_update,c_t = h_t_update[0],h_t_update[1]
            o_t1 = self.RNN.forward_ode(o_t_update,dt_i,xi_i,n_intermediate=nsteps).squeeze(0)
            outputs_i = self.OutputNN(o_t1)
            outputs.append(outputs_i)
            o_t1 = o_t1[-1]
            h_t = (o_t1,c_t)
        return outputs

class neuralJumpModel(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
        if preNN is None:
            input_size_update = input_dims['input_dim_t']
        else:
            input_size_update = hidden_dims['hidden_dim_t']
        odenet = ODENetHITT(hidden_dims['hidden_dim_t'],input_size_update)
        jumpnn = JumpNet1(hidden_dims['hidden_dim_t'],input_size_update)
        ctjumpnn = torchctrnn.neuralJumpODECell(jumpnn,odenet,input_size_update,
                                                tol={'atol':1e-2,'rtol':1e-2},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
        gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(ctjumpnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
        self.save_hyperparameters({'net':'neuralJumpModel'})
        
    def forward(self, dt, x, training = False, p = 0.0, include_update=False):
        """
        x a tuple
        """
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            if (include_update == True):
                h_t_update = self.RNN.forward_update(xt_i,h_t)
                h_t = self.RNN.forward_ode(h_t_update,dt_i,xt_i).squeeze(0)
                output_update[:,i,:] = self.OutputNN(h_t_update)
                output[:,i,:] = self.OutputNN(h_t)
            else:
                h_t = self.RNN(xt_i,h_t,dt_i,xt_i).squeeze(0)
                output[:,i,:] = self.OutputNN(h_t)
        if (include_update == True):
            return output,output_update
        else:
            return output
        
    def forward_trajectory(self, dt, x, nsteps=10):
        xt, x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        outputs = []
        h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            h_t = self.RNN(xt_i,h_t,dt_i,xt_i,n_intermediate=nsteps).squeeze(0)
            outputs_i = self.OutputNN(h_t)
            outputs.append(outputs_i)
            h_t = h_t[-1]
        return outputs

#     def forward_(self,dt,xt,h_t,xi,include_update):
#         if (include_update == True):
#             h_t_update = self.RNN.forward_update(xt,h_t)
#             h_t = self.RNN.forward_ode(h_t_update,dt,xt).squeeze(0)
#             return self.OutputNN(h_t_update),self.OutputNN(h_t)
#         else:
#             h_t = self.RNN(xt,h_t,dt,xi).squeeze(0)
#             return self.OutputNN(h_t)

class resNeuralJumpModel(BaseModelCT):

    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
        if preNN is None:
            input_size_update = input_dims['input_dim_t']
        else:
            input_size_update = hidden_dims['hidden_dim_t']
        odenet = ODENetHITT(hidden_dims['hidden_dim_t'],input_size_update)
        jumpnn = JumpNet2(hidden_dims['hidden_dim_t'],input_size_update)
        ctjumpnn = torchctrnn.neuralJumpODECell(jumpnn,odenet,input_size_update,tol={'atol':1e-2,'rtol':1e-2},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
        gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(ctjumpnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
        self.save_hyperparameters({'net':'resNeuralJumpModel'})

    def forward(self, dt, x, training = False, p = 0.0, include_update=False):
        """
        x a tuple
        """
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            if (include_update == True):
                h_t_update = self.RNN.forward_update(xt_i,h_t)
                h_t = self.RNN.forward_ode(h_t_update,dt_i,xt_i).squeeze(0)
                output_update[:,i,:] = self.OutputNN(h_t_update)
                output[:,i,:] = self.OutputNN(h_t)
            else:
                h_t = self.RNN(xt_i,h_t,dt_i,xt_i).squeeze(0)
                output[:,i,:] = self.OutputNN(h_t)
        if (include_update == True):
            return output,output_update
        else:
            return output
        
    def forward_trajectory(self, dt, x, nsteps=10):
        xt, x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        outputs = []
        h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            h_t = self.RNN(xt_i,h_t,dt_i,xt_i,n_intermediate=nsteps).squeeze(0)
            outputs_i = self.OutputNN(h_t)
            outputs.append(outputs_i)
            h_t = h_t[-1]
        return outputs

#     def forward_(self,dt,xt,h_t,xi,include_update):
#         if (include_update == True):
#             h_t_update = self.RNN.forward_update(xt,h_t)
#             h_t = self.RNN.forward_ode(h_t_update,dt,xt).squeeze(0)
#             return self.OutputNN(h_t_update),self.OutputNN(h_t)
#         else:
#             h_t = self.RNN(xt,h_t,dt,xi).squeeze(0)
#             return self.OutputNN(h_t)
            
class IMODE(BaseModelCT):
    
    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
        input_dim_t = input_dims['input_dim_t']
        input_dim_i = input_dims['input_dim_i']
        input_dim_0 = input_dims['input_dim_0']
        if preNN is None:
            input_dims = input_dims
        else:
            input_dims['input_dim_t'] = hidden_dims['hidden_dim_t']

        self.hidden_dim_t = hidden_dims['hidden_dim_t']
        hidden_dim_0 = hidden_dims['hidden_dim_0']
        self.hidden_dim_x = hidden_dims['hidden_dim_t']
        self.hidden_dim_i = hidden_dims['hidden_dim_i']
        
        odenet = IMODE_ODENet(hidden_dims,input_dims)
        jumpnn = IMODE_JumpNN(hidden_dims,input_dims)
        ctjumpnn = torchctrnn.neuralJumpODECell(jumpnn,odenet,None,tol={'atol':1e-2,'rtol':1e-2},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
        gaussianNN = outputNN(self.hidden_dim_t,g=g,ginv=ginv)
        super().__init__(ctjumpnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
        self.save_hyperparameters({'net':'IMODE'})
        
    def forward(self, dt, x, training = False, p = 0.0, include_update=False):
        """
        x a tuple
        """
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim_x+self.hidden_dim_i,device=self.device)
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            
            if (include_update == True):
                h_t_update = self.RNN.forward_update((xt_i,xi[:,i,:]),h_t)
                h_t = self.RNN.forward_ode(h_t_update,dt_i).squeeze(0)
                output_update[:,i,:] = self.OutputNN(h_t_update[:,0:self.hidden_dim_x])
                output[:,i,:] = self.OutputNN(h_t[:,0:self.hidden_dim_x])
            else:
                h_t = self.RNN((xt_i,xi[:,i,:]),h_t,dt_i).squeeze(0)
                output[:,i,:] = self.OutputNN(h_t[:,0:self.hidden_dim_x])
        if (include_update == True):
            return output,output_update
        else:
            return output
        
    def forward_trajectory(self, dt, x, nsteps=10):
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        outputs = []
        h_t = torch.zeros(batch_size, self.hidden_dim_x+self.hidden_dim_i,device=self.device)
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            elif (self.preNN != None):
                xt_i = self.preNN(xt_i)
            dt_i = dt[:,i,:]
            h_t = self.RNN((xt_i,xi[:,i,:]),h_t,dt_i,n_intermediate=nsteps).squeeze(0)
            outputs_i = self.OutputNN(h_t[:,:,0:self.hidden_dim_x])
            outputs.append(outputs_i)
            h_t = h_t[-1]
        return outputs

class dtRNNModel(BaseModelDT):

    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=None,merror=1e-2,dt_scaler=1.0):
        if preNN is None:
            input_size_update = input_dims['input_dim_t']
        else:
            input_size_update = hidden_dims['hidden_dim_t']        
        rnn = nn.RNNCell(input_size_update,hidden_dims['hidden_dim_t'])
        gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(rnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate)
        self.save_hyperparameters({'net':'dtRNNModel'})
        
class dtGRUModel(BaseModelDT):

    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=None,merror=1e-2,dt_scaler=1.0):
        if preNN is None:
            input_size_update = input_dims['input_dim_t']
        else:
            input_size_update = hidden_dims['hidden_dim_t']        
        rnn = nn.GRUCell(input_size_update,hidden_dims['hidden_dim_t'])
        gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(rnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate)
        self.save_hyperparameters({'net':'dtGRUModel'})
        
class dtLSTMModel(BaseModelDT):

    def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=None,merror=1e-2,dt_scaler=1.0):
        if preNN is None:
            input_size_update = input_dims['input_dim_t']
        else:
            input_size_update = hidden_dims['hidden_dim_t']        
        rnn = nn.LSTMCell(input_size_update,hidden_dims['hidden_dim_t']   )
        gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
        super().__init__(rnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate)
        self.save_hyperparameters({'net':'dtLSTMModel'})
        
    def forward(self, dt, x, training = False, p = 0.0):
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = (torch.zeros(batch_size, self.hidden_dim_t,device=self.device),
               torch.zeros(batch_size, self.hidden_dim_t,device=self.device))
        if (self.NN0 != None):
            z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            if (self.NN0 != None) & (self.preNN != None):
                xt_i = self.preNN(torch.cat((xt_i,z0),1))
            dt_i = dt[:,i,:]
            h_t = self.RNN(xt_i,h_t)
            o_t = F.dropout(h_t[0],training=training,p=p)
            output[:,i,:] = self.OutputNN(o_t)
        return output