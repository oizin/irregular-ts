from .base import BaseModel,BaseModelAblate
from .output import *
#from .odenet import *
#from .jumpnn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchctrnn as ct
#from torch.nn.utils.parametrizations import spectral_norm

#ode_tol = {'atol':1e-2,'rtol':1e-2}

# def g(x):
#     x = x.copy()
#     x = np.log(x) - np.log(140)
#     return x

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


# NN0 = nn.Sequential(
#             nn.Linear(dims['input_dim_0'],dims['input_dim_0'] // 2),
#             nn.Dropout(0.2),
#             nn.Tanh(),
#             nn.Linear(dims['input_dim_0'] // 2,dict_args['hidden_dim_0']),
#             nn.Dropout(0.2),
#             nn.Tanh())
# preNN = nn.Sequential(
#             nn.Linear(dims['input_dim_t']+dict_args['hidden_dim_0'],(dims['input_dim_t']+dict_args['hidden_dim_0']) // 2),
#             nn.Dropout(0.2),
#             nn.Tanh(),
#             nn.Linear((dims['input_dim_t']+dict_args['hidden_dim_0']) // 2,dict_args['hidden_dim_t']),
#             nn.Dropout(0.2),
#             nn.Tanh())


class ODEFunc(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self,hidden):
        output = self.layers(hidden)
        return output

class DecayFlow(nn.Module):
    
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        while True:
            A = torch.randn((input_dim,input_dim))
            if torch.matrix_rank(A) == input_dim:
                break
        B = torch.matmul(A.t(), A)
        self.A = nn.Parameter(B)

    def forward(self,hidden,dt):
        At = self.A.unsqueeze(0).repeat_interleave(dt.shape[0],0) * dt.view((-1,1,1))
        e_At = torch.matrix_exp(-At)
        output = torch.matmul(e_At,hidden.unsqueeze(2)).squeeze(2)
        return hidden

class Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self,input):
        output = self.layers(input)
        return output

# GRU flavours -------------------------------------------------------------------------------

class ODEGRUModel(BaseModel):
    def __init__(self,dims,outputNN,ginv,eval_fn,NN0=nn.Identity(),**kwargs):
        encoder = Encoder(dims['input_size_update'],30,dims['hidden_dim_t'])
        func = ODEFunc(dims['hidden_dim_t'],50,dims['hidden_dim_t'])
        odenet = ct.NeuralODE(func,time_func='tanh',time_dependent=False,data_dependent=False,
                            solver='euler',solver_options={'step_size':1e-1})
        odernn = ct.ODEGRUCell(odenet,dims['hidden_dim_t'],dims['hidden_dim_t'])
        outNN = outputNN(dims['hidden_dim_t'])
        super().__init__(odernn,outNN,encoder,NN0,dims,ginv,eval_fn,**kwargs)
        self.save_hyperparameters({'model':'ODEGRUModel'})

class FlowGRUModel(BaseModel):
    def __init__(self,dims,outputNN,ginv,eval_fn,NN0=nn.Identity(),**kwargs):
        encoder = Encoder(dims['input_size_update'],12,dims['hidden_dim_t'])
        func = ct.ResNetFlow(dims['hidden_dim_t'],12)
        odenet = ct.NeuralFlow(func)
        odernn = ct.FlowGRUCell(odenet,dims['hidden_dim_t'],dims['hidden_dim_t'])
        outNN = outputNN(dims['hidden_dim_t'])
        super().__init__(odernn,outNN,encoder,NN0,dims,ginv,eval_fn,**kwargs)
        self.save_hyperparameters({'model':'FlowGRUModel'})

class GRUModel(BaseModelAblate):
    def __init__(self,dims,outputNN,ginv,eval_fn,NN0=nn.Identity(),**kwargs):
        encoder = Encoder(dims['input_size_update'],30,dims['hidden_dim_t'])
        rnn = nn.RNNCell(dims['hidden_dim_t'],dims['hidden_dim_t'])
        outNN = outputNN(dims['hidden_dim_t'])
        super().__init__(rnn,outNN,encoder,NN0,dims,ginv,eval_fn,**kwargs)
        self.save_hyperparameters({'model':'GRUModel'})

class DecayGRUModel(BaseModel):
    def __init__(self,dims,outputNN,ginv,eval_fn,NN0=nn.Identity(),**kwargs):
        encoder = Encoder(dims['input_size_update'],30,dims['hidden_dim_t'])
        func = DecayFlow(dims['hidden_dim_t'])
        odenet = ct.NeuralFlow(func)
        odernn = ct.FlowGRUCell(odenet,dims['hidden_dim_t'],dims['hidden_dim_t'])
        outNN = outputNN(dims['hidden_dim_t'])
        super().__init__(odernn,outNN,encoder,NN0,dims,ginv,eval_fn,**kwargs)
        self.save_hyperparameters({'model':'DecayGRUModel'})

# LSTM flavours -------------------------------------------------------------------------------

class BaseModelLSTM(BaseModel):
    # def __init__(self,RNN,OutputNN,preNN,NN0,dims,ginv,**kwargs):
    #     super().__init__(RNN,OutputNN,preNN,NN0,dims,ginv,**kwargs)
    
    def forward(self, dt, x, training = False, p = 0.0, include_update=False):
        xt,x0,xi = x
        T = xt.size(1)
        batch_size = xt.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = (torch.zeros(batch_size, self.hidden_dim_t,device=self.device),
               torch.zeros(batch_size, self.hidden_dim_t,device=self.device))
        z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xi_i = xi[:,i,:]
            xt_i = self.preNN(torch.cat((xt_i,z0),1))
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
        
#     def forward_trajectory(self, dt, x, nsteps=10):
#         xt,x0,xi = x
#         T = xt.size(1)
#         batch_size = xt.size(0)
#         outputs = []
#         h_t = (torch.zeros(batch_size, self.hidden_dim_t,device=self.device),
#                torch.zeros(batch_size, self.hidden_dim_t,device=self.device))
#         z0 = self.NN0(x0)
#         for i in range(0,T):
#             xt_i = xt[:,i,:]
#             xi_i = xi[:,i,:]
#             xt_i = self.preNN(torch.cat((xt_i,z0),1))
#             dt_i = dt[:,i,:]
#             h_t_update = self.RNN.forward_update(xt_i,h_t)
#             o_t_update,c_t = h_t_update[0],h_t_update[1]
#             o_t1 = self.RNN.forward_ode(o_t_update,dt_i,xi_i,n_intermediate=nsteps).squeeze(0)
#             outputs_i = self.OutputNN(o_t1)
#             outputs.append(outputs_i)
#             o_t1 = o_t1[-1]
#             h_t = (o_t1,c_t)
#         return outputs

class ODELSTMModel(BaseModelLSTM):

    def __init__(self,dims,outputNN,ginv,eval_fn,NN0=nn.Identity(),**kwargs):
        encoder = Encoder(dims['input_size_update'],30,dims['hidden_dim_t'])
        func = ODEFunc(dims['hidden_dim_t'],50,dims['hidden_dim_t'])
        odenet = ct.NeuralODE(func,time_func='tanh',time_dependent=False,data_dependent=False,
                            solver='euler',solver_options={'step_size':1e-1})
        odernn = ct.ODELSTMCell(odenet,dims['hidden_dim_t'],dims['hidden_dim_t'])
        outNN = outputNN(dims['hidden_dim_t'])
        super().__init__(odernn,outNN,encoder,NN0,dims,ginv,eval_fn,**kwargs)
        self.save_hyperparameters({'model':'ODELSTMModel'})
                
class FlowLSTMModel(BaseModelLSTM):

    def __init__(self,dims,outputNN,ginv,eval_fn,NN0=nn.Identity(),**kwargs):
        encoder = Encoder(dims['input_size_update'],30,dims['hidden_dim_t'])
        func = ct.ResNetFlow(dims['hidden_dim_t'],50)
        odenet = ct.NeuralFlow(func)
        odernn = ct.FlowLSTMCell(odenet,dims['hidden_dim_t'],dims['hidden_dim_t'])
        outNN = outputNN(dims['hidden_dim_t'])
        super().__init__(odernn,outNN,encoder,NN0,dims,ginv,eval_fn,**kwargs)
        self.save_hyperparameters({'model':'FlowLSTMModel'})
        
# IMODE model -----------------------------------------------------------------------------------------

class IMODE_ODENet(nn.Module):
    def __init__(self,dims):
        super().__init__()
        
        # dimensions
        hx_dim,hi_dim = dims['hidden_dim_t'],dims['hidden_dim_i']
        self.hx_dim = hx_dim
        self.hi_dim = hi_dim
        
        # neural nets
        self.hx_net = nn.Sequential(
            nn.Linear(hx_dim + hi_dim, (hx_dim + hi_dim)*2),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear((hx_dim + hi_dim)*2, hx_dim),
            nn.Tanh(),
        )
        self.hi_net = nn.Sequential(
            nn.Linear(hi_dim, (hi_dim)*2),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear((hi_dim)*2, hi_dim),
            nn.Tanh(),
        )
        
        # initial parameters
        for m in self.hx_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        for m in self.hi_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self,hidden):
        h_i = hidden[:,(self.hx_dim):(self.hx_dim+self.hi_dim)]
        h_x = self.hx_net(hidden)
        h_i = self.hi_net(h_i)
        h_all = torch.cat((h_x,h_i),1)
        return h_all

class IMODE_UpdateNN(nn.Module):
    def __init__(self,dims):
        super().__init__()
        
        # dimensions
        self.hx_dim = dims['hidden_dim_t']
        self.hi_dim = dims['hidden_dim_i']
        self.x_input = dims['input_dim_t'] + dims['input_dim_0']
        self.i_input = dims['input_dim_i']
        
        # neural nets
        self.hx_net = nn.Sequential(
            nn.Linear(self.hx_dim + self.x_input, (self.hx_dim + self.x_input)*2),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear((self.hx_dim + self.x_input)*2, self.hx_dim),
            nn.Tanh(),
        )
        self.hi_net = nn.Sequential(
            nn.Linear(self.hi_dim + self.hx_dim + self.i_input, (self.hi_dim + self.hx_dim + self.i_input)*2),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear((self.hi_dim + self.hx_dim + self.i_input)*2, self.hi_dim),
            nn.Tanh(),
        )
        
    def forward(self,input,hidden):
        xt,xi = input
        h_x = hidden[:,0:(self.hx_dim)]
        h_i = hidden[:,(self.hx_dim):(self.hx_dim+self.hi_dim)]
        h_x = self.hx_net(torch.cat((h_x,xt),1))
        h_i = self.hi_net(torch.cat((h_x,h_i,xi),1))
        h_all = torch.cat((h_x,h_i),1)
        return h_all

class IMODE(BaseModel):
    
    def __init__(self,dims,outputNN,ginv,eval_fn,NN0=nn.Identity(),**kwargs):
        # input_dim_t = dims['input_dim_t']
        # input_dim_i = dims['input_dim_i']
        # input_dim_0 = dims['input_dim_0']

        self.hidden_dim_t = dims['hidden_dim_t']
        self.hidden_dim_0 = dims['hidden_dim_0']
        self.hidden_dim_x = dims['hidden_dim_t']
        self.hidden_dim_i = dims['hidden_dim_i']
        
        func = IMODE_ODENet(dims)
        odenet = ct.NeuralODE(func,time_func='tanh',
                            solver='euler',solver_options={'step_size':1e-1})
        jumpnn = IMODE_UpdateNN(dims)
        odernn = ct.neuralJumpODECell(jumpnn,odenet)
        gaussianNN = outputNN(self.hidden_dim_t)
        super().__init__(odernn,gaussianNN,nn.Identity(),NN0,dims,ginv,eval_fn,**kwargs)
        self.save_hyperparameters({'model':'IMODE'})
        
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
        z0 = self.NN0(x0)
        for i in range(0,T):
            xt_i = xt[:,i,:]
            xt_i = self.preNN(torch.cat((xt_i,z0),1))
            dt_i = dt[:,i,:]
            if (include_update == True):
                h_t_update = self.RNN.forward_update((xt_i,xi[:,i,:]),h_t)
                h_t = self.RNN.forward_ode(h_t_update,dt_i)
                output_update[:,i,:] = self.OutputNN(h_t_update[:,0:self.hidden_dim_x])
                output[:,i,:] = self.OutputNN(h_t[:,0:self.hidden_dim_x])
            else:
                h_t = self.RNN((xt_i,xi[:,i,:]),h_t,dt_i)
                output[:,i,:] = self.OutputNN(h_t[:,0:self.hidden_dim_x])
        if (include_update == True):
            return output,output_update
        else:
            return output
        
#     def forward_trajectory(self, dt, x, nsteps=10):
#         xt,x0,xi = x
#         T = xt.size(1)
#         batch_size = xt.size(0)
#         outputs = []
#         h_t = torch.zeros(batch_size, self.hidden_dim_x+self.hidden_dim_i,device=self.device)
#         if (self.NN0 != None):
#             z0 = self.NN0(x0)
#         for i in range(0,T):
#             xt_i = xt[:,i,:]
#             if (self.NN0 != None) & (self.preNN != None):
#                 xt_i = self.preNN(torch.cat((xt_i,z0),1))
#             elif (self.preNN != None):
#                 xt_i = self.preNN(xt_i)
#             dt_i = dt[:,i,:]
#             h_t = self.RNN((xt_i,xi[:,i,:]),h_t,dt_i,n_intermediate=nsteps).squeeze(0)
#             outputs_i = self.OutputNN(h_t[:,:,0:self.hidden_dim_x])
#             outputs.append(outputs_i)
#             h_t = h_t[-1]
#         return outputs


# Feedforward models -------------------------------------------------------------------------------

# class MLP(BaseModel):












# class ODEGRUBayes(BaseModelCT):

#     def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
#         if preNN is None:
#             input_size_update = input_dims['input_dim_t']
#         else:
#             input_size_update = hidden_dims['hidden_dim_t']
#         odenet = GRUNet(hidden_dims['hidden_dim_t'],input_dims['input_dim_i'])
#         odernn = torchctrnn.ODEGRUCell(odenet,input_size_update,
#                                        tol={'atol':1e-8,'rtol':1e-8},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
#         outNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
#         super().__init__(odernn,outNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
#         self.save_hyperparameters({'net':'ODEGRUBayes'})


# class neuralJumpModel(BaseModelCT):

#     def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
#         if preNN is None:
#             input_size_update = input_dims['input_dim_t']
#         else:
#             input_size_update = hidden_dims['hidden_dim_t']
#         odenet = ODENetHITT(hidden_dims['hidden_dim_t'],input_size_update)
#         jumpnn = JumpNet1(hidden_dims['hidden_dim_t'],input_size_update)
#         ctjumpnn = torchctrnn.neuralJumpODECell(jumpnn,odenet,input_size_update,
#                                                 tol={'atol':1e-2,'rtol':1e-2},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
#         gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
#         super().__init__(ctjumpnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
#         self.save_hyperparameters({'net':'neuralJumpModel'})
        
#     def forward(self, dt, x, training = False, p = 0.0, include_update=False):
#         """
#         x a tuple
#         """
#         xt,x0,xi = x
#         T = xt.size(1)
#         batch_size = xt.size(0)
#         output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
#         output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
#         h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
#         if (self.NN0 != None):
#             z0 = self.NN0(x0)
#         for i in range(0,T):
#             xt_i = xt[:,i,:]
#             xi_i = xi[:,i,:]
#             if (self.NN0 != None) & (self.preNN != None):
#                 xt_i = self.preNN(torch.cat((xt_i,z0),1))
#             elif (self.preNN != None):
#                 xt_i = self.preNN(xt_i)
#             dt_i = dt[:,i,:]
#             if (include_update == True):
#                 h_t_update = self.RNN.forward_update(xt_i,h_t)
#                 h_t = self.RNN.forward_ode(h_t_update,dt_i,xt_i).squeeze(0)
#                 output_update[:,i,:] = self.OutputNN(h_t_update)
#                 output[:,i,:] = self.OutputNN(h_t)
#             else:
#                 h_t = self.RNN(xt_i,h_t,dt_i,xt_i).squeeze(0)
#                 output[:,i,:] = self.OutputNN(h_t)
#         if (include_update == True):
#             return output,output_update
#         else:
#             return output
        
#     def forward_trajectory(self, dt, x, nsteps=10):
#         xt, x0,xi = x
#         T = xt.size(1)
#         batch_size = xt.size(0)
#         outputs = []
#         h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
#         if (self.NN0 != None):
#             z0 = self.NN0(x0)
#         for i in range(0,T):
#             xt_i = xt[:,i,:]
#             xi_i = xi[:,i,:]
#             if (self.NN0 != None) & (self.preNN != None):
#                 xt_i = self.preNN(torch.cat((xt_i,z0),1))
#             elif (self.preNN != None):
#                 xt_i = self.preNN(xt_i)
#             dt_i = dt[:,i,:]
#             h_t = self.RNN(xt_i,h_t,dt_i,xt_i,n_intermediate=nsteps).squeeze(0)
#             outputs_i = self.OutputNN(h_t)
#             outputs.append(outputs_i)
#             h_t = h_t[-1]
#         return outputs

# #     def forward_(self,dt,xt,h_t,xi,include_update):
# #         if (include_update == True):
# #             h_t_update = self.RNN.forward_update(xt,h_t)
# #             h_t = self.RNN.forward_ode(h_t_update,dt,xt).squeeze(0)
# #             return self.OutputNN(h_t_update),self.OutputNN(h_t)
# #         else:
# #             h_t = self.RNN(xt,h_t,dt,xi).squeeze(0)
# #             return self.OutputNN(h_t)

# class resNeuralJumpModel(BaseModelCT):

#     def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=0.1,merror=1e-2,dt_scaler=1.0):
#         if preNN is None:
#             input_size_update = input_dims['input_dim_t']
#         else:
#             input_size_update = hidden_dims['hidden_dim_t']
#         odenet = ODENetHITT(hidden_dims['hidden_dim_t'],input_size_update)
#         jumpnn = JumpNet2(hidden_dims['hidden_dim_t'],input_size_update)
#         ctjumpnn = torchctrnn.neuralJumpODECell(jumpnn,odenet,input_size_update,tol={'atol':1e-2,'rtol':1e-2},method='euler',options={'step_size':0.1},dt_scaler=dt_scaler)
#         gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
#         super().__init__(ctjumpnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate,update_loss,merror)
#         self.save_hyperparameters({'net':'resNeuralJumpModel'})

#     def forward(self, dt, x, training = False, p = 0.0, include_update=False):
#         """
#         x a tuple
#         """
#         xt,x0,xi = x
#         T = xt.size(1)
#         batch_size = xt.size(0)
#         output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
#         output_update = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
#         h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
#         if (self.NN0 != None):
#             z0 = self.NN0(x0)
#         for i in range(0,T):
#             xt_i = xt[:,i,:]
#             xi_i = xi[:,i,:]
#             if (self.NN0 != None) & (self.preNN != None):
#                 xt_i = self.preNN(torch.cat((xt_i,z0),1))
#             elif (self.preNN != None):
#                 xt_i = self.preNN(xt_i)
#             dt_i = dt[:,i,:]
#             if (include_update == True):
#                 h_t_update = self.RNN.forward_update(xt_i,h_t)
#                 h_t = self.RNN.forward_ode(h_t_update,dt_i,xt_i).squeeze(0)
#                 output_update[:,i,:] = self.OutputNN(h_t_update)
#                 output[:,i,:] = self.OutputNN(h_t)
#             else:
#                 h_t = self.RNN(xt_i,h_t,dt_i,xt_i).squeeze(0)
#                 output[:,i,:] = self.OutputNN(h_t)
#         if (include_update == True):
#             return output,output_update
#         else:
#             return output
        
#     def forward_trajectory(self, dt, x, nsteps=10):
#         xt, x0,xi = x
#         T = xt.size(1)
#         batch_size = xt.size(0)
#         outputs = []
#         h_t = torch.zeros(batch_size, self.hidden_dim_t,device=self.device)
#         if (self.NN0 != None):
#             z0 = self.NN0(x0)
#         for i in range(0,T):
#             xt_i = xt[:,i,:]
#             xi_i = xi[:,i,:]
#             if (self.NN0 != None) & (self.preNN != None):
#                 xt_i = self.preNN(torch.cat((xt_i,z0),1))
#             elif (self.preNN != None):
#                 xt_i = self.preNN(xt_i)
#             dt_i = dt[:,i,:]
#             h_t = self.RNN(xt_i,h_t,dt_i,xt_i,n_intermediate=nsteps).squeeze(0)
#             outputs_i = self.OutputNN(h_t)
#             outputs.append(outputs_i)
#             h_t = h_t[-1]
#         return outputs

# #     def forward_(self,dt,xt,h_t,xi,include_update):
# #         if (include_update == True):
# #             h_t_update = self.RNN.forward_update(xt,h_t)
# #             h_t = self.RNN.forward_ode(h_t_update,dt,xt).squeeze(0)
# #             return self.OutputNN(h_t_update),self.OutputNN(h_t)
# #         else:
# #             h_t = self.RNN(xt,h_t,dt,xi).squeeze(0)
# #             return self.OutputNN(h_t)
            

# # class dtRNNModel(BaseModelDT):

# #     def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=None,merror=1e-2,dt_scaler=1.0):
# #         if preNN is None:
# #             input_size_update = input_dims['input_dim_t']
# #         else:
# #             input_size_update = hidden_dims['hidden_dim_t']        
# #         rnn = nn.RNNCell(input_size_update,hidden_dims['hidden_dim_t'])
# #         gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
# #         super().__init__(rnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate)
# #         self.save_hyperparameters({'net':'dtRNNModel'})
        
        
# class LSTMModel(BaseModelDT):

#     def __init__(self,input_dims,hidden_dims,outputNN,preNN=None,NN0=None,learning_rate=0.1,update_loss=None,merror=1e-2,dt_scaler=1.0):
#         if preNN is None:
#             input_size_update = input_dims['input_dim_t']
#         else:
#             input_size_update = hidden_dims['hidden_dim_t']        
#         rnn = nn.LSTMCell(input_size_update,hidden_dims['hidden_dim_t']   )
#         gaussianNN = outputNN(hidden_dims['hidden_dim_t'],g=g,ginv=ginv)
#         super().__init__(rnn,gaussianNN,preNN,NN0,hidden_dims,input_dims,learning_rate)
#         self.save_hyperparameters({'net':'dtLSTMModel'})
        
#     def forward(self, dt, x, training = False, p = 0.0):
#         xt,x0,xi = x
#         T = xt.size(1)
#         batch_size = xt.size(0)
#         output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
#         h_t = (torch.zeros(batch_size, self.hidden_dim_t,device=self.device),
#                torch.zeros(batch_size, self.hidden_dim_t,device=self.device))
#         if (self.NN0 != None):
#             z0 = self.NN0(x0)
#         for i in range(0,T):
#             xt_i = xt[:,i,:]
#             if (self.NN0 != None) & (self.preNN != None):
#                 xt_i = self.preNN(torch.cat((xt_i,z0),1))
#             dt_i = dt[:,i,:]
#             h_t = self.RNN(xt_i,h_t)
#             o_t = F.dropout(h_t[0],training=training,p=p)
#             output[:,i,:] = self.OutputNN(o_t)
#         return output