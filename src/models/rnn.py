import torch 
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
import numpy as np
import inspect

# Notes
# training data of shape: (batch,seq,features)
# this is for one time so data of shape (batch,features), time index dropped

class UpdateNNBase(nn.Module):
    def __init__(self,UpdateNN):
        super(UpdateNNBase, self).__init__()

        args = inspect.getfullargspec(UpdateNN.forward)[0]
        if (args != ['self', 'input', 'hidden']) and (args != ['self', 'input', 'hx']):
            raise NameError("UpdateNN's forward method should have arguments: 'input' and 'hidden' or 'input' and 'hx' (in that order)")
        if 'hidden_size' not in dir(UpdateNN):
            raise ValueError("UpdateNN should have attribute hidden_size")

        self.net = UpdateNN
    
    def forward(self,input,hidden):
        """function with types documented in the docstring.
        
        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            The return value. True for success, False otherwise.
        """
        output = self.net(input,hidden)
        return output
                
class ODENetBase(nn.Module):
    def __init__(self,ODENet):
        super(ODENetBase, self).__init__()
        
        args = inspect.getfullargspec(ODENet.forward)[0]
        if args != ['self', 'input', 't', 'hidden']:
            raise NameError("ODENet's forward method should have arguments: 'input', 't' and 'hidden' (in that order)")
        if 'hidden_size' not in dir(ODENet):
            raise ValueError("ODENet should have attribute hidden_size")
        
        self.net = ODENet
        self.input_ode = torch.zeros(1,1)
        self.time_gaps = torch.zeros(1,1)
                
    def forward(self,t,hidden):
        output = self.net(self.input_ode,t.reshape(1,1)*self.time_gaps,hidden)*self.time_gaps
        return output

class CDENetBase(nn.Module):
    """CDENetBase
    Args:
        ....
    """
    def __init__(self,CDENet):
        super(CDENetBase, self).__init__()
        
        self.net = CDENet
        self.input_ode = torch.zeros(1,1)
        self.time_gaps = torch.zeros(1,1)

    def forward(self,t,hidden):
        output = self.net(t.reshape(1,1)*self.time_gaps,hidden) * self.time_gaps.unsqueeze(1)
        return torch.matmul(output,self.input_ode.unsqueeze(2)).squeeze(2)
    
class ODERNNBase(nn.Module):
    """Base class for continuous time recurrent neural network (RNN) (e.g. vanilla RNNs, Jump NNs, GRUs and LSTMs)
    
    Args:
        ODENet (nn.Module): The neural network
        UpdateNN (nn.Module): The neural network
    
    Structure of ODENet:
            output_size: dimension of output
            input_update_size: dimension of update features (can be larger than output)
            input_ode_size: dimension of features you wish to pass to ODENet
            hidden_size: dimension of hidden state

    Structure of UpdateNN:
            output_size: dimension of output
            input_update_size: dimension of update features (can be larger than output)
            input_ode_size: dimension of features you wish to pass to ODENet
            hidden_size: dimension of hidden state
            
    Return:
        Tensor
    """
    
    def __init__(self,UpdateNN,ODENet,output_size=1, device='cpu'):
        super(ODERNNBase,self).__init__()
        
        self.ODENet = ODENetBase(ODENet)
        self.updateNN = UpdateNNBase(UpdateNN)
        self.device=device
                
    def forward(self,input_update,h_0,times,input_ode=None):   
        """ 
        forward
        
        Args:
            times (): 2d t0 and t1
        """
        # discrete update/jump as new information receieved
        hidden = self.forward_update(input_update,h_0)
        # use ODENet to 'evolve' state to next timestep
        output = self.forward_ode(hidden,times,input_ode)
        return output
    
    def forward_update(self,input_update,h_0):
        """
        forward_update
        """
        output = self.updateNN(input_update,h_0)
        return output

    def forward_ode(self,hidden,times,input_ode=None):
        """
        forward_ode
        -----> use for predicting a trajectory
        """
        # enable input and time_gaps to be passed to ODENet.forward
        self.ODENet.input_ode = input_ode
        self.ODENet.time_gaps = times[:,1:2] - times[:,0:1]
        output = self.solve_ode(self.ODENet,hidden,torch.tensor([0,1.0]).to(self.device))[1]
        return output
    
    def solve_ode(self,vector_field,h_0,time):
        """
        solve_ode
        """
        # numerical integration until next time step
        output = odeint(vector_field, h_0, time)
        return output    

class LatentJumpODECell(ODERNNBase):
    """LatentJumpODECell
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        The return value. True for success, False otherwise.
    """
    def __init__(self,UpdateNN,ODENet,output_size=1,device='cpu'):
        ODERNNBase.__init__(self,UpdateNN,ODENet,output_size,device)
    
class ODERNNCell(ODERNNBase):
    """
    ODERNNCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu'):
        hidden_size = ODENet.hidden_size
        rnn = nn.RNNCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device)
        
class ODEGRUCell(ODERNNBase):
    """
    ODEGRUCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu'):
        hidden_size = ODENet.hidden_size
        rnn = nn.GRUCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device)
        
class ODELSTMCell(ODERNNBase):
    """
    ODELSTMCell
    """
    def __init__(self,ODENet,input_size_update,output_size=1,device='cpu'):
        hidden_size = ODENet.hidden_size
        rnn = nn.LSTMCell(input_size_update,hidden_size)
        ODERNNBase.__init__(self,rnn,ODENet,output_size,device)
        
    def forward_update(self,input_update,h_0):
        """
        forward_update
        """
        output = self.updateNN(input_update,h_0)
        return output
    
    def forward(self,input_update,h_0,times,input_ode=None):   
        """ 
        forward
        """
        if (type(h_0) != tuple):
            raise ValueError("h_0 should be a tuple of (hidden state, cell state)")
        # discrete update/jump as new information receieved
        hidden,cell = self.forward_update(input_update,h_0)
        # use ODENet to 'evolve' hidden state (but not cell state) to next timestep
        output = self.forward_ode(hidden,times,input_ode)
        return output,cell

    
class neuralCDECell(ODERNNBase):
    """neuralCDECell. Rectilinear.
    
    Args:
    
    Structure of CDENet:
            
    Return:
        Tensor
    """
    
    def __init__(self,CDENet,output_size=1, device='cpu'):
        super(ODERNNBase,self).__init__()
        
        self.ODENet = CDENetBase(CDENet)
        self.device = device
                
    def forward(self,h_0,times,input_t1,input_t0=None):   
        """ 
        forward
        
        """
        # calculate dX and dt
        if input_t0 == None:
            dinput = torch.zeros_like(input_t1)
        else:
            dinput = input_t1 - input_t0
        dt = times[:,1:2] - times[:,0:1]
        dt_z = torch.cat((dt,torch.zeros_like(dinput)),1)
        dinput_z = torch.cat((torch.zeros_like(dt),dinput),1)
        
        # use ODENet to update as new information recieved
        hidden = self.forward_update(dinput_z,h_0)
        # use ODENet to 'evolve' state to next timestep
        output = self.forward_ode(hidden,times,dinput_z)
        return output
    
    def forward_update(self,input_update,h_0):
        """
        forward_update
        """
        output = self.forward_ode(h_0,torch.tensor([0,1.0]).expand(input_update.size(0),2),input_update)
        return output
    
#     def forward_ode(self,hidden,times,input_ode=None):
#         """
#         forward_ode
#         -----> use for predicting a trajectory
#         """
#         # enable input and time_gaps to be passed to ODENet.forward
#         self.ODENet.input_ode = input_ode
#         self.ODENet.time_gaps = times[:,1:2] - times[:,0:1]
#         output = self.solve_ode(self.CDENet,hidden,torch.tensor([0,1.0]).to(self.device))[1]
#         return output
    