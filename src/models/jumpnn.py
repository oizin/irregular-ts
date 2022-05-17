import torch
import torch.nn as nn
    
class JumpNet1(nn.Module):
    """FF2: basic feedforward network (2)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, (feature_dim)*2),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear((feature_dim)*2, hidden_dim)
        )
                
    def forward(self,input,hidden):
        output = self.net(input)
        return output

class JumpNet2(nn.Module):
    """ResNet
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, (feature_dim)*2),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear((feature_dim)*2, hidden_dim)
        )
                
    def forward(self,input,hidden):
        output = hidden + self.net(input)
        return output
    
class IMODE_JumpNN(nn.Module):
    def __init__(self,hidden_dims,feature_dims):
        super().__init__()
        
        # dimensions
        hx_dim,hi_dim = hidden_dims['hidden_dim_t'],hidden_dims['hidden_dim_i']
        self.hx_dim = hx_dim
        self.hi_dim = hi_dim
        x_input, i_input = feature_dims['input_dim_t'],feature_dims['input_dim_i']
        self.x_input = x_input
        self.i_input = i_input
        
        # neural nets
        self.hx_net = nn.Sequential(
            nn.Linear(hx_dim + x_input, (hx_dim + x_input)*2),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear((hx_dim + x_input)*2, hx_dim),
            nn.Tanh(),
        )
        self.hi_net = nn.Sequential(
            nn.Linear(hi_dim + hx_dim + i_input, (hi_dim + hx_dim + i_input)*2),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear((hi_dim + hx_dim + i_input)*2, hi_dim),
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
 