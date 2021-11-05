import torch
import torch.nn as nn

# need to add time gap to all! and check still works

class ODENetH(nn.Module):
    """FF1: basic feedforward network (1)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim*2,60)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim*2,60), hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self,input,t,dt,hidden):
        output = self.net(hidden)
        return output

class ODENetHIT(nn.Module):
    """FF1: basic feedforward network (1)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim+feature_dim+1, max((hidden_dim+feature_dim+1)*2,60)),
            nn.Tanh(),
            nn.Linear(max((hidden_dim+feature_dim+1)*2,60), hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self,input,t,dt,hidden):
        output = self.net(torch.cat((input,t,hidden),1))
        return output
    
class ODENetHITT(nn.Module):
    """FF1: basic feedforward network (1)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim+feature_dim+2, max((hidden_dim+feature_dim+2)*2,60)),
            nn.Tanh(),
            nn.Linear(max((hidden_dim+feature_dim+2)*2,60), hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self,input,t,dt,hidden):
        output = self.net(torch.cat((input,t,dt,hidden),1))
        return output

class ODENetHI(nn.Module):
    """FF1: basic feedforward network (1)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim+feature_dim, max((hidden_dim+feature_dim)*2,60)),
            nn.Tanh(),
            nn.Linear(max((hidden_dim+feature_dim)*2,60), hidden_dim),
            nn.Tanh(),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
                
    def forward(self,input,t,dt,hidden):
        output = self.net(torch.cat((input,hidden),1))
        return output

class GRUNet(nn.Module):
    """GRUNet: A GRU
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.GRUCell(feature_dim, hidden_dim)
        
        nn.init.normal_(self.net.weight_ih, mean=0, std=0.1)
        nn.init.normal_(self.net.weight_hh, mean=0, std=0.1)
        nn.init.constant_(self.net.bias_ih , val=0)
        nn.init.constant_(self.net.bias_hh , val=0)
        
    def forward(self,input,t,dt,hidden):
        output = self.net(input,hidden)
        return output
    
class IMODE_ODENet(nn.Module):
    def __init__(self,hidden_dims,feature_dims):
        super().__init__()
        
        # dimensions
        hx_dim,hi_dim = hidden_dims['hidden_dim_x'],hidden_dims['hidden_dim_i']
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

    def forward(self,input,t,dt,hidden):
        h_i = hidden[:,(self.hx_dim):(self.hx_dim+self.hi_dim)]
        h_x = self.hx_net(hidden)
        h_i = self.hi_net(h_i)
        h_all = torch.cat((h_x,h_i),1)
        return h_all