import torch
import torch.nn as nn

DT_SCALER = 1/12

class FF1(nn.Module):
    """FF1: basic feedforward network (1)
    
    """
    def __init__(self,hidden_dim,feature_dim):
        super().__init__()
        
        self.hidden_size = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
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
            nn.Linear(hidden_dim+feature_dim, (hidden_dim+feature_dim)*1),
            nn.Dropout(p=0.2),
            nn.Tanh(),
            nn.Linear((hidden_dim+feature_dim)*1, hidden_dim)
        )
                
    def forward(self,input,hidden):
        output = self.net(torch.cat((input,hidden),1))
        return output