
# other imports
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """BaseModel
    
    loss function a property of output NN
    """
    def __init__(self,RNN,OutputNN):
        superXXXYXYXYXYXYXY
        
        self.RNN = RNN
        self.OutputNN = OutputNN
        
    def forward(self, dt, x, p=0.0):
        
        T = x.size(1)
        batch_size = x.size(0)
        output = torch.zeros(batch_size,T,self.OutputNN.output_dim,device = self.device)
        h_t = torch.zeros(batch_size, self.rnn.hidden_size,device=self.device)
        for i in range(0,T):
            x_i = x[:,i:(i+1),:]
            dt_i = (dt[:,i,:][:,1] - dt[:,i,:][:,0]).unsqueeze(1)
            h_t = self.RNN(x_i.squeeze(1),h_t,dt_i)
            h_t = F.dropout(h_t,training=training,p=p)
            output[i] = self.OutputNN(h_t)
        return output
    
#     def forward_trajectory(self, dt, x, p=0.0): # NEEDS TO BE CODED!
        
#         T = x.size(1)
#         outputs = []
#         # ODE
#         for i in range(0,T):
#             t0 = t[0,i,0]
#             t1 = t[0,i,1]
#             ts = torch.linspace(t0,t1,nsteps)
#             y0 = x[:,i:(i+1),:]
#             h_t = self.RNN.forward(...,n_intermediate=10)
#             outputs_i = self.OutputNN(h_t)
#             outputs.append(outputs_i)
#         return outputs
    
    def train_step(self):
        2 + 3
        
    def validation_step(self):
        2 + 3
        
    def test_step(self):
        2 + 3

class GaussianOutputNN(nn.Module):
    """GaussianOutputNN
    
    """
    
    def __init__(self):
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(hidden_dim,hidden_dim//2)
        self.distribution_mu = nn.Linear(hidden_dim//2, 1)
        self.relu = nn.ReLU()
        self.sigma_net = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus(),
        ).to(device)
    
    def forward(self,x):
        mu = self.l1(h_t)
        mu = F.dropout(mu,training=training,p=p)
        mu = self.relu(mu)
        mu_out[:,i:(i+1),:] = self.distribution_mu(mu).unsqueeze(1)
        sigma_out[:,i:(i+1),:] = self.sigma_net(h_t.squeeze(1)).unsqueeze(1)
        return torch.cat((mu_out,sigma_out),1)
    
    def loss_fn(self,preds,y,msk):
        # extract
        m_, s_ = preds[0]

        # log probs
        distribution = torch.distributions.normal.Normal(m_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk].unsqueeze(1))

        llik = torch.sum(likelihood)
        return -llik

    def eval_fn(self,preds,y,msk):
        # extract
        m_, s_ = preds
                
        # log probs
        distribution = torch.distributions.normal.Normal(m_[msk], s_[msk])
        likelihood = distribution.log_prob(y[msk].unsqueeze(1))
        
        llik = torch.sum(likelihood)
        return -llik

# class BinnedOutputNN