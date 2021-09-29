import torch
import torch.nn as nn
import numpy as np
import scipy
import properscoring as ps 

def ginv(x):
    x = x.copy()
    x = np.exp(x + np.log(140))
    return x

class GaussianOutputNN(nn.Module):
    """GaussianOutputNN
    (batch_size,:,features) -> (batch_size,:,2)
    (batch_size,features) -> (batch_size,2)
    """
    def __init__(self,hidden_dim):
        super().__init__()
        self.output_dim = 2
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus(),
        )
    
    def forward(self,x):
        """
        x of dimension ()
        """
        mu_out = self.mu_net(x)
        sigma_out = self.sigma_net(x)
        return torch.cat((mu_out,sigma_out),1)
    
    def loss_fn(self,pred,y):
        """
        (batch_size,vals) -> (?)
        """
        # log probs
        m,s = pred[:,0],pred[:,1]
        distribution = torch.distributions.normal.Normal(m, s)
        likelihood = distribution.log_prob(y)
        
        llik = torch.sum(likelihood)
        return -llik
    
    def sse_fn(self,pred,y):
        """
        Method for calculation of the sum of squared errors
        """
        # log probs
        m,s = pred[:,0],pred[:,1]
        c = torch.log(torch.tensor(140.0))
        mse = torch.sum((torch.exp(m + c) - torch.exp(y + c))**2)
        return mse
    
    def probabilistic_eval_fn(self,pred,y,alpha=0.05):
        m,s = pred[:,0],pred[:,1]
        alpha_q = scipy.stats.norm.ppf(1-alpha/2)
        lower = m - alpha_q*s
        upper = m + alpha_q*s
        pit_epoch = scipy.stats.norm(m, s).cdf(y)
        var_pit = np.var(pit_epoch)
        crps = ps.crps_gaussian(y, mu=m, sig=s)
        crps_mean = np.mean(crps)
        ig = scipy.stats.norm.logpdf(y,loc=m, scale=s)
        ig_mean = np.mean(ig)
        int_score = (upper - lower) + 2/alpha*(lower - y)*(y < lower) + 2/alpha*(y - upper)*(y > upper)
        int_score_mean = np.mean(int_score)
        int_coverage = sum((lower < y) & (upper > y))/y.shape[0]
        int_av_width = np.mean(ginv(upper) - ginv(lower))
        int_med_width = np.median(ginv(upper) - ginv(lower))
        return {'crps_mean':crps_mean,
                'ig_mean':ig_mean,
                'int_score_mean':int_score_mean,
                'var_pit':var_pit,
                'int_coverage':int_coverage,
                'int_av_width':int_av_width,
                'int_med_width':int_med_width}

class BinnedOutputNN(nn.Module):
    """BinnedOutputNN
    
    """
    def __init__(self):
        raise AttributeError('Not implemented')
