import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianOutputNNBase(nn.Module):
    """GaussianOutputNN
    (batch_size,:,features) -> (batch_size,:,2)
    (batch_size,features) -> (batch_size,2)
    """
    def __init__(self,hidden_dim):
        super().__init__()
        self.output_dim = 2
        # self.ginv=ginv
        # self.g=g
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sigma_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self,x):
        """
        x of dimension ()
        """
        mu_out = self.mu_net(x)
        sigma_out = torch.exp(self.sigma_net(x))
        return torch.cat((mu_out,sigma_out),1)
    

    # def probabilistic_eval_fn(self,pred,y,alpha=0.05):
    #     m,s = pred[:,0],pred[:,1]
    #     alpha_q = scipy.stats.norm.ppf(1-alpha/2)
    #     lower = m - alpha_q*s
    #     upper = m + alpha_q*s
    #     pit_epoch = scipy.stats.norm(m, s).cdf(y)
    #     var_pit = np.var(pit_epoch)
    #     crps = ps.crps_gaussian(y, mu=m, sig=s)
    #     crps_mean = np.mean(crps)
    #     ig = scipy.stats.norm.logpdf(y,loc=m, scale=s)
    #     ig_mean = np.mean(ig)
    #     upper = self.ginv(upper)
    #     lower = self.ginv(lower)
    #     yinv = self.ginv(y)
    #     int_score = (upper - lower) + 2/alpha*(lower - yinv)*(yinv < lower) + 2/alpha*(yinv - upper)*(yinv > upper)
    #     int_score_mean = np.mean(int_score)
    #     int_coverage = sum((lower < yinv) & (upper > yinv))/yinv.shape[0]
    #     int_av_width = np.mean(upper - lower)
    #     int_med_width = np.median(upper - lower)
    #     return {'crps_mean':crps_mean,
    #             'ig_mean':ig_mean,
    #             'int_score_mean':int_score_mean,
    #             'var_pit':var_pit,
    #             'int_coverage':int_coverage,
    #             'int_av_width':int_av_width,
    #             'int_med_width':int_med_width}

class GaussianOutputNNKL(GaussianOutputNNBase):
    """GaussianOutputNN
    (batch_size,:,features) -> (batch_size,:,2)
    (batch_size,features) -> (batch_size,2)
    """
    
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
        
    def loss_update_fn(self,pred,y,e):
        """
        (batch_size,vals) -> (?)
        """
        m,s = pred[:,0],pred[:,1]
        # update distribution
        distribution_post = torch.distributions.normal.Normal(m, s)
        likelihood_post = distribution_post.log_prob(y)
        # observation distribution
        distribution_obs = torch.distributions.normal.Normal(y, e)
        likelihood_obs = distribution_obs.log_prob(y)
        kl = F.kl_div(likelihood_post,likelihood_obs,reduction="none",log_target=True).sum()
        return kl

class GaussianOutputNNLL(GaussianOutputNNBase):
    """GaussianOutputNN
    (batch_size,:,features) -> (batch_size,:,2)
    (batch_size,features) -> (batch_size,2)
    """
    
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
        
    def loss_update_fn(self,pred,y,e):
        """
        (batch_size,vals) -> (?)
        """
        # log probs
        m,s = pred[:,0],pred[:,1]
        distribution = torch.distributions.normal.Normal(m, s + 1e-5)
        likelihood = distribution.log_prob(y)
        llik = torch.sum(likelihood)
        return -llik

class BinnedOutputNN(nn.Module):
    """BinnedOutputNN
    
    """
    def __init__(self):
        raise AttributeError('Not implemented')

class ConditionalExpectNN(nn.Module):
    """conditionalExpectNN
    E(y|x)
    (batch_size,:,features) -> (batch_size,:,2)
    (batch_size,features) -> (batch_size,2)
    """
    def __init__(self,hidden_dim,g=lambda a : a,ginv=lambda a : a):
        super().__init__()
        self.output_dim = 1
        self.ginv=ginv
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.Tanh(),
            #nn.Dropout(p=0.1),
            nn.Linear(hidden_dim//2, 1),
        )
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(self,x):
        """
        x of dimension ()
        """
        mu_out = self.mu_net(x)
        return mu_out
    
    def loss_fn(self,pred,y):
        """
        (batch_size,vals) -> (?)
        """
        mse = self.mse(pred.squeeze(1),y)
        return mse
    
    def loss_update_fn(self,pred,y,e=0.0):
        """
        (batch_size,vals) -> (?)
        """
        mse = self.mse(pred.squeeze(1),y)
        return mse
    
    def sse_fn(self,pred,y):
        """
        Method for calculation of the sum of squared errors
        """
        #mse = self.mse(pred.squeeze(1),y)
#         print(self.ginv(pred[:,0].cpu().numpy()))
#         print(self.ginv(y.cpu().numpy()))
        mse = torch.tensor(np.sum((self.ginv(pred[:,0].cpu().numpy()) - self.ginv(y.cpu().numpy()))**2))
        return mse
        
    def probabilistic_eval_fn(self,pred,y,alpha=0.05):
        return {'crps_mean':np.NaN,
                'ig_mean':np.NaN,
                'int_score_mean':np.NaN,
                'var_pit':np.NaN,
                'int_coverage':np.NaN,
                'int_av_width':np.NaN,
                'int_med_width':np.NaN}
