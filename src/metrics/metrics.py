import numpy as np
import scipy
import properscoring as ps 

## P(Y|X) where P is a Guassian evaluation function ##
def gaussian_eval_fn(pred,y,ginv = lambda x: x,alpha=0.05):
    """
    Evaluate a probabilistic prediction across multiple metrics.
    Assumes the predicted distribution is Gaussian.

    Args:
        pred: the predicted mean and standard deviation - a 2D numpy array
        y: the observed outcome - a 1D numpy array
        ginv: a transform to apply to y (and any quantiles constructed from the predictions) prior to evaluation
    """
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
    upper = ginv(upper)
    lower = ginv(lower)
    yinv = ginv(y)
    int_score = (upper - lower) + 2/alpha*(lower - yinv)*(yinv < lower) + 2/alpha*(yinv - upper)*(yinv > upper)
    int_score_mean = np.mean(int_score)
    int_coverage = sum((lower < yinv) & (upper > yinv))/yinv.shape[0]
    int_av_width = np.mean(upper - lower)
    int_med_width = np.median(upper - lower)
    rmse = np.sqrt(np.mean((ginv(m) - ginv(y))**2))
    return {'crps_mean':crps_mean,
            'ig_mean':ig_mean,
            'int_score_mean':int_score_mean,
            'var_pit':var_pit,
            'int_coverage':int_coverage,
            'int_av_width':int_av_width,
            'int_med_width':int_med_width,
            'rmse':rmse}

## E(Y|X) evaluation function ##
def conditional_eval_fn(pred,y,ginv = lambda x: x,alpha=0.05):
    """
    Evaluate a conditional expectation prediction across multiple metrics.

    Args:
        pred: the predicted value - a 1D numpy array
        y: the observed outcome - a 1D numpy array
        ginv: a transform to apply to y (and any quantiles constructed from the predictions) prior to evaluation
    """
    m = pred
    rmse = np.sqrt(np.mean((ginv(m) - ginv(y))**2))
    return {'rmse':rmse}

# def sse_fn(pred,y,ginv=lambda x: x):
#     """
#     Method for calculation of the sum of squared errors
#     """
#     # log probs
#     m,s = pred[:,0],pred[:,1]
#     mse = torch.tensor(np.sum((ginv(m.cpu().numpy()) - ginv(y.cpu().numpy()))**2))
#     return mse

