## Linear Model ##
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np

class LinearModel():
    def __init__(self,task,eval_fn):

        self.task = task
        self.eval_fn = eval_fn

    def fit(self,X,y):
        self.imp = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0.0)
        self.imp.fit(X)
        if self.task == 'gaussian':
            self.model_mu = LinearRegression()
            self.model_sd = LinearRegression()
            X = self.imp.transform(X)
            self.model_mu.fit(X,y)
            y_pred = self.model_mu.predict(X)
            self.model_sd.fit(X,(y - y_pred)**2 + 1e-2)
        elif self.task == 'conditional_expectation':
            X = self.imp.transform(X)
            self.model_mu = LinearRegression()
            self.model_mu.fit(X,y)

    def predict(self,X):
        if self.task == 'gaussian':
            X = self.imp.transform(X)
            mu = self.model_mu.predict(X)
            sd = np.sqrt(self.model_sd.predict(X))
            mu = mu.reshape(mu.shape[0],1)
            sd = sd.reshape(sd.shape[0],1)
            out = np.concatenate((mu,sd),1)
            return(out)
        elif self.task == 'conditional_expectation':
            mu = self.model_mu.predict(X)
            return(mu)

def lm_feature_engineer(df,features,n_shift=5):
    df = df.copy()
    print("linear model feature engineering...")
    # time gap
    df.loc[:,'delta_t'] = df.loc[:,features['time_vars'][1]] - df.loc[:,features['time_vars'][0]]
    new_vars = ['delta_t']
    # previous values
    for var in features['timevarying']:
        for n in range(1,n_shift+1):
            var_name = var + '_s' + str(n)
            df[var_name] = df.groupby(features['id'])[var].shift(n)
            new_vars.append(var_name)
    return df,new_vars