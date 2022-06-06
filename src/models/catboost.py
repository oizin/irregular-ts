## Catboost Model ##
from catboost import CatBoostRegressor

class CatboostModel(CatBoostRegressor):
    def __init__(self,niter,task,eval_fn):

        # match model/loss with task:
        if task == 'gaussian':
            loss = 'RMSEWithUncertainty'
        elif task == 'conditional_expectation':
            loss = 'RMSE'
        elif task == 'categorical':
            loss = 'RMSEWithUncertainty'
        super().__init__(iterations=niter, 
                          depth=3, 
                          learning_rate=1e-1, 
                          loss_function=loss,
                          od_type='IncToDec',
                          od_pval=0.1,
                          train_dir='experiments/mimic/catboost/')
        self.eval_fn = eval_fn

def catboost_feature_engineer(df,features,n_shift=5):
    df = df.copy()
    print("catboost feature engineering...")
    # time gap
    df.loc[:,'delta_t'] = df.loc[:,features['time_vars'][1]] - df.loc[:,features['time_vars'][0]]
    new_vars = ['delta_t']
    # previous valueWs
    for var in features['timevarying']:
        for n in range(1,n_shift+1):
            var_name = var + '_s' + str(n)
            df[var_name] = df.groupby(features['id'])[var].shift(n)
            new_vars.append(var_name)
    return df,new_vars