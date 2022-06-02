# Catboost -----------------------------------------------------------------------------------
from catboost import Pool, CatBoostRegressor
from ..metrics.metrics import probabilistic_eval_fn,sse_fn

class CatboostModel(CatBoostRegressor):
    def __init__(self):
        super().__init__(iterations=10000, 
                          depth=3, 
                          learning_rate=1e-1, 
                          loss_function='RMSEWithUncertainty',
                          od_type='IncToDec',
                          od_pval=0.1,
                          train_dir='experiments/mimic/catboost/')

    def eval_fn(self,pred,y,ginv):
        return probabilistic_eval_fn(pred,y,ginv)

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