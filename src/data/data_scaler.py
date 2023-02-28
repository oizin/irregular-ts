# module to scale the data
# and fill in missings
import numpy as np
import pandas as pd

def glc_transform(x):
    x = x.to_numpy().copy()
    x[x > 0] = np.log(x[x > 0]) - np.log(140)
    return x

class PreProcessMIMIC():

    def __init__(self,features,scaler,features_autoreg_scaler=glc_transform):
        """"
        Feature scaling
        """
        # feature sets
        self.features_autoreg = [features['target']] + [features['timevarying'][0]]
        self.features_divide = list(features['features_divider'].keys())
        vars = list(features['timevarying']) + list(features['static']) + list(features['intervention'])
        vars = list(set(vars)) # unique
        self.features_scaler = [v for v in vars if v not in self.features_divide + self.features_autoreg]

        # scalers
        self.scaler = scaler
        self.dividers = features['features_divider']
        self.features_autoreg_scaler = features_autoreg_scaler
        
    def fit(self,df):
        if len(self.features_scaler) > 0:
            self.scaler.fit(df.loc[:,self.features_scaler])
        
    def transform(self,df):
        df = df.copy()
        # target / lagged target
        for feature in self.features_autoreg:
            df.loc[:,feature] = self.features_autoreg_scaler(df[feature])
        # treatments / things to be divided
        for feature in self.features_divide:
            df.loc[:,feature] = df.loc[:,feature] / self.dividers[feature]
        # remaining variables
        if len(self.features_scaler) > 0:
            df.loc[:,self.features_scaler] = self.scaler.transform(df[self.features_scaler])
            df.loc[:,self.features_scaler] = df.loc[:,self.features_scaler].fillna(value=0.0)
        
        return df
    
class PreProcessSim():
    """
    args:
        inputs: inputs
    """
    def __init__(self,inputs,outputs,scaler_inputs,scaler_outputs):
        self.inputs = inputs
        self.outputs = outputs
#         self.scaler_inputs = scaler_inputs
#         self.scaler_outputs = scaler_outputs
        
    def fit(self,df):
        return None
#         self.scaler_inputs.fit(df.loc[:,self.inputs])
#         self.scaler_outputs.fit(df.loc[:,self.outputs])

    def transform(self,df):
#         df = df.copy()
#         df.loc[:,self.inputs] = self.scaler_inputs.transform(df[self.inputs].copy(deep=True))
#         df = df.copy()
#         df.loc[:,self.outputs] = self.scaler_outputs.transform(df[self.outputs].copy(deep=True))
        # glucose
        # df = df.copy()
        # df.loc[:,'x'] = glc_transform(df['x'].to_numpy())
        # df = df.copy()
        # df.loc[:,'y'] = glc_transform(df['y'].to_numpy())
        # df.loc[:,'g'] = df.loc[:,'g'] / 2.0
        # df.loc[:,'m'] = df.loc[:,'m']
        return df