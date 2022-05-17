# module to scale the data
# and fill in missings
import numpy as np
import pandas as pd

def glc_transform(x):
    x = x.copy()
    x[x > 0] = np.log(x[x > 0]) - np.log(140)
    return x

class PreProcess():
    """
    args:
        inputs: a dictionary
    """
    def __init__(self,inputs,scaler):
        self.inputs = inputs['timevarying'] + inputs['static']
        tmp = self.inputs.copy()
        tmp.remove('glc')
        self.oth_inputs = tmp
        self.scaler = scaler
        
    def fit(self,df):
        self.scaler.fit(df.loc[:,self.oth_inputs])
        
    def transform(self,df):
        # glucose
        df = df.copy()
        df.loc[:,'glc'] = glc_transform(df['glc'].to_numpy())
        df = df.copy()
        df.loc[:,'glc_dt'] = glc_transform(df['glc_dt'].to_numpy())
        # remaining variables
        df = df.copy()
        df.loc[:,self.oth_inputs] = self.scaler.transform(df[self.oth_inputs].copy(deep=True))
        df = df.copy()
        df.loc[:,self.oth_inputs] = df.loc[:,self.oth_inputs].fillna(value=0.)
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
        df = df.copy()
        df.loc[:,'x'] = glc_transform(df['x'].to_numpy())
        df = df.copy()
        df.loc[:,'y'] = glc_transform(df['y'].to_numpy())
        df.loc[:,'g'] = df.loc[:,'g'] / 2.0
        df.loc[:,'m'] = df.loc[:,'m']
        return df