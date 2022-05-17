import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .data_scaler import PreProcess,PreProcessSim
from sklearn.preprocessing import QuantileTransformer,StandardScaler

import pytorch_lightning as pl

def import_data(path,verbose=True):
    df = pd.read_csv(path)
    ids = df.icustay_id.unique()
    for id_ in ids:
        df_id = df.loc[df.icustay_id == id_,:]
        if (sum(df_id.msk) == df_id.shape[0]):
            df.drop(df.loc[df.icustay_id == id_,:].index,inplace=True)
            if verbose:
                print("excluding:",id_)
    return df

TIME_VARS = ["timer","timer_dt"]

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    '''
    ## get sequence lengths
    lengths = torch.tensor([b[0].shape[0] for b in batch])
    ## padding
    xt = [torch.Tensor(b[0]) for b in batch]
    x0 = [torch.Tensor(b[1]) for b in batch]
    xi = [torch.Tensor(b[2]) for b in batch]
    y = [torch.Tensor(b[3]) for b in batch]
    msk = [torch.Tensor(b[4]) for b in batch]
    dt = [torch.Tensor(b[5]) for b in batch]
    msk0 = [torch.Tensor(b[6]) for b in batch]
    id = [b[7] for b in batch]
    xt = torch.nn.utils.rnn.pad_sequence(xt,batch_first=True)
    x0 = torch.nn.utils.rnn.pad_sequence(x0,batch_first=True)
    xi = torch.nn.utils.rnn.pad_sequence(xi,batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(y,batch_first=True)
    msk = torch.nn.utils.rnn.pad_sequence(msk,batch_first=True,padding_value=int(1))
    dt = torch.nn.utils.rnn.pad_sequence(dt,batch_first=True)
    msk0 = torch.nn.utils.rnn.pad_sequence(msk0,batch_first=True,padding_value=int(1))
    return xt,x0,xi,y,msk,dt,msk0,id

TIME_VARS = ["timer","timer_dt"]

class MIMICDataset(Dataset):
    """
    Args:
        patientunitstayids: 
        df:
        ...
    
    Example:
    """
    def __init__(self,df,features,pad=-1,verbose=True):
        self.pad = pad
        #self.maxrows = maxrows
        self.Xt,self.X0,self.Xi,self.y,self.msk,self.dt,self.msk0,self.seqlen,self.id = self.load_data(df,features,verbose=verbose)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # pad
        Xt = self.Xt[idx].astype(np.float32)
        X0 = self.X0[idx].astype(np.float32)
        Xi = self.Xi[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        msk = self.msk[idx].astype(np.int32)
        dt = self.dt[idx].astype(np.float32)
        msk0 = self.msk0[idx].astype(np.int32)
        seqlen = self.seqlen[idx]
        id = self.id[idx].astype(np.int32)

        return Xt,X0,Xi,y,msk,dt,msk0,id
    
    def load_data(self,df,features,verbose):
        """
        features a dict
        """
        excl = []
        n_excl_pt = 0
        n_excl_rws = 0
        Xt_list,X0_list,Xi_list, y_list, msk_list, dt_list, msk0_list, seqlen_list,id_list = [], [], [], [], [], [],[],[],[]
        ids = df.icustay_id.unique()
        if verbose:
            print("reconfiguring data...")
        for id_ in ids:
            if self.pad == -1:
                df_id = df.loc[df.icustay_id == id_,:]
            else:
                df_id = df.loc[df.icustay_id == id_,:].iloc[0:self.pad]
            Xt = df_id.loc[:,features['timevarying']]
            X0 = df_id.loc[:,features['static']]
            Xi = df_id.loc[:,features['intervention']]
            X0 = X0.iloc[0,:]
            y = df_id.loc[:,"glc_dt"]
            msk = df_id.loc[:,"msk"]
            dt = df_id.loc[:,TIME_VARS]
            msk0 = df_id.loc[:,"msk0"]
            seqlen = df_id.shape[0]
            Xt = np.array(Xt).astype(np.float32)
            Xi = np.array(Xi).astype(np.float32)
            X0 = np.array(X0).astype(np.float32)
            y = np.array(y).astype(np.float32)
            msk = np.array(msk).astype(np.int32)
            dt = np.array(dt).astype(np.float32)
            msk0 = np.array(msk0).astype(np.int32)
            id_ = np.array(id_).astype(np.int32)
            Xt_list.append(Xt)
            Xi_list.append(Xi)
            X0_list.append(X0)
            y_list.append(y)
            msk_list.append(msk)
            dt_list.append(dt)
            msk0_list.append(msk0)
            seqlen_list.append(seqlen)
            id_list.append(id_)
        if verbose == True:
            print("excluded patients:",n_excl_pt)
            print("excluded rows:",n_excl_rws)
        return Xt_list,X0_list,Xi_list,y_list,msk_list,dt_list,msk0_list,seqlen_list,id_list
    

class MIMIC3DataModule(pl.LightningDataModule):
    def __init__(self, features,df_train,df_test, batch_size = 64, max_length=100, testing = False, verbose = False):
        """
        features a dict
        """
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.batch_size = batch_size
        self.max_length = max_length
        self.testing = testing
        self.verbose = verbose
        self.features = features

    def setup(self, stage=None):
        df_train = self.df_train
        df_test = self.df_test

        # train-validation split
        train_ids, valid_ids = train_test_split(df_train.icustay_id.unique(),test_size=0.1)
        df_valid = df_train.loc[df_train.icustay_id.isin(valid_ids)].copy(deep=True)
        df_train = df_train.loc[df_train.icustay_id.isin(train_ids)].copy(deep=True)

        # preprocess
        preproc = PreProcess(self.features,QuantileTransformer())
        preproc.fit(df_train)
        df_train = preproc.transform(df_train)
        df_valid = preproc.transform(df_valid)
        df_test = preproc.transform(df_test)
        self.data_train = MIMICDataset(df_train,self.features,pad=self.max_length,verbose=self.verbose)
        self.data_valid = MIMICDataset(df_valid,self.features,verbose=self.verbose)
        self.data_test = MIMICDataset(df_test,self.features,verbose=self.verbose)
        
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,collate_fn=collate_fn_padd,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_valid, batch_size=self.batch_size,collate_fn=collate_fn_padd,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,collate_fn=collate_fn_padd,num_workers=4)
    
    
# Simulations -----------------------------------------------------------------------------------------------
    
class SimulationsDataset(Dataset):
    def __init__(self,df,features):
        self.features = features
        self.Xt,self.X0,self.Xi,self.y,self.msk,self.dt,self.msk0,self.id = self.load_data(df)
        """SimulationsDataset
        
        Args
            df: 
        """
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        Xt = self.Xt[idx].astype(np.float32)
        X0 = self.X0[idx].astype(np.float32)
        Xi = self.Xi[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        msk = self.msk[idx].astype(np.int32)
        dt = self.dt[idx].astype(np.float32)*0.1  # scale time
        msk0 = self.msk0[idx].astype(np.int32)
        id = self.id[idx].astype(np.int32)

        return Xt,X0,Xi,y,msk,dt,msk0,id
    
    def load_data(self,df):
        """
        features a dict
        """
        Xt_list,X0_list,Xi_list,y_list, msk_list, dt_list, msk0_list,id_list = [],[], [], [], [], [], [], []
        ids = df.id.unique()
        for id_ in ids:
            df_id = df.loc[df.id == id_,:]
            Xt = df_id.loc[:,self.features['timevarying']]
            Xi = df_id.loc[:,self.features['intervention']]
            y = df_id.loc[:,'y']
            msk = df_id.loc[:,'msk']
            dt = df_id.loc[:,['t0','t1']]
            msk0 = df_id.loc[:,'msk0']
            Xt = np.array(Xt).astype(np.float32)
            Xi = np.array(Xi).astype(np.float32)
            X0 = np.zeros_like(Xt).astype(np.float32)
            y = np.array(y).astype(np.float32)
            msk = np.array(msk).astype(np.int32)
            dt = np.array(dt).astype(np.float32)
            msk0 = np.array(msk0).astype(np.int32)
            id_ = np.array(id_).astype(np.int32)
            Xt_list.append(Xt)
            Xi_list.append(Xi)
            X0_list.append(X0)
            y_list.append(y)
            msk_list.append(msk)
            dt_list.append(dt)
            msk0_list.append(msk0)
            id_list.append(id_)
            #seqlen_list.append(seqlen)
        return Xt_list,X0_list,Xi_list,y_list,msk_list,dt_list,msk0_list,id_list

class SimulationsDataModule(pl.LightningDataModule):
    def __init__(self,features,data_dir = "data/simulation.csv", batch_size = 64, verbose = False):
        """SimulationsDataModule
        
        Args:
            data_dir:
            batch_size:
            testing:
            verbose:
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.verbose = verbose
        self.preproc = None
        self.features = features

    def setup(self, stage=None):
        df = pd.read_csv(self.data_dir)
        
        # filter to "observed data"
        df = df.loc[df.obs == 1,:].copy()
        df.sort_values(by=['id','t'],inplace=True)
        df.reset_index(drop=True,inplace=True)
        df.drop(columns=["obs"],inplace=True)
        
        # reshape timestamp -> t0,t1 and value -> y,x
        df.rename(columns={'x':'y'},inplace=True)
        df['x'] = df.groupby('id')['y'].shift()
        df['t0'] = df.groupby('id')['t'].shift()
        df.rename(columns={'t':'t1'},inplace=True)
        
        # drop first
        df = df.loc[~df.x.isnull(),:]
        
        # msk where y is NaN
        df['msk'] =  df.y.isnull()
        df['msk0'] =  df.x.isnull()

        # train-test split
        train_ids, test_ids = train_test_split(df.id.unique(),test_size=0.2)
        df_test = df.loc[df.id.isin(test_ids)].copy(deep=True)
        df_train = df.loc[df.id.isin(train_ids)].copy(deep=True)

        # train-validation split
        train_ids, valid_ids = train_test_split(df_train.id.unique(),test_size=0.1)
        df_valid = df_train.loc[df_train.id.isin(valid_ids)].copy(deep=True)
        df_train = df_train.loc[df_train.id.isin(train_ids)].copy(deep=True)
        
        # save IDs
        self.train_ids = train_ids
        self.valid_ids = valid_ids
        self.test_ids  = test_ids

        # preprocess
        self.preproc = PreProcessSim(['x'],['y'],QuantileTransformer(),QuantileTransformer())
        self.preproc.fit(df_train)
        df_train = self.preproc.transform(df_train)
        df_valid = self.preproc.transform(df_valid)
        df_test = self.preproc.transform(df_test)
        
        # Dataloaders
        self.data_train = SimulationsDataset(df_train,self.features)
        self.data_valid = SimulationsDataset(df_valid,self.features)
        self.data_test = SimulationsDataset(df_test,self.features)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size,collate_fn=collate_fn_padd,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_valid, batch_size=self.batch_size,collate_fn=collate_fn_padd,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,collate_fn=collate_fn_padd,num_workers=4)