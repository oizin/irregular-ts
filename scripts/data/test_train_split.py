import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_loader import import_data
from src.utils import seed_everything

# seed
seed_everything(1)

DATA_PATH = "data/analysis.csv"
df = import_data(DATA_PATH)
print(df.shape)

train_ids, test_ids = train_test_split(df.icustay_id.unique(),test_size=0.25)

# variable scaling - fit, transform
df_train = df.loc[df.icustay_id.isin(train_ids)].copy(deep=True)
df_train.sort_values(by=['icustay_id','timer'],inplace=True)
df_train.reset_index(drop=True,inplace=True)

# variable scaling - transform
df_test = df.loc[df.icustay_id.isin(test_ids)].copy(deep=True)
df_test.sort_values(by=['icustay_id','timer'],inplace=True)
df_test.reset_index(drop=True,inplace=True)

# save
df_train.to_csv("data/train.csv",index=False)
df_test.to_csv("data/test.csv",index=False)