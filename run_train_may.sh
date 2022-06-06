#!/bin/sh
python mimic.py --hidden_dim_t=12 --model=GRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2
python mimic.py --hidden_dim_t=12 --model=ODEGRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2
python mimic.py --hidden_dim_t=12 --model=FlowGRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2
python mimic.py --hidden_dim_t=12 --model=DecayGRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2
python mimic.py --hidden_dim_t=12 --model=CatboostModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2
python mimic.py --hidden_dim_t=12 --model=ODELSTMModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2
python mimic.py --hidden_dim_t=12 --model=FlowLSTMModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2
python mimic.py --hidden_dim_t=12 --model=IMODE --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results2