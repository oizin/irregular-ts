#!/bin/sh
python mimic.py --hidden_dim_t=10 --model=ODEGRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
python mimic.py --hidden_dim_t=10 --model=GRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
python mimic.py --hidden_dim_t=10 --model=FlowGRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
python mimic.py --hidden_dim_t=10 --model=DecayGRUModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
python mimic.py --hidden_dim_t=10 --model=CatboostModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
python mimic.py --hidden_dim_t=10 --model=ODELSTMModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
python mimic.py --hidden_dim_t=10 --model=FlowLSTMModel --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
python mimic.py --hidden_dim_t=10 --model=IMODE --max_epochs=100 --test=True --nfolds=3 --seed 3 --gpu 1 --logfolder results1
