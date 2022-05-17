#!/bin/sh
python train.py --hidden_dim_t=12 --net=ctRNNModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --net=dtRNNModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --net=ctGRUModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --net=ODEGRUBayes --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --net=dtGRUModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --net=ctLSTMModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --net=dtLSTMModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --hidden_dim_i=4 --net=neuralJumpModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --hidden_dim_i=4 --net=resNeuralJumpModel --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1
python train.py --hidden_dim_t=12 --hidden_dim_i=4 --net=IMODE --max_epochs=100 --update_loss=0.1 --logfolder=model_results_19112021 --test=True --nfolds=1 --seed 1



python train.py --hidden_dim_t=12 --net=ctLSTMModel --max_epochs=5 --update_loss=0.1 --logfolder=model_results_24112021 --test=True --nfolds=1 --seed 1