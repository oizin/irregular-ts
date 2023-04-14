#!/bin/bash

models=( FlowGRUModel GRUModel FlowLSTMModel LSTMModel )

hidden=( 2 8 16 24 32 48 64 96 128 256 512 1024 )


for h in "${hidden[@]}"
do
    for m in "${models[@]}"
    do
    echo "hidden dim $h\n"
    echo "$m\n"
    python train.py --model $m --max_epochs 100 --hidden_dim_t $h --hidden_dim_i 4 --logfolder mimic/hyperparameters/hidden/h${h} --data mimic_val --nfolds 1
    done
done

models=( ODEGRUModel DecayGRUModel ODELSTMModel )

hidden=( 2 8 16 24 32 )

for h in "${hidden[@]}"
do
    for m in "${models[@]}"
    do
    echo "hidden dim $h\n"
    echo "$m\n"
    python train.py --model $m --max_epochs 100 --hidden_dim_t $h --hidden_dim_i 4 --logfolder mimic/hyperparameters/hidden/h${h} --data mimic_val --nfolds 1
    done
done


mixing=( 0.0000001 0.000001 0.00001 0.0001 0.001 0.01 0.1 1.0 10.0 )

models=( FlowGRUModel FlowLSTMModel )

for l in "${mixing[@]}"
do
    for m in "${models[@]}"
    do
    echo "mixing $l\n"
    echo "$m\n"
    python train.py --model $m --max_epochs 100 --hidden_dim_t 8 --hidden_dim_i 4 --logfolder mimic/hyperparameters/mixing/m${l} --data mimic_val --nfolds 1 --update-mixing $l
    done
done

error=( 0.0001 0.001 0.01 0.05 0.1 0.5 1.0 2.0 3.0 )

models=( FlowGRUModel FlowLSTMModel )

for e in "${error[@]}"
do
    for m in "${models[@]}"
    do
    echo "error $e\n"
    echo "$m\n"
    python train.py --model $m --max_epochs 100 --hidden_dim_t 8 --hidden_dim_i 4 --logfolder mimic/hyperparameters/error/e${e} --data mimic_val --nfolds 1 --merror $e
    done
done

