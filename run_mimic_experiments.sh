#!/bin/bash

models=( LinearModel CatboostModel FlowGRUModel DecayGRUModel  FlowLSTMModel IMODE GRUModel LSTMModel ODEGRUModel ODELSTMModel)

hidden=( 0 0 32 8 32 32 32 32 32 32 24 24 )

mixing=( 0 0 0.0001 0.0001 0.0001 0.001 0.001 0.001 0 0 0.001 0.0001)

error=( 0 0 0.0001 0.0001 0.0001 0.001 0.001 0.001 0 0 0.001 0.001)

models=( ODEGRUModel ODELSTMModel DecayGRUModel  )

for i in "${!models[@]}"
do
python train.py --hidden_dim_t=${hidden[i]} --hidden_dim_i=8 --model=${models[i]} --max_epochs=100 --nfolds=3 --seed 3 --data mimic --logfolder mimic --update-mixing ${mixing[i]} --merror ${error[i]} --accelerator gpu
done