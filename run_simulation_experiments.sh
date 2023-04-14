#!/bin/bash

models=( LinearModel CatboostModel FlowGRUModel DecayGRUModel  FlowLSTMModel IMODE GRUModel LSTMModel ODEGRUModel ODELSTMModel )

hidden=( 0 0 2 2 2 24 24 16 2 24 8 24 )

mixing=( 0 0 0.0001 0.0001 0.0001 0.001 0.001 0.001 0 0 0.001 0.0001 )

error=( 0 0 0.0001 0.0001 0.0001 0.001 0.001 0.001 0 0 0.0001 0.001 )

echo "1_000 examples simulation\n"
for s in 0 1 2
do
    for i in "${!models[@]}"
    do
    echo "1_000 examples simulation $s\n"
    echo "${models[i]}\n"
    python train.py --model ${models[i]} --max_epochs 100 --hidden_dim_t ${hidden[i]} --hidden_dim_t 2 --logfolder sim/data_size_1000 --data simulation_1000_v${s} --update-mixing ${mixing[i]} --merror ${error[i]} --accelerator gpu
    done
done

echo "5_000 examples simulation\n"
for s in 0 1 2
do
    for i in "${!models[@]}"
    do
    echo "5_000 examples simulation $s\n"
    python train.py --model ${models[i]} --max_epochs 100 --hidden_dim_t ${hidden[i]} --hidden_dim_i 2 --logfolder sim/data_size_5000 --data simulation_5000_v${s} --update-mixing ${mixing[i]} --merror ${error[i]} --accelerator gpu
    done
done

echo "10_000 examples simulation\n"
for s in 0 1 2
do
    for i in "${!models[@]}"
    do
    echo "10_000 examples simulation $s\n"
    python train.py --model ${models[i]} --max_epochs 100 --hidden_dim_t ${hidden[i]} --hidden_dim_i 2 --logfolder sim/data_size_10000 --data simulation_10000_v${s} --update-mixing ${mixing[i]} --merror ${error[i]}
    done
done

echo "non-stationary\n"
for s in 0 1 2
do
echo "non-stationary $s\n"
    for i in "${!models[@]}"
    do
    echo "non-stationary examples simulation $s\n"
    python train.py --model ${models[i]} --max_epochs 100 --hidden_dim_t ${hidden[i]} --hidden_dim_i 2 --logfolder sim/non_stationary --data simulation_nonstationary_v${s} --update-mixing ${mixing[i]} --merror ${error[i]} --accelerator gpu
    done
done

echo "measurement-error\n"
for s in 0 1 2
do
echo "measurement-error $s\n"
    for i in "${!models[@]}"
    do
    echo "measurement-error examples simulation $s\n"
    python train.py --model ${models[i]} --max_epochs 100 --hidden_dim_t ${hidden[i]} --hidden_dim_i 2 --logfolder sim/measurement_error --data simulation_error_v${s} --update-mixing ${mixing[i]} --merror 1.0 --accelerator gpu
    done
done
