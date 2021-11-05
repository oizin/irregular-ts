#!/bin/sh
echo "1_000 examples simulation\n"
julia scripts/data/simulate_ou.jl 1000 123
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 1000

echo "5_000 examples simulation\n"
julia scripts/data/simulate_ou.jl 5000 123
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 1000

echo "10_000 examples simulation\n"
julia scripts/data/simulate_ou.jl 10000 123
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 1000

echo "20_000 examples simulation\n"
julia scripts/data/simulate_ou.jl 20000 123
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 1000
