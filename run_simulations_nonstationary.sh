#!/bin/sh
echo "non-stationary\n"
for s in 1 2 3
do
echo "non-stationary $s\n"
julia scripts/simulations/simulate_glucose.jl 10000 0.0 $s 1
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder nonstationary1 --N 10000 --stationary 0
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder nonstationary1 --N 10000  --stationary 0
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder nonstationary1 --N 10000 --stationary 0
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder nonstationary1 --N 10000 --stationary 0
python simulation.py --net neuralJumpModel --max_epochs 200 --logfolder nonstationary1 --N 10000 --stationary 0
python simulation.py --net resNeuralJumpModel --max_epochs 200 --logfolder nonstationary1 --N 10000 --stationary 0
python simulation.py --net IMODE --max_epochs 100 --logfolder nonstationary1 --N 10000 --stationary 0
done