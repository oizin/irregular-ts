#!/bin/sh
echo "1_000 examples simulation\n"
for s in 1 2 3
do
echo "1_000 examples simulation $s\n"
julia scripts/simulations/simulate_glucose.jl 1000 0.0 $s 
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 1000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 1000
done

echo "5_000 examples simulation\n"
for s in 1 2 3
do
echo "5_000 examples simulation $s\n"
julia scripts/simulations/simulate_glucose.jl 5000 0.0 $s
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 5000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 5000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 5000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 5000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 5000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 5000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 5000
done

echo "10_000 examples simulation\n"
for s in 1 2 3
do
echo "10_000 examples simulation $s\n"
julia scripts/simulations/simulate_glucose.jl 10000 0.0 $s
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 10000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 10000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 10000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 10000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 10000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 10000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 10000
done

echo "20_000 examples simulation\n"
for s in 1 2 3
do
echo "20_000 examples simulation $s\n"
julia scripts/simulations/simulate_glucose.jl 20000 0.0 $s
python simulation.py --net ctRNNModel --max_epochs 100 --logfolder data_size --N 20000
python simulation.py --net ctGRUModel --max_epochs 100 --logfolder data_size --N 20000
python simulation.py --net ctLSTMModel --max_epochs 100 --logfolder data_size --N 20000
python simulation.py --net ODEGRUBayes --max_epochs 100 --logfolder data_size --N 20000
python simulation.py --net neuralJumpModel --max_epochs 100 --logfolder data_size --N 20000
python simulation.py --net resNeuralJumpModel --max_epochs 100 --logfolder data_size --N 20000
python simulation.py --net IMODE --max_epochs 100 --logfolder data_size --N 20000
done