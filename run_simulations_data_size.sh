#!/bin/sh

echo "1_000 examples simulation\n"
for s in 1 2 3
do
echo "1_000 examples simulation $s\n"
python train.py --model LinearModel --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model CatboostModel --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model ODEGRUModel --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model FlowGRUModel --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model DecayGRUModel --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model ODELSTMModel --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model FlowLSTMModel --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model IMODE --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model GRUModel --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
python train.py --model LSTMModel --max_epochs 100 --logfolder data_size_1000 --data simulation_1000_v${s}
done

echo "5_000 examples simulation\n"
for s in 1 2 3
do
echo "5_000 examples simulation $s\n"
python train.py --model LinearModel --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model CatboostModel --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model ODEGRUModel --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model FlowGRUModel --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model DecayGRUModel --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model ODELSTMModel --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model FlowLSTMModel --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model IMODE --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model GRUModel --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
python train.py --model LSTMModel --max_epochs 100 --logfolder data_size_5000 --data simulation_1000_v${s}
done

echo "10_000 examples simulation\n"
for s in 1 2 3
do
echo "10_000 examples simulation $s\n"
python train.py --model LinearModel --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model CatboostModel --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model ODEGRUModel --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model FlowGRUModel --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model DecayGRUModel --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model ODELSTMModel --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model FlowLSTMModel --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model IMODE --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model GRUModel --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
python train.py --model LSTMModel --max_epochs 100 --logfolder data_size_10000 --data simulation_10000_v${s}
done