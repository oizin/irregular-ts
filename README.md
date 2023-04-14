# Forecasting with irregularly measured blood glucose with deep learning

Repository for the paper:

Continuous time recurrent neural networks : overview and application to forecasting blood glucose in the intensive care unit

## Data setup

### MIMIC-IV

As we are not data custodians we cannot publicaly share the MIMIC-IV data used. However, it is available to those with
credentialed access to physionet.org. Credentialed access can be requested through your physionet account. 

We used dbt to connect to the Google Bigquery MIMIC-IV database that can be autopopulated
through the physionet interface. 

https://docs.getdbt.com/reference/warehouse-setups/bigquery-setup

After setting up dbt run: 

```
cd scripts/data/mimic4glucose
dbt run
```

Then run the notebook: `scripts/data/setup_mimic_data.ipynb`

### Simulations

After installing torchctrnn run the notebook: `scripts/data/setup_simulations.ipynb`

### Notable dependencies
    
- python 3.9+
- dbt
- db-dtype
- pytorch
- torchctrnn: pip install https://github.com/oizin/torchctrnn/tarball/main
- numba
- properscoring
    
## Repeating the experiments
    
The full analysis is reproducible as follows:

1) Run the bash scripts:

```
./run_simulation_experiments.sh
./run_mimic_experiments.sh
```

2) Run the evaluation notebooks

- scripts/results/simulation_data_size.ipynb
- scripts/results/simulation_all_5000.ipynb 
- scripts/results/mimic_results_from_predictions.ipynb
    
The following is an example of running a single experiment
```
python train.py --model=LinearModel --logfolder=results --test --nfolds=1 --seed 1 --data mimic
```

