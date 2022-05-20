# How to run experiments
For now, one can choose from two methods of running experiments: pipelined approach and run experiments with run_average_results.py on generated config.
## Pipelined Approach
Examples of this method could be found in ./ner_scripts and ./miscl_scripts. This folders contain scripts for running experiments for NER/Classification tasks. To obtain more information on scripts in folder, look for README files in this folder.
## Regularization
For now there are two types of regularizers - reg-curr and metric.

To enable reg-curr regularizer, set in exp config/in args argument of command following: ue.reg_type=reg-curr ue.lamb=0.05 ue.use_selective=True

To enable metric regularizer, set in exp config/in args argument of command following: ue.reg_type=metric ue.lamb=0.05 ue.margin=0.05 ue.lamb_intra=0.05 ue.use_selective=True
### Examples of different experiments
#### Classification
1. Train, estimate and save results for MRPC dataset with metric regularization, MC dropout on last layer
```
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py task_configs=mrpc_selective.yaml cuda_devices=[0] seeds=[1,45,74] args='data.task_name\=mrpc ue.dropout_subs\=last ue.lamb\=0.05 ue.margin\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_train_models/electra-metric-005-005/mrpc/last'
```
```
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[0] seeds=[1,3,67,56,92] config_path=../configs/mrpc_selective.yaml model_series_dir=../workdir/run_train_models/electra-metric-005-005/mrpc/last/models/mrpc_selective/ args='data.task_name\=mrpc ue.dropout_subs\=last ue.lamb\=0.05 ue.margin\=0.05 ue.reg_type\=metric ue.calibrate\=False ue.use_cache\=True' output_dir='../workdir/run_glue_for_model_series/electra-metric-005-005/mrpc/last'
```
```
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric-005-005/mrpc/last/results'
```
#### NER
1. Train, estimate and save results for 10% of CoNLL-2003, Monte-Carlo dropout on last layer
```
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_raw_01/last'
```
```
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_raw_01/last/models/conll2003_selective/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/last'
```
```
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/last/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_raw_01/last'
```
Note: it's crucial to use model name in model_series_dir (e.g. bert, electra), because without it model could be loaded with errors.

## Run Average Results
Algorithm:
1. Generate config for all experiments with /src/generate_series_of_exps.ipynb
2. Set command name and output dir for current experiment in /src/run_average_results.py
3. Set pathes for every results folder in run_task_on_multiple_gpus.py (if necessary)
4. Set visible devices in /src/utils_tasks.py (if necessary)
5. Run following command:
```
CUDA_VISIBLE_DEVICES=1 HYDRA_CONFIG_PATH=../configs HYDRA_CONFIG_NAME=series_exps python run_average_results.py 
```
# How to reproduce current results
Go to folder ./ue_scripts and look for README in this folder.
