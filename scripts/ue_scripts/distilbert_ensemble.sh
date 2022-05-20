cd ../../src

# Train models
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py script=run_glue.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.1 data.subsample_perc\=0.63 training\=electra_base ue.dropout_subs\=last model.model_name_or_path\=distilbert-base-cased ue.use_selective\=False ue.reg_type\=raw training.learning_rate\=7e-5 training.num_train_epochs\=6 training.per_device_train_batch_size\=32 +training.weight_decay\=0.0' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/distilbert_ensemble/mrpc
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py script=run_glue.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.1 data.subsample_perc\=0.63 training\=electra_base ue.dropout_subs\=last model.model_name_or_path\=distilbert-base-cased ue.use_selective\=False ue.reg_type\=raw training.learning_rate\=7e-5 training.num_train_epochs\=6 training.per_device_train_batch_size\=32 +training.weight_decay\=0.0' task_configs=cola.yaml output_dir=../workdir/run_train_models/distilbert_ensemble/cola
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py script=run_glue.py cuda_devices=[0,1,2,3,4,5] args='ue\=mc do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.1 data.subsample_perc\=0.1 training\=electra_base ue.dropout_subs\=last model.model_name_or_path\=distilbert-base-cased ue.use_selective\=False ue.reg_type\=raw training.learning_rate\=3e-5 training.num_train_epochs\=14 training.per_device_train_batch_size\=32 +training.weight_decay\=0.0' task_configs=sst2.yaml output_dir=../workdir/run_train_models/distilbert_ensemble/sst2

HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py script=run_conll2003.py task_configs=conll2003.yaml args='do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 model.model_name_or_path\=distilbert-base-cased ue.use_selective\=False ue.reg_type\=raw training.learning_rate\=3e-5 training.num_train_epochs\=15 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1' cuda_devices=[0,1,2,3,4,5] output_dir='../workdir/run_train_models/distilbert_ensemble/conll'



HYDRA_CONFIG_PATH=../configs/run_tasks_for_ensemble_series.yaml python run_tasks_for_ensemble_series.py cuda_devices=0 config_path=../configs/mrpc.yaml args='ue.calibrate\=False' ensemble_series_dir=../workdir/run_train_models/distilbert_ensemble/mrpc/ensembles/ output_dir=../workdir/run_glue_for_ensemble_series/distilbert_ensemble/mrpc/
HYDRA_CONFIG_PATH=../configs/run_tasks_for_ensemble_series.yaml python run_tasks_for_ensemble_series.py cuda_devices=0 config_path=../configs/cola.yaml args='ue.calibrate\=False' ensemble_series_dir=../workdir/run_train_models/distilbert_ensemble/cola/ensembles/ output_dir=../workdir/run_glue_for_ensemble_series/distilbert_ensemble/cola/
HYDRA_CONFIG_PATH=../configs/run_tasks_for_ensemble_series.yaml python run_tasks_for_ensemble_series.py cuda_devices=0 config_path=../configs/sst2.yaml args='ue.calibrate\=False' ensemble_series_dir=../workdir/run_train_models/distilbert_ensemble/sst2/ensembles/ output_dir=../workdir/run_glue_for_ensemble_series/distilbert_ensemble/sst2/

HYDRA_CONFIG_PATH=../configs/run_tasks_for_ensemble_series.yaml python run_tasks_for_ensemble_series.py script=run_conll2003.py cuda_devices=0 config_path=../configs/conll2003.yaml args='data.subsample_perc\=0.1 ue.calibrate\=False data.subsample_perc_val\=0.1' ensemble_series_dir=../workdir/run_train_models/distilbert_ensemble/conll/ensembles/ output_dir=../workdir/run_glue_for_ensemble_series/distilbert_ensemble/conll2003/



# Finally, calc metrics
# last
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py extract_config=False runs_dir='../workdir/run_glue_for_ensemble_series/distilbert_ensemble/mrpc/final_results/' output_dir='../workdir/run_calc_ues_metrics/distilbert_ensemble/mrpc/de'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py extract_config=False runs_dir='../workdir/run_glue_for_ensemble_series/distilbert_ensemble/cola/final_results' output_dir='../workdir/run_calc_ues_metrics/distilbert_ensemble/cola/de'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py extract_config=False runs_dir='../workdir/run_glue_for_ensemble_series/distilbert_ensemble/sst2/final_results' output_dir='../workdir/run_calc_ues_metrics/distilbert_ensemble/sst2/de'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py extract_config=False runs_dir='../workdir/run_glue_for_ensemble_series/distilbert_ensemble/conll2003/final_results' output_dir='../workdir/run_calc_ues_metrics/distilbert_ensemble/conll2003/de'
