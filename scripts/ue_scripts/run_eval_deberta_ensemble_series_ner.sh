cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_ensemble_series.yaml python run_tasks_for_ensemble_series.py cuda_devices=0 config_path=../configs/conll2003.yaml args='data.subsample_perc\=0.143 ue.calibrate\=False data.subsample_perc_val\=0.1 training.num_train_epochs\=12 training.learning_rate\=5e-05 training.per_device_train_batch_size\=32 +training.weight_decay\=0 +ue.margin\=5.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' ensemble_series_dir=../workdir/run_train_ensemble_series/deberta/conll2003/ensembles/ output_dir=../workdir/run_tasks_for_ensemble_series/deberta/conll2003/ script=run_conll2003.py
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_ensemble_series/deberta/conll2003/final_results extract_config=False output_dir=../workdir/run_calc_ues_metrics/deberta_raw_no_sn/conll2003/ensemble;