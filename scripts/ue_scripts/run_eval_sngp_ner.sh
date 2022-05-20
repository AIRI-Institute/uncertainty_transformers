cd ../../src
# for this script, switch to torch 1.7.1
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 training.per_device_eval_batch_size\=64 training\=electra_base ue.committee_size\=10 +ue.sngp.use_paper_version\=True +ue.use_paper_version\=True' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/conll2003/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/conll2003/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/conll2003/sngp;