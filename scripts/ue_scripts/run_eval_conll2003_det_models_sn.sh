cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.subsample_perc\=0.1 data.subsample_perc_val\=0.1 training\=electra_base' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series_sn/electra_metric_sn/conll2003/0.1/mahalanobis model_series_dir=../workdir/run_train_models_sn/electra_metric_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series_sn/electra_metric_sn/conll2003/0.1/mahalanobis extract_config=False output_dir=../workdir/run_calc_ues_metrics_sn/electra_metric_sn/conll2003/mahalanobis;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.subsample_perc\=0.1 data.subsample_perc_val\=0.1 training\=electra_base' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series_sn/electra_reg_sn/conll2003/0.1/mahalanobis model_series_dir=../workdir/run_train_models_sn/electra_reg_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series_sn/electra_reg_sn/conll2003/0.1/mahalanobis extract_config=False output_dir=../workdir/run_calc_ues_metrics_sn/electra_reg_sn/conll2003/mahalanobis;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_conll2003.py args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.subsample_perc\=0.1 data.subsample_perc_val\=0.1 training\=electra_base' config_path=../configs/conll2003.yaml output_dir=../workdir/run_tasks_for_model_series_sn/electra_raw_sn/conll2003/0.1/mahalanobis model_series_dir=../workdir/run_train_models_sn/electra_raw_sn/conll2003/0.1/models/conll2003
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir=../workdir/run_tasks_for_model_series_sn/electra_raw_sn/conll2003/0.1/mahalanobis extract_config=False output_dir=../workdir/run_calc_ues_metrics_sn/electra_raw_sn/conll2003/mahalanobis;