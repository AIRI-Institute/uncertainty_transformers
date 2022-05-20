cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True ue.reg_type\=metric do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series_sn_old_net/electra_metric_sn/mrpc/0.0/mahalanobis model_series_dir=../workdir/run_train_models_sn_old_net/electra_metric_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series_sn_old_net/electra_metric_sn/mrpc/0.0/mahalanobis extract_config=False output_dir=../workdir/run_calc_ues_metrics_sn_old_net/electra_metric_sn/mrpc/mahalanobis;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True ue.reg_type\=reg-curr do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series_sn_old_net/electra_reg_sn/mrpc/0.0/mahalanobis model_series_dir=../workdir/run_train_models_sn_old_net/electra_reg_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series_sn_old_net/electra_reg_sn/mrpc/0.0/mahalanobis extract_config=False output_dir=../workdir/run_calc_ues_metrics_sn_old_net/electra_reg_sn/mrpc/mahalanobis;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False ue.reg_type\=raw do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series_sn_old_net/electra_raw_sn/mrpc/0.0/mahalanobis model_series_dir=../workdir/run_train_models_sn_old_net/electra_raw_sn/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series_sn_old_net/electra_raw_sn/mrpc/0.0/mahalanobis extract_config=False output_dir=../workdir/run_calc_ues_metrics_sn_old_net/electra_raw_sn/mrpc/mahalanobis;