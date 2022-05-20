cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1] script=run_glue.py args='ue\=mahalanobis ++ue.use_spectralnorm\=False ue.use_selective\=False ue.reg_type\=raw do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=12 training.learning_rate\=7e-06 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 ++ue.margin\=0.1 ++ue.lamb_intra\=0.003 ++ue.lamb\=0.006 training.per_device_eval_batch_size\=64' config_path=../configs/sst5.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/sst5//mahalanobis model_series_dir=../workdir/run_train_models/electra_raw_no_sn/sst5/models/sst5
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1] script=run_glue.py args='ue\=mc ++ue.use_spectralnorm\=False ue.use_selective\=False ue.reg_type\=raw do_ue_estimate\=True ue.use_cache\=False ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=12 training.learning_rate\=7e-06 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 ++ue.margin\=0.1 ++ue.lamb_intra\=0.003 ++ue.lamb\=0.006 training.per_device_eval_batch_size\=64' config_path=../configs/sst5.yaml output_dir=../workdir/run_glue_for_model_series/electra_raw_no_sn/sst5//mc model_series_dir=../workdir/run_train_models/electra_raw_no_sn/sst5/models/sst5