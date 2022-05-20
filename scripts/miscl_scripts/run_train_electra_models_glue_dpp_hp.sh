cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue_dpp_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=12 training.learning_rate\=5e-05 training.per_device_train_batch_size\=32 +training.weight_decay\=0.1 +ue.margin\=0.5 +ue.lamb_intra\=0.05 ue.lamb\=0.001' task_configs=mrpc.yaml output_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/mrpc/0.0
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue_dpp_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=1e-05 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1 +ue.margin\=1.0 +ue.lamb_intra\=0.05 ue.lamb\=0.02' task_configs=cola.yaml output_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/cola/0.0
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue_dpp_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=1e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.1 +ue.margin\=1.0 +ue.lamb_intra\=0.005 ue.lamb\=0.02' task_configs=sst2.yaml output_dir=../workdir/run_train_models_dpp_hp/electra_raw_no_sn/sst2/0.0