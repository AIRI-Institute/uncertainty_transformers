cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] script=run_glue_dpp_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 model.model_name_or_path\=microsoft/deberta-base training\=electra_base training.num_train_epochs\=12 training.learning_rate\=3e-05 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1 +ue.margin\=10.0 +ue.lamb_intra\=0.005 ue.lamb\=0.1' task_configs=mrpc.yaml output_dir=../workdir/run_train_models_dpp_hp/deberta_raw_no_sn/mrpc/0.0
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] script=run_glue_dpp_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 model.model_name_or_path\=microsoft/deberta-base training\=electra_base training.num_train_epochs\=13 training.learning_rate\=7e-06 training.per_device_train_batch_size\=4 +training.weight_decay\=0 +ue.margin\=5.0 +ue.lamb_intra\=0.05 ue.lamb\=0.003' task_configs=cola.yaml output_dir=../workdir/run_train_models_dpp_hp/deberta_raw_no_sn/cola/0.0
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] script=run_glue_dpp_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 model.model_name_or_path\=microsoft/deberta-base training\=electra_base training.num_train_epochs\=5 training.learning_rate\=3e-05 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=0.5 +ue.lamb_intra\=0.001 ue.lamb\=0.003' task_configs=sst2.yaml output_dir=../workdir/run_train_models_dpp_hp/deberta_raw_no_sn/sst2/0.0