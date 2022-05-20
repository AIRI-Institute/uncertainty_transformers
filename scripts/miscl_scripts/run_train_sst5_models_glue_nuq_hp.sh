cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_20newsgroups_dpp_hp.py args='ue\=mc do_ue_estimate\=False +ue.use_spectralnorm\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw data.validation_subsample\=0.0 +data.eval_subsample\=0.2 training\=electra_base training.num_train_epochs\=12 training.learning_rate\=7e-06 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.1 +ue.lamb_intra\=0.003 ue.lamb\=0.006' task_configs=sst5.yaml output_dir=../workdir/run_train_models_nuq_hp/electra_raw_no_sn/sst5/0.0