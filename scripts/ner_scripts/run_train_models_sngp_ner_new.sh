cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1] script=run_conll2003.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.use_selective\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=5e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.01' task_configs=conll2003.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/conll2003/0.1