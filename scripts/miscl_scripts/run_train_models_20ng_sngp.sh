cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python run_train_models_20newsgroups.py cuda_devices=[0,1] args='ue\=sngp do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.use_spectralnorm\=False data.validation_subsample\=0.0 ue.sngp.ridge_factor\=0.001 ue.sngp.momentum\=0.9999 training\=electra_base training.num_train_epochs\=12 training.learning_rate\=2e-05 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1' task_configs=20newsgroups.yaml output_dir=../workdir/run_train_models/electra-raw-sngp-correct-hp/20newsgroups/0.001_0.9999_0.0