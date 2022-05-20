cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.01 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.025 +ue.lamb_intra\=0.2 ue.lamb\=1.0' task_configs=clinc.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/clinc/0.01
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.02 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.025 +ue.lamb_intra\=0.2 ue.lamb\=1.0' task_configs=clinc.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/clinc/0.02
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.05 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.025 +ue.lamb_intra\=0.2 ue.lamb\=1.0' task_configs=clinc.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/clinc/0.05
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.1 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.025 +ue.lamb_intra\=0.2 ue.lamb\=1.0' task_configs=clinc.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/clinc/0.1
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.15 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.025 +ue.lamb_intra\=0.2 ue.lamb\=1.0' task_configs=clinc.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/clinc/0.15
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.2 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.025 +ue.lamb_intra\=0.2 ue.lamb\=1.0' task_configs=clinc.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/clinc/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.3 training\=electra_base training.num_train_epochs\=7 training.learning_rate\=7e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01 +ue.margin\=0.025 +ue.lamb_intra\=0.2 ue.lamb\=1.0' task_configs=clinc.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/clinc/0.3
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.01 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=5e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/rostd/0.01
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.02 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=5e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/rostd/0.02
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.05 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=5e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/rostd/0.05
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.1 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=5e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/rostd/0.1
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.15 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=5e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/rostd/0.15
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.2 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=5e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/rostd/0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2] script=run_ood.py args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=False ue.reg_type\=raw +ue.use_spectralnorm\=False data.subsample_perc\=0.3 training\=electra_base training.num_train_epochs\=8 training.learning_rate\=5e-06 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 +ue.margin\=10.0 +ue.lamb_intra\=0.008 ue.lamb\=0.008' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_no_sn/rostd/0.3