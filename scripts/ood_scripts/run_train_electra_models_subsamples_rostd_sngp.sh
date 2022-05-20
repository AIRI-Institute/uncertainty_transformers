cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.subsample_perc\=0.0 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.0
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.subsample_perc\=0.005 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.005
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.subsample_perc\=0.01 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.01
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.subsample_perc\=0.015 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.015
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.subsample_perc\=0.02 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.02
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.subsample_perc\=0.03 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.03
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.subsample_perc\=0.04 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0' task_configs=rostd.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.04