cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.01 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.01
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.02 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.02
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.05 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.05
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.1 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.1
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.15 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.15
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.2 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.2
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.25 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.25
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.3 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.3
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=False ue.calibrate\=False data.validation_subsample\=0.0 data.subsample_perc\=0.4 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=7e-06 training.per_device_train_batch_size\=8 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra_raw_sngp/cola/subsample_perc_0.4