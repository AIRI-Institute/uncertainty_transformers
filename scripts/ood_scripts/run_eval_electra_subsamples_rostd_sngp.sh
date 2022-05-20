cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.0 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 training.per_device_eval_batch_size\=64' config_path=../configs/rostd.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/rostd/subsample_perc_0.0/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.0/models/rostd
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.005 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 training.per_device_eval_batch_size\=64' config_path=../configs/rostd.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/rostd/subsample_perc_0.005/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.005/models/rostd
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.01 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 training.per_device_eval_batch_size\=64' config_path=../configs/rostd.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/rostd/subsample_perc_0.01/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.01/models/rostd
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.015 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 training.per_device_eval_batch_size\=64' config_path=../configs/rostd.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/rostd/subsample_perc_0.015/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.015/models/rostd
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.02 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 training.per_device_eval_batch_size\=64' config_path=../configs/rostd.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/rostd/subsample_perc_0.02/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.02/models/rostd
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.03 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 training.per_device_eval_batch_size\=64' config_path=../configs/rostd.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/rostd/subsample_perc_0.03/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.03/models/rostd
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_ood.py seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.04 training\=electra_base training.num_train_epochs\=13 training.learning_rate\=3e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0 training.per_device_eval_batch_size\=64' config_path=../configs/rostd.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/rostd/subsample_perc_0.04/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/rostd/subsample_perc_0.04/models/rostd