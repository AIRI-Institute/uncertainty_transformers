cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py task_configs=mrpc.yaml args='data.subsample_perc\=0.9 ue.calibrate\=True data.validation_subsample\=0.0 training.num_train_epochs\=12 training.learning_rate\=5e-05 training.per_device_train_batch_size\=32 +training.weight_decay\=0.1' output_dir='../workdir/run_train_ensemble_series/mrpc/electra/' cuda_devices=[0,1,2,3,4,5] script=run_glue.py
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py task_configs=cola.yaml args='data.subsample_perc\=0.9 ue.calibrate\=True data.validation_subsample\=0.0 training.num_train_epochs\=8 training.learning_rate\=1e-05 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1' output_dir='../workdir/run_train_ensemble_series/cola/electra/' cuda_devices=[0,1,2,3,4,5] script=run_glue.py
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py task_configs=sst2.yaml args='data.subsample_perc\=0.143 ue.calibrate\=True data.validation_subsample\=0.0 training.num_train_epochs\=15 training.learning_rate\=1e-05 training.per_device_train_batch_size\=64 +training.weight_decay\=0.1' output_dir='../workdir/run_train_ensemble_series/sst2/electra/' cuda_devices=[0,1,2,3,4,5] script=run_glue.py