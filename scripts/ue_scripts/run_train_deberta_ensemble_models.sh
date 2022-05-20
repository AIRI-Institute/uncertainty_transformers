cd ../../src
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py task_configs=mrpc.yaml args='data.subsample_perc\=0.9 ue.calibrate\=True data.validation_subsample\=0.0 do_train\=True do_eval\=False do_ue_estimate\=False training.num_train_epochs\=12 training.learning_rate\=3e-05 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1 model.model_name_or_path\=microsoft/deberta-base' output_dir='../workdir/run_train_ensemble_series/deberta/mrpc/' cuda_devices=[0,1,2,3,4,5]
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py task_configs=cola.yaml args='data.subsample_perc\=0.9 ue.calibrate\=True data.validation_subsample\=0.0 do_train\=True do_eval\=False do_ue_estimate\=False training.num_train_epochs\=13 training.learning_rate\=7e-06 training.per_device_train_batch_size\=4 +training.weight_decay\=0 model.model_name_or_path\=microsoft/deberta-base' output_dir='../workdir/run_train_ensemble_series/deberta/cola/' cuda_devices=[0,1,2,3,4,5]
HYDRA_CONFIG_PATH=../configs/run_train_ensemble_series.yaml python ./run_train_ensemble_series.py task_configs=sst2.yaml args='data.subsample_perc\=0.143 ue.calibrate\=True data.validation_subsample\=0.0 do_train\=True do_eval\=False do_ue_estimate\=False training.num_train_epochs\=5 training.learning_rate\=3e-05 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01 model.model_name_or_path\=microsoft/deberta-base' output_dir='../workdir/run_train_ensemble_series/deberta/sst2/' cuda_devices=[0,1,2,3,4,5]