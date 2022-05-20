cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.01 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.01/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.01/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.01/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.01/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.02 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.02/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.02/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.02/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.02/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.05 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.05/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.05/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.05/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.05/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.1 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.1/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.1/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.1/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.1/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.15 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.15/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.15/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.15/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.15/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.2 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.2/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.2/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.2/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.2/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.25 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.25/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.25/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.25/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.25/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.3 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.3/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.3/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.3/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.3/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.4 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=5 training.learning_rate\=0.0001 training.per_device_train_batch_size\=16 +training.weight_decay\=0' config_path=../configs/mrpc.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.4/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/mrpc/subsample_perc_0.4/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/mrpc/subsample_perc_0.4/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/mrpc/subsample_perc_0.4/sngp;