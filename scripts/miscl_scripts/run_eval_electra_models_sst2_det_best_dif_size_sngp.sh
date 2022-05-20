cd ../../src
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.001 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=2e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.001/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/sst2/subsample_perc_0.001/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.001/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/sst2/subsample_perc_0.001/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.005 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=2e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.005/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/sst2/subsample_perc_0.005/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.005/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/sst2/subsample_perc_0.005/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.01 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=2e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.01/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/sst2/subsample_perc_0.01/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.01/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/sst2/subsample_perc_0.01/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.015 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=2e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.015/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/sst2/subsample_perc_0.015/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.015/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/sst2/subsample_perc_0.015/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.02 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=2e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.02/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/sst2/subsample_perc_0.02/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.02/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/sst2/subsample_perc_0.02/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.025 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=2e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.025/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/sst2/subsample_perc_0.025/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.025/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/sst2/subsample_perc_0.025/sngp;
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] script=run_glue.py args='ue\=sngp do_ue_estimate\=True ue.calibrate\=False data.subsample_perc\=0.03 data.validation_subsample\=0.0 training\=electra_base training.num_train_epochs\=15 training.learning_rate\=2e-05 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' config_path=../configs/sst2.yaml output_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.03/sngp model_series_dir=../workdir/run_train_models/electra_raw_sngp/sst2/subsample_perc_0.03/models/sst2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir=../workdir/run_tasks_for_model_series/electra_raw_sngp/sst2/subsample_perc_0.03/sngp extract_config=False output_dir=../workdir/run_calc_ues_metrics/electra_raw_sngp/sst2/subsample_perc_0.03/sngp;