cd ../../src
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-raw-True/mrpc/0.0/mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-raw-True/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-reg-True/mrpc/0.0/mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-reg-True/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mc_mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-raw-True/mrpc/0.0/mc_mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-raw-True/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mc_mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/mrpc.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-reg-True/mrpc/0.0/mc_mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-reg-True/mrpc/0.0/models/mrpc
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-raw-True/cola/0.0/mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-raw-True/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-reg-True/cola/0.0/mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-reg-True/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mc_mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-raw-True/cola/0.0/mc_mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-raw-True/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[1] args='ue\=mc_mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/cola.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-reg-True/cola/0.0/mc_mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-reg-True/cola/0.0/models/cola
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[2,3] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-raw-True/sst2/0.0/mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-raw-True/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[2,3] args='ue\=mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-reg-True/sst2/0.0/mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-reg-True/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[2,3] args='ue\=mc_mahalanobis ue.use_spectralnorm\=False ue.use_selective\=False do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-raw-True/sst2/0.0/mc_mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-raw-True/sst2/0.0/models/sst2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python run_glue_for_model_series.py cuda_devices=[2,3] args='ue\=mc_mahalanobis ue.use_spectralnorm\=False ue.use_selective\=True do_ue_estimate\=True ue.use_cache\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base' config_path=../configs/sst2.yaml output_dir=../workdir/run_glue_for_model_series_sn/electra-reg-True/sst2/0.0/mc_mahalanobis model_series_dir=/mnt/users/avazhentsev/uncertainty-estimation/workdir/run_train_models_sn/electra-reg-True/sst2/0.0/models/sst2