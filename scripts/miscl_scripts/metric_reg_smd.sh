cd ../../src
# Script for training all models on MRPC, COLA, SST2-10% - for Maha and Maha+SN. Note - to proper metrics calculation res path have to contain maha_mc or maha_sn_mc substring.

# maha + mc
# train
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=mc_mahalanobis do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/electra-metric/mrpc/maha_mc
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=mc_mahalanobis do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra-metric/cola/maha_mc
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=mc_mahalanobis do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' task_configs=sst2.yaml output_dir=../workdir/run_train_models/electra-metric/sst2/maha_mc
# estimate
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/electra-metric/mrpc/maha_mc/models/mrpc/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/electra-metric/cola/maha_mc/models/cola/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/electra-metric/cola/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/electra-metric/sst2/maha_mc/models/sst2/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/maha_mc'

# calc metrics
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/mrpc/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/cola/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/cola/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/sst2/maha_mc'


# maha + sn + mc
# train
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=mc_mahalanobis do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/electra-metric/mrpc/maha_sn_mc
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=mc_mahalanobis do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra-metric/cola/maha_sn_mc
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=mc_mahalanobis do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' task_configs=sst2.yaml output_dir=../workdir/run_train_models/electra-metric/sst2/maha_sn_mc
# estimate
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/electra-metric/mrpc/maha_sn_mc/models/mrpc/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' output_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/electra-metric/cola/maha_sn_mc/models/cola/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' output_dir='../workdir/run_glue_for_model_series/electra-metric/cola/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/electra-metric/sst2/maha_sn_mc/models/sst2/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' output_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/maha_sn_mc'

# calc metrics
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/maha_sn_mc/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/mrpc/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/cola/maha_sn_mc/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/cola/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/maha_sn_mc/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/sst2/maha_sn_mc'




# nuq
# train
#HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=nuq do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/electra-metric/mrpc/nuq
#HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=nuq do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra-metric/cola/nuq
#HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=nuq do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' task_configs=sst2.yaml output_dir=../workdir/run_train_models/electra-metric/sst2/nuq
# estimate
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/electra-metric/mrpc/nuq/models/mrpc/ args='ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/nuq'
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/electra-metric/cola/nuq/models/cola/ args='ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/electra-metric/cola/nuq'
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/electra-metric/sst2/nuq/models/sst2/ args='ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/nuq'

# calc metrics
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/nuq/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/mrpc/nuq'
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/cola/nuq/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/cola/nuq'
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/nuq/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/sst2/nuq'


# nuq + sn
# train
#HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=nuq do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/electra-metric/mrpc/nuq_sn
#HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=nuq do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' task_configs=cola.yaml output_dir=../workdir/run_train_models/electra-metric/cola/nuq_sn
#HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] args='ue\=nuq do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' task_configs=sst2.yaml output_dir=../workdir/run_train_models/electra-metric/sst2/nuq_sn
# estimate
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/electra-metric/mrpc/nuq_sn/models/mrpc/ args='ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' output_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/nuq_sn'
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/electra-metric/cola/nuq_sn/models/cola/ args='ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' output_dir='../workdir/run_glue_for_model_series/electra-metric/cola/nuq_sn'
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[2,3] seeds=[17,42,51,77,91,102] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/electra-metric/sst2/nuq_sn/models/sst2/ args='ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.05 +ue.margin\=0.05 ue.reg_type\=metric ue.use_spectralnorm\=True' output_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/nuq_sn'

# calc metrics
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/mrpc/nuq_sn/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/mrpc/nuq_sn'
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/cola/nuq_sn/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/cola/nuq_sn'
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/electra-metric/sst2/nuq_sn/results' output_dir='../workdir/run_calc_ues_metrics/electra-metric/sst2/nuq_sn'