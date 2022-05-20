cd ../../src
# Also add CoNLL-2003 - metric

# Script for training all models on MRPC, COLA, SST2-10% and CoNLL-2003 for new metric loss with 3 params
# train - metric loss, no SN
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.lamb_intra\=0.005 training.learning_rate\=2e-05 +ue.margin\=0.1 training.num_train_epochs\=7 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1 ue.reg_type\=metric' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/metric_electra/mrpc/metric

HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.001 +ue.lamb_intra\=0.05 training.learning_rate\=5e-06 +ue.margin\=5.0 training.num_train_epochs\=15 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1 ue.reg_type\=metric' task_configs=cola.yaml output_dir=../workdir/run_train_models/metric_electra/cola/metric
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.02 +ue.lamb_intra\=0.02 training.learning_rate\=1e-05 +ue.margin\=0.5 training.num_train_epochs\=12 training.per_device_train_batch_size\=32 +training.weight_decay\=0.1 ue.reg_type\=metric' task_configs=sst2.yaml output_dir=../workdir/run_train_models/metric_electra/sst2/metric
# CONLL
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_conll2003.py task_configs=conll2003.yaml args='ue\=mc do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=True ue.lamb\=1.0 +ue.lamb_intra\=0.006 training.learning_rate\=0.0001 +ue.margin\=0.01 training.num_train_epochs\=8 training.per_device_train_batch_size\=16 +training.weight_decay\=0.1 ue.reg_type\=metric' seeds=[23419,705525,4837,10671619,1084218,43] cuda_devices=[0,1,2,3,4,5] output_dir='../workdir/run_train_models/metric_electra/conll/metric'

# train - metric loss, SN
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.1 +ue.lamb_intra\=1.0 training.learning_rate\=3e-05 +ue.margin\=0.1 training.num_train_epochs\=9 training.per_device_train_batch_size\=16 +training.weight_decay\=0.1 ue.reg_type\=metric +ue.use_spectralnorm\=True spectralnorm_layer\=last' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/metric_electra/mrpc/metric_sn
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.2 +ue.lamb_intra\=1.0 training.learning_rate\=3e-05 +ue.margin\=2.5 training.num_train_epochs\=14 training.per_device_train_batch_size\=64 +training.weight_decay\=0.1 ue.reg_type\=metric +ue.use_spectralnorm\=True spectralnorm_layer\=last' task_configs=cola.yaml output_dir=../workdir/run_train_models/metric_electra/cola/metric_sn
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.001 +ue.lamb_intra\=0.02 training.learning_rate\=3e-05 +ue.margin\=2.5 training.num_train_epochs\=4 training.per_device_train_batch_size\=32 +training.weight_decay\=0.01 ue.reg_type\=metric +ue.use_spectralnorm\=True spectralnorm_layer\=last' task_configs=sst2.yaml output_dir=../workdir/run_train_models/metric_electra/sst2/metric_sn
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_conll2003.py task_configs=conll2003.yaml args='ue\=mc do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=True ue.lamb\=0.2 +ue.lamb_intra\=0.006 training.learning_rate\=2e-05 +ue.margin\=0.25 training.num_train_epochs\=7 training.per_device_train_batch_size\=4 +training.weight_decay\=0 ue.reg_type\=metric +ue.use_spectralnorm\=True spectralnorm_layer\=last' seeds=[23419,705525,4837,10671619,1084218,43] cuda_devices=[0,1,2,3,4,5] output_dir='../workdir/run_train_models/metric_electra/conll/metric_sn'












#HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py script=run_conll2003.py config_path=../configs/conll2003.yaml model_series_dir='../workdir/run_train_models/metric_electra/conll/metric/models/conll2003/' args='ue\=mc ue.calibrate\=False ue.use_cache\=True do_ue_estimate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 +ue.margin\=0.25 +ue.lamb_intra\=2.5' cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/last'
# all

HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py script=run_conll2003.py config_path=../configs/conll2003.yaml model_series_dir='../workdir/run_train_models/metric_electra/conll/metric/models/conll2003/' args='ue\=mc ue.calibrate\=False ue.use_cache\=True do_ue_estimate\=True ue.dropout_subs\=all data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 +ue.margin\=0.25 +ue.lamb_intra\=2.5' cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/all'


# DPP 2, metric model
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py script=run_conll2003.py config_path=../configs/conll2003.yaml model_series_dir='../workdir/run_train_models/metric_electra/conll/metric/models/conll2003/' args='ue\=mc-dpp ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.dropout.use_ood_sampling\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 +ue.margin\=0.25 +ue.lamb_intra\=2.5 ue.dropout.max_frac\=0.3 ue.dropout.committee_size\=20 ue.committee_size\=100' cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/dpp'
# DPP with ood, metric model
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py script=run_conll2003.py config_path=../configs/conll2003.yaml model_series_dir='../workdir/run_train_models/metric_electra/conll/metric/models/conll2003/' args='ue\=mc-dpp ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.dropout.use_ood_sampling\=True ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 +ue.margin\=0.25 +ue.lamb_intra\=2.5 ue.dropout.max_frac\=0.3 ue.dropout.committee_size\=20 ue.committee_size\=100' cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/dpp_with_ood'
# There is some problem
# Maha
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py script=run_conll2003.py config_path=../configs/conll2003.yaml model_series_dir='../workdir/run_train_models/metric_electra/conll/metric/models/conll2003/' args='ue\=mahalanobis ue.calibrate\=False ue.use_cache\=True do_ue_estimate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.25 ue.lamb_intra\=2.5' cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/maha'
# Maha SN
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py script=run_conll2003.py config_path=../configs/conll2003.yaml model_series_dir='../workdir/run_train_models/metric_electra/conll/metric_sn/models/conll2003/' args='ue\=mahalanobis ue.calibrate\=False ue.use_cache\=True do_ue_estimate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.5 ue.lamb_intra\=0.025 ue.use_spectralnorm\=True spectralnorm_layer\=last' cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric_sn/maha'

#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/last' output_dir='../workdir/run_calc_ues_metrics/metric_electra/conll/metric/last'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/all' output_dir='../workdir/run_calc_ues_metrics/metric_electra/conll/metric/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/dpp' output_dir='../workdir/run_calc_ues_metrics/metric_electra/conll/metric/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/dpp_with_ood' output_dir='../workdir/run_calc_ues_metrics/metric_electra/conll/metric/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric/maha' output_dir='../workdir/run_calc_ues_metrics/metric_electra/conll/metric/maha'
# SN

HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_electra/conll/metric_sn/maha' output_dir='../workdir/run_calc_ues_metrics/metric_electra/conll/metric_sn/maha'



# Estimate
# Last
#HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_electra/mrpc/metric/models/mrpc/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/last'
#HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_electra/cola/metric/models/cola/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/cola/last'
#HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_electra/sst2/metric/models/sst2/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/last'

# all
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_electra/mrpc/metric/models/mrpc/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=all ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/all'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_electra/cola/metric/models/cola/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=all ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/cola/all'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_electra/sst2/metric/models/sst2/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=all ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/all'

# DPP 2
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_electra/mrpc/metric/models/mrpc/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.5' output_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/dpp'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_electra/cola/metric/models/cola/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.3' output_dir='../workdir/run_glue_for_model_series/metric_electra/cola/dpp'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_electra/sst2/metric/models/sst2/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.5' output_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/dpp'

# DPP with OOD
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_electra/mrpc/metric/models/mrpc/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric ue.dropout.use_ood_sampling\=True ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.4' output_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_electra/cola/metric/models/cola/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric ue.dropout.use_ood_sampling\=True ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.6' output_dir='../workdir/run_glue_for_model_series/metric_electra/cola/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_electra/sst2/metric/models/sst2/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric ue.dropout.use_ood_sampling\=True ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.6' output_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/dpp_with_ood'

# maha
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_electra/mrpc/metric/models/mrpc/ args='ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.lamb\=1.0 ue.margin\=5.0 ue.lamb_intra\=0.1 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/maha'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_electra/cola/metric/models/cola/ args='ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.006 ue.margin\=0.025 ue.lamb_intra\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/cola/maha'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_electra/sst2/metric/models/sst2/ args='ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.008 ue.margin\=0.25 ue.lamb_intra\=0.01 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/maha'


# Now estimate maha with SN model
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_electra/mrpc/metric_sn/models/mrpc/ args='ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.01 ue.margin\=0.25 ue.lamb_intra\=2.5 ue.reg_type\=metric ue.use_spectralnorm\=True spectralnorm_layer\=last' output_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/maha_sn'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_electra/cola/metric_sn/models/cola/ args='ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.006 ue.margin\=0.5 ue.lamb_intra\=2.5 ue.reg_type\=metric ue.use_spectralnorm\=True spectralnorm_layer\=last' output_dir='../workdir/run_glue_for_model_series/metric_electra/cola/maha_sn'
HYDRA_CONFIG_PATH=../configs/run_tasks_for_model_series.yaml python ./run_tasks_for_model_series.py cuda_devices=[0,1,2,3,4,5] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_electra/sst2/metric_sn/models/sst2/ args='ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=False data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.006 ue.margin\=5.0 ue.lamb_intra\=0.01 ue.reg_type\=metric ue.use_spectralnorm\=True spectralnorm_layer\=last' output_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/maha_sn'



# Finally, calc metrics
# last
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/last' output_dir='../workdir/run_calc_ues_metrics/metric_electra/mrpc/last'
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/cola/last' output_dir='../workdir/run_calc_ues_metrics/metric_electra/cola/last'
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/last' output_dir='../workdir/run_calc_ues_metrics/metric_electra/sst2/last'
# all
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/all' output_dir='../workdir/run_calc_ues_metrics/metric_electra/mrpc/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/cola/all' output_dir='../workdir/run_calc_ues_metrics/metric_electra/cola/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/all' output_dir='../workdir/run_calc_ues_metrics/metric_electra/sst2/all'
# dpp2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/dpp' output_dir='../workdir/run_calc_ues_metrics/metric_electra/mrpc/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/cola/dpp' output_dir='../workdir/run_calc_ues_metrics/metric_electra/cola/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/dpp' output_dir='../workdir/run_calc_ues_metrics/metric_electra/sst2/dpp'
# dpp with ood
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/dpp_with_ood' output_dir='../workdir/run_calc_ues_metrics/metric_electra/mrpc/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/cola/dpp_with_ood' output_dir='../workdir/run_calc_ues_metrics/metric_electra/cola/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/dpp_with_ood' output_dir='../workdir/run_calc_ues_metrics/metric_electra/sst2/dpp_with_ood'
# maha
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/maha' output_dir='../workdir/run_calc_ues_metrics/metric_electra/mrpc/maha'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/cola/maha' output_dir='../workdir/run_calc_ues_metrics/metric_electra/cola/maha'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/maha' output_dir='../workdir/run_calc_ues_metrics/metric_electra/sst2/maha'
# sn maha
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/mrpc/maha_sn' output_dir='../workdir/run_calc_ues_metrics/metric_electra/mrpc/maha_sn'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/cola/maha_sn' output_dir='../workdir/run_calc_ues_metrics/metric_electra/cola/maha_sn'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_electra/sst2/maha_sn' output_dir='../workdir/run_calc_ues_metrics/metric_electra/sst2/maha_sn'
