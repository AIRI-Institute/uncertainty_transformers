cd ../../src
# Also add CoNLL-2003 - metric

# Script for training all models on MRPC, COLA, SST2-10% and CoNLL-2003 for new metric loss with 3 params
# train - metric loss, no SN
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric training.learning_rate\=3e-5 training.num_train_epochs\=9 training.per_device_train_batch_size\=8 +training.weight_decay\=0.1' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric

HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric training.learning_rate\=5e-6 training.num_train_epochs\=12 training.per_device_train_batch_size\=32 +training.weight_decay\=0' task_configs=cola.yaml output_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric training.learning_rate\=2e-5 training.num_train_epochs\=9 training.per_device_train_batch_size\=16 +training.weight_decay\=0.01' task_configs=sst2.yaml output_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric
# CONLL
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective_paper.yaml args='do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.25 ue.lamb_intra\=2.5 training.learning_rate\=3e-5 training.num_train_epochs\=12 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1' seeds=[23419,705525,4837,10671619,1084218,43] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric'

# train - metric loss, SN
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.01 +ue.margin\=0.25 +ue.lamb_intra\=2.5 ue.reg_type\=metric +ue.use_spectralnorm\=True spectralnorm_layer\=last training.learning_rate\=7e-5 training.num_train_epochs\=12 training.per_device_train_batch_size\=32 +training.weight_decay\=0.0' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric_sn
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.5 +ue.lamb_intra\=2.5 ue.reg_type\=metric +ue.use_spectralnorm\=True spectralnorm_layer\=last training.learning_rate\=6e-6 training.num_train_epochs\=14 training.per_device_train_batch_size\=64 +training.weight_decay\=0.01' task_configs=cola.yaml output_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric_sn
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=5.0 +ue.lamb_intra\=0.01 ue.reg_type\=metric +ue.use_spectralnorm\=True spectralnorm_layer\=last training.learning_rate\=7e-6 training.num_train_epochs\=9 training.per_device_train_batch_size\=32 +training.weight_decay\=0.01' task_configs=sst2.yaml output_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric_sn
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective_paper.yaml args='do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.5 ue.lamb_intra\=0.025 +ue.use_spectralnorm\=True +spectralnorm_layer\=last training.learning_rate\=2e-5 training.num_train_epochs\=15 training.per_device_train_batch_size\=4 +training.weight_decay\=0.1' seeds=[23419,705525,4837,10671619,1084218,43] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric_sn'



HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric/models/conll2003_selective_paper/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.25 ue.lamb_intra\=2.5' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/last'
# all
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric/models/conll2003_selective_paper/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.dropout_subs\=all data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.25 ue.lamb_intra\=2.5' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/all'
# DPP 2, metric model
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric/models/conll2003_selective_paper/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.25 ue.lamb_intra\=2.5 ue.committee_size\=100 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/dpp'
# DPP with ood, metric model
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric/models/conll2003_selective_paper/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=True ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=True ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.25 ue.lamb_intra\=2.5 ue.committee_size\=100 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/dpp_with_ood'
# SMD - Maha and MC Maha in one run
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric/models/conll2003_selective_paper/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.ue_type\=mc_maha ue.dropout_subs\=all data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.25 ue.lamb_intra\=2.5' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/maha_mc'
# SMD - Maha and MC Maha in one run
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/metric_opt_electra_3hyp/conll/metric_sn/models/conll2003_selective_paper/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.ue_type\=mc_maha ue.dropout_subs\=all data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.reg_type\=metric ue.use_selective\=True ue.lamb\=0.1 ue.margin\=0.5 ue.lamb_intra\=0.025 +ue.use_spectralnorm\=True +spectralnorm_layer\=last' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric_sn/maha_mc'

HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/last/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/conll/metric/last'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/all/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/conll/metric/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/dpp/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/conll/metric/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/conll/metric/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/conll/metric/maha_mc'
# SN

HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/metric_opt_electra_3hyp/conll/metric_sn/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/conll/metric_sn/maha_mc'



# Also add DPP and DPP with OOD for raw and CER on CONLL
# DPP 2, raw model
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_opt_hyp/raw/models/conll2003_selective_paper/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.committee_size\=100 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/raw/dpp'
# DPP with ood, raw model
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_opt_hyp/raw/models/conll2003_selective_paper/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=True ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=True ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.committee_size\=100 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/raw/dpp_with_ood'
# DPP 2, reg model
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_opt_hyp/reg/models/conll2003_selective_paper/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=True ue.lamb\=1e-3 ue.committee_size\=100 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/reg/dpp'
# DPP with ood, reg model
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective_paper.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_opt_hyp/reg/models/conll2003_selective_paper/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=True ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=True ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 ue.use_selective\=True ue.lamb\=1e-3 ue.committee_size\=100 ue.dropout.max_frac\=0.3 ue.dropout.mask_name_for_mask\=rbf' cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/reg/dpp_with_ood'

HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/raw/dpp/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_opt_hyp/raw/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/raw/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_opt_hyp/raw/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/reg/dpp/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_opt_hyp/reg/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_opt_hyp/reg/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_opt_hyp/reg/dpp_with_ood'

# Estimate
# TODO - 3 datasets, last|all|DPP|DPP OOD|mc_maha
# Last
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric/models/mrpc/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/last'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric/models/cola/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/last'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric/models/sst2/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/last'

# all
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric/models/mrpc/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=all ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/all'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric/models/cola/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=all ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/all'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric/models/sst2/ args='ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=all ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/all'

# DPP 2
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric/models/mrpc/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.5' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/dpp'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric/models/cola/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.3' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/dpp'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric/models/sst2/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.5' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/dpp'

# DPP with OOD
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric/models/mrpc/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric ue.dropout.use_ood_sampling\=True ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.4' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric/models/cola/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric ue.dropout.use_ood_sampling\=True ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.6' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric/models/sst2/ args='ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric ue.dropout.use_ood_sampling\=True ue.dropout.committee_size\=50 ue.committee_size\=100 ue.dropout.max_frac\=0.6' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/dpp_with_ood'

# MC maha
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric/models/mrpc/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=1.0 +ue.margin\=5.0 +ue.lamb_intra\=0.1 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric/models/cola/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.006 +ue.margin\=0.025 +ue.lamb_intra\=0.05 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric/models/sst2/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.008 +ue.margin\=0.25 +ue.lamb_intra\=0.01 ue.reg_type\=metric' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/maha_mc'


# Now estimate mc_maha with SN model
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/mrpc.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/mrpc/metric_sn/models/mrpc/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.01 +ue.margin\=0.25 +ue.lamb_intra\=2.5 ue.reg_type\=metric ue.use_spectralnorm\=True spectralnorm_layer\=last' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/cola.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/cola/metric_sn/models/cola/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.006 +ue.margin\=0.5 +ue.lamb_intra\=2.5 ue.reg_type\=metric ue.use_spectralnorm\=True spectralnorm_layer\=last' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py cuda_devices=[1,2,3] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/sst2.yaml model_series_dir=../workdir/run_train_models/metric_opt_electra_3hyp/sst2/metric_sn/models/sst2/ args='ue\=mc_mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.lamb\=0.006 +ue.margin\=5.0 +ue.lamb_intra\=0.01 ue.reg_type\=metric ue.use_spectralnorm\=True spectralnorm_layer\=last' output_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/maha_sn_mc'



# Finally, calc metrics
# last
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/last/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/mrpc/last'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/last/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/cola/last'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/last/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/sst2/last'
# all
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/all/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/mrpc/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/all/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/cola/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/all/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/sst2/all'
# dpp2
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/dpp/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/mrpc/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/dpp/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/cola/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/dpp/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/sst2/dpp'
# dpp with ood
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/mrpc/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/cola/dpp_with_ood'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/sst2/dpp_with_ood'
# mc maha
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/mrpc/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/cola/maha_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/maha_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/sst2/maha_mc'
# sn mc maha
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/mrpc/maha_sn_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/mrpc/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/cola/maha_sn_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/cola/maha_sn_mc'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py runs_dir='../workdir/run_glue_for_model_series/metric_opt_electra_3hyp/sst2/maha_sn_mc/results' output_dir='../workdir/run_calc_ues_metrics/metric_opt_electra_3hyp/sst2/maha_sn_mc'
