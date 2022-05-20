cd ../../src

# Maha sn
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.ue_type\=maha ue.dropout_subs\=last data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 +ue.use_spectralnorm\=True +spectralnorm_layer\=last' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_sn_last_01/maha'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_sn_last_01/maha/models/conll2003_selective/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.ue_type\=maha ue.dropout_subs\=last data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 +ue.use_spectralnorm\=True +spectralnorm_layer\=last' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_sn_last_01/maha'

HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_sn_last_01/maha/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_sn_last_01/maha'

# Maha sn + reg
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.ue_type\=maha ue.dropout_subs\=last data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 +ue.use_spectralnorm\=True +spectralnorm_layer\=last ue.use_selective\=True ue.lamb\=0.05' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_sn_last_reg_01/maha'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_sn_last_reg_01/maha/models/conll2003_selective/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.ue_type\=maha ue.dropout_subs\=last data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 +ue.use_spectralnorm\=True +spectralnorm_layer\=last ue.use_selective\=True ue.lamb\=0.05' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_sn_last_reg_01/maha'

HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_sn_last_reg_01/maha/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_sn_last_reg_01/maha'
