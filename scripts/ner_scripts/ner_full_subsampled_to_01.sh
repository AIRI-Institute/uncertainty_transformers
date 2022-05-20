# Test script - run it to ensure that all working fine in your environment
cd ../../src

# MC last, raw model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_raw_01/last'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_raw_01/last/models/conll2003_selective/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/last'

# MC all, raw model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.dropout_subs\=all data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_raw_01/all'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_raw_01/all/models/conll2003_selective/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.dropout_subs\=all data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/all'

# DPP 2, raw model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_raw_01/dpp'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_raw_01/dpp/models/conll2003_selective/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=False ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/dpp'

# DPP with ood, raw model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=True ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_raw_01/dpp_with_ood'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_raw_01/dpp_with_ood/models/conll2003_selective/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=True ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=True ue.calibrate\=True data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/dpp_with_ood'



# MC last, reg-curr model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/last'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/last/models/conll2003_selective/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/last'

# MC all, reg-curr model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.dropout_subs\=all ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/all'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/all/models/conll2003_selective/' args='ue.calibrate\=True ue.use_cache\=True do_ue_estimate\=True ue.dropout_subs\=all ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/all'

# DPP 2, reg-curr model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=False ue.calibrate\=True ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/dpp'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/dpp/models/conll2003_selective/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=False ue.calibrate\=True ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/dpp'

# DPP with ood, reg-curr model
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models_ner.py task_configs=conll2003_selective.yaml args='do_ue_estimate\=False ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=False ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=True ue.calibrate\=True ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' seeds=[17,42,51] cuda_devices=[1,2,3] output_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/dpp_with_ood'
# after run ue part
HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_ner_for_model_series.py config_path=../configs/conll2003_selective.yaml model_series_dir='../workdir/run_train_models/conll2003_electra_reg_01_fix/dpp_with_ood/models/conll2003_selective/' args='ue.use_cache\=True do_ue_estimate\=True ue.dropout.is_reused_mask\=True ue.use_ood_sampling\=True ue.ue_type\=mc-dpp ue.dropout_type\=DPP +ue.dropout.use_ood_sampling\=True ue.calibrate\=True ue.use_selective\=True ue.lamb\=0.05 data.subsample_perc_val\=0.1 data.subsample_perc\=0.1' cuda_devices=[1,2,3] seeds=[1,2,4,5,7] output_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/dpp_with_ood'
# and finally print obtained results
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/last/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_raw_01/last'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/all/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_raw_01/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/dpp/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_raw_01/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_raw_01/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_raw_01/dpp_with_ood'

HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/last/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_reg_01_fix/last'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/all/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_reg_01_fix/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/dpp/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_reg_01_fix/dpp'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics_ner.py runs_dir='../workdir/run_ner_for_model_series/conll2003_electra_reg_01_fix/dpp_with_ood/results' output_dir='../workdir/run_calc_ues_metrics/conll2003_electra_reg_01_fix/dpp_with_ood'