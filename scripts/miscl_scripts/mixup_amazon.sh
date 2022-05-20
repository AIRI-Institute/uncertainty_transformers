cd ../../src

# Script for training all models on amazon for mixup with XLNet model
#HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_newsgroup.py cuda_devices=[0,1,2] seeds=[10671619,1084218,43] args='ue\=mc do_ue_estimate\=False ue.use_selective\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last ue.use_cache\=False model.model_name_or_path\=xlnet-base-cased training.num_train_epochs\=4 training.learning_rate\=1e-4 training.per_device_train_batch_size\=32 data.max_seq_length\=512' task_configs=amazon_mixup.yaml output_dir=../workdir/run_train_models/mixup_xlnet_test_last/amazon_512


# Test - make models with last/all MC dropout
# MSD - for XLNet exps we set  ue.inference_prob\=0.3 ue.committee_size\=100, as in paper
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py script=run_newsgroup.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/amazon_mixup.yaml model_series_dir=../workdir/run_train_models/mixup_xlnet_test_last/amazon/models/amazon_mixup/ args='ue\=msd do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.use_cache\=False ue.inference_prob\=0.3 ue.committee_size\=100' output_dir='../workdir/run_glue_for_model_series/mixup_xlnet_test_last/amazon/msd/all'
#HYDRA_CONFIG_PATH=../configs/run_glue_for_model_series.yaml python ./run_glue_for_model_series.py script=run_newsgroup.py cuda_devices=[0,1,2] seeds=[23419,705525,4837,10671619,1084218,43] config_path=../configs/amazon_mixup.yaml model_series_dir=../workdir/run_train_models/mixup_xlnet_test_last/amazon_512/models/amazon_mixup/ args='ue\=msd do_ue_estimate\=True ue.use_selective\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.use_cache\=False ue.dropout_subs\=last ue.inference_prob\=0.3 ue.committee_size\=100 data.max_seq_length\=512' output_dir='../workdir/run_glue_for_model_series/mixup_xlnet_test_last/amazon_512/msd/last'

# Finally, calc metrics
#HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py table_metrics=['table_f1_micro','table_f1_macro'] runs_dir='../workdir/run_glue_for_model_series/mixup_xlnet_test_last/amazon/msd/all/results' output_dir='../workdir/run_calc_ues_metrics/mixup_xlnet_test_last/amazon/msd/all'
HYDRA_CONFIG_PATH=../configs/run_calc_ues_metrics.yaml python ./run_calc_ues_metrics.py table_metrics=['table_f1_micro','table_f1_macro'] runs_dir='../workdir/run_glue_for_model_series/mixup_xlnet_test_last/amazon_512/msd/last/results' output_dir='../workdir/run_calc_ues_metrics/mixup_xlnet_test_last/amazon_512/msd/last'
