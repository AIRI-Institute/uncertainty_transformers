cd ../../src


# CER models
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_glue_with_hyp_search.py cuda_devices=[1] seeds=[23419] args='ue\=mc do_ue_estimate\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/distilbert_hypopt/cer/mrpc
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_glue_with_hyp_search.py cuda_devices=[1] seeds=[23419] args='ue\=mc do_ue_estimate\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr' task_configs=cola.yaml output_dir=../workdir/run_train_models/distilbert_hypopt/cer/cola
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_glue_with_hyp_search.py cuda_devices=[1] seeds=[23419] args='ue\=mc do_ue_estimate\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr' task_configs=sst2.yaml output_dir=../workdir/run_train_models/distilbert_hypopt/cer/sst2

HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_conll2003_with_hyp_search.py task_configs=conll2003.yaml args='do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr' seeds=[23419] cuda_devices=[1] output_dir='../workdir/run_train_models/distilbert_hypopt/cer/conll'

# CER SN models
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_glue_with_hyp_search.py cuda_devices=[1] seeds=[23419] args='ue\=mc do_ue_estimate\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr +ue.use_spectralnorm\=True spectralnorm_layer\=last' task_configs=mrpc.yaml output_dir=../workdir/run_train_models/distilbert_hypopt/cer_sn/mrpc
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_glue_with_hyp_search.py cuda_devices=[1] seeds=[23419] args='ue\=mc do_ue_estimate\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr +ue.use_spectralnorm\=True spectralnorm_layer\=last' task_configs=cola.yaml output_dir=../workdir/run_train_models/distilbert_hypopt/cer_sn/cola
HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_glue_with_hyp_search.py cuda_devices=[1] seeds=[23419] args='ue\=mc do_ue_estimate\=False ue.calibrate\=True data.validation_subsample\=0.0 training\=electra_base ue.dropout_subs\=last +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr +ue.use_spectralnorm\=True spectralnorm_layer\=last' task_configs=sst2.yaml output_dir=../workdir/run_train_models/distilbert_hypopt/cer_sn/sst2

HYDRA_CONFIG_PATH=../configs/run_train_models.yaml python ./run_train_models.py script=run_conll2003_with_hyp_search.py task_configs=conll2003.yaml args='do_ue_estimate\=False data.subsample_perc_val\=0.1 data.subsample_perc\=0.1 +training.save_strategy\=no model.model_name_or_path\=distilbert-base-cased ue.use_selective\=True ue.reg_type\=reg-curr +ue.use_spectralnorm\=True spectralnorm_layer\=last' seeds=[23419] cuda_devices=[1] output_dir='../workdir/run_train_models/distilbert_hypopt/cer_sn/conll'
