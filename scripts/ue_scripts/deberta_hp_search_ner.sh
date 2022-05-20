cd ../../src
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python ./run_conll2003_with_hyp_search.py ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.reg_type\=metric +ue.use_spectralnorm\=False training\=electra_base hydra.run.dir=../workdir/hp_search/deberta_metric_no_sn/conll2003 model.model_name_or_path='microsoft/deberta-base'
wait
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python ./run_conll2003_with_hyp_search.py ue\=mc do_ue_estimate\=False ue.use_selective\=True ue.reg_type\=metric +ue.use_spectralnorm\=True training\=electra_base hydra.run.dir=../workdir/hp_search/deberta_metric_sn/conll2003 model.model_name_or_path='microsoft/deberta-base'
wait