cd ../../src
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=../configs/clinc.yaml python ./run_ood_with_hyp_search.py ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw training\=electra_base hydra.run.dir=../workdir/hp_search/electra_raw_sngp/clinc &
CUDA_VISIBLE_DEVICES=1 HYDRA_CONFIG_PATH=../configs/clinc.yaml python ./run_ood_with_hyp_search.py ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw training\=electra_base hydra.run.dir=../workdir/hp_search/bert_raw_sngp/clinc model.model_name_or_path='bert-base-uncased'