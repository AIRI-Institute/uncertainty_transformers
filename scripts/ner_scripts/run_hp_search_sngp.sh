cd ../../src
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python ./run_conll2003_with_hyp_search.py ue\=sngp do_ue_estimate\=False ue.use_selective\=False ue.reg_type\=raw training\=electra_base hydra.run.dir=../workdir/hp_search/electra_raw_sngp/conll2003 model.model_name_or_path='google/electra-base-discriminator'
wait