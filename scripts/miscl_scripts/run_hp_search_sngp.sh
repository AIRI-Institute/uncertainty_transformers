cd ../../src
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=../configs/mrpc.yaml python ./run_glue_with_hyp_search.py ue\=sngp do_ue_estimate\=False ue.use_selective\=False ue.reg_type\=raw training\=electra_base hydra.run.dir=../workdir/hp_search/electra_raw_sngp/mrpc model.model_name_or_path='google/electra-base-discriminator' &
CUDA_VISIBLE_DEVICES=1 HYDRA_CONFIG_PATH=../configs/cola.yaml python ./run_glue_with_hyp_search.py ue\=sngp do_ue_estimate\=False ue.use_selective\=False ue.reg_type\=raw training\=electra_base hydra.run.dir=../workdir/hp_search/electra_raw_sngp/cola model.model_name_or_path='google/electra-base-discriminator'
wait
CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG_PATH=../configs/sst2.yaml python ./run_glue_with_hyp_search.py ue\=sngp do_ue_estimate\=False ue.use_selective\=False ue.reg_type\=raw training\=electra_base hydra.run.dir=../workdir/hp_search/electra_raw_sngp/sst2 model.model_name_or_path='google/electra-base-discriminator' &