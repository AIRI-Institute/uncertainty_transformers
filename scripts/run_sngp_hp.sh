CUDA_VISIBLE_DEVICES=1 HYDRA_CONFIG_PATH=../configs/mrpc.yaml nohup python run_glue_with_hyp_search.py ue=sngp do_eval=True do_ue_estimate=False ue.calibrate=False data.validation_subsample=0.0 ue.use_selective=False output_dir=../workdir/sngp/ &
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/cola.yaml nohup python run_glue_with_hyp_search.py ue=sngp do_eval=True do_ue_estimate=False ue.calibrate=False data.validation_subsample=0.0 ue.use_selective=False output_dir=../workdir/sngp/ &
CUDA_VISIBLE_DEVICES=3 HYDRA_CONFIG_PATH=../configs/sst2.yaml nohup python run_glue_with_hyp_search.py ue=sngp do_eval=True do_ue_estimate=False ue.calibrate=False data.validation_subsample=0.1 ue.use_selective=False output_dir=../workdir/sngp/ &