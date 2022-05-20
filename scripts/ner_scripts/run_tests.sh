# exit when any command fails
set -e
cd ../src

CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_no_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=decomposing_md do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_sn/conll2003/decomposing_md
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_no_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=nuq do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_sn/conll2003/nuq
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_no_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mahalanobis do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=True data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_sn/conll2003/mahalanobis
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_no_sn/conll2003/mc
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_metric_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_metric_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=metric ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_metric_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_reg_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_reg_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=True ue.reg_type\=reg-curr ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_reg_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='microsoft/deberta-base' output_dir=../workdir/run_tests/deberta_raw_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=mc-dpp do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='distilbert-base-cased' output_dir=../workdir/run_tests/distilbert_raw_no_sn/conll2003/mc-dpp
CUDA_VISIBLE_DEVICES=2 HYDRA_CONFIG_PATH=../configs/conll2003.yaml python run_conll2003.py ue\=sngp do_ue_estimate\=True ue.use_selective\=False ue.reg_type\=raw ++ue.use_spectralnorm\=False data.subsample_perc_val\=0.01 data.subsample_perc\=0.01 training\=electra_base model.model_name_or_path\='google/electra-base-discriminator' output_dir=../workdir/run_tests/electra_raw_no_sn/conll2003/sngp