# Uncertainty Estimation
There is a repository for Uncertainty Estimation (UE) for NLP tasks, such as classification and NER.

# Implemented UE Methods
Full code for all methods could be found in [src/ue4nlp](src/ue4nlp). For now following methods are implemented:
1. [Monte-Carlo Dropout (MC Dropout)](src/ue4nlp/ue_estimator_mc.py)
2. [Diverse Determinantal Point Process Monte Carlo Dropout (DDPP MC Dropout) with two strategies: DPP and OOD](src/ue4nlp/dropout_dpp.py)
3. [Spectral-normalized Neural Gaussian Process (SNGP)](src/ue4nlp/ue_estimator_sngp.py)
4. [Mahalanobis Distance (MD)](src/ue4nlp/ue_estimator_mahalanobis.py)
5. [Mahalanobis Distance with Spectral-normalized Network (MD SN)](src/ue4nlp/ue_estimator_mahalanobis.py)
6. [Confident Error Regularizer (CER)](src/ue4nlp/transformers_regularized.py)
7. [Metric Regularizer](src/ue4nlp/transformers_regularized.py)
8. [Deep Ensemble](src/run_train_ensemble_series.py)
9. [MSD](src/ue4nlp/transformers_mixup.py)

Some of these methods rely on regularization, others are either deterministic or rely on widely used MC Dropout.

# Models and Datasets
All aforementioned UE methods were tested with ELECTRA and DeBERTa models on MRPC, CoLA, SST2 and CoNLL-2003. Moreover, all these methods could be easily used with other models or datasets from [Hugging Face Transformers/Datasets](https://huggingface.co/) libraries. To use one of the methods with another model, one could reuse the following estimator from [src/ue4nlp](src/ue4nlp).

# Examples
Example scripts for training models and calculating their UE scores could be found in the [scripts](scripts) directory. This directory also contains instructions for reproducing results from the paper.

Configuration files with parameters of models, datasets and uncertainty estimation methods are located in [configs](configs).
# Citation
```bibtex
@inproceedings{vazhentsev-etal-2022-uncertainty,
    title = "Uncertainty Estimation of Transformer Predictions for Misclassification Detection",
    author = "Vazhentsev, Artem  and
      Kuzmin, Gleb  and
      Shelmanov, Artem  and
      Tsvigun, Akim  and
      Tsymbalov, Evgenii  and
      Fedyanin, Kirill  and
      Panov, Maxim  and
      Panchenko, Alexander  and
      Gusev, Gleb  and
      Burtsev, Mikhail  and
      Avetisian, Manvel  and
      Zhukov, Leonid",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.566",
    pages = "8237--8252",
    abstract = "Uncertainty estimation (UE) of model predictions is a crucial step for a variety of tasks such as active learning, misclassification detection, adversarial attack detection, out-of-distribution detection, etc. Most of the works on modeling the uncertainty of deep neural networks evaluate these methods on image classification tasks. Little attention has been paid to UE in natural language processing. To fill this gap, we perform a vast empirical investigation of state-of-the-art UE methods for Transformer models on misclassification detection in named entity recognition and text classification tasks and propose two computationally efficient modifications, one of which approaches or even outperforms computationally intensive methods.",
}
```
# License
© 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI). All rights reserved.

Licensed under the [MIT License](LICENSE)
