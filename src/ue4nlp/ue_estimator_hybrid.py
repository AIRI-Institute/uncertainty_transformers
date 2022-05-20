import torch
import numpy as np
from tqdm import tqdm
import time

from utils.utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)
from utils.utils_inference import (
    is_custom_head,
    unpad_features,
    pad_scores
)
from ue4nlp.mahalanobis_distance import (
    mahalanobis_distance,
    mahalanobis_distance_relative,
    mahalanobis_distance_marginal,
    compute_centroids,
    compute_covariance
)

import copy
from scipy.stats import rankdata
import logging

log = logging.getLogger()

def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)

def deepfool(x, net, max_iter=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device)
    x = torch.from_numpy(x).to(device)
        
    x_pert = torch.clone(x).detach().to(device)
    x_pert.requires_grad_()
    
    preds_orig = net(x)[0]
    num_classes = preds_orig.shape[0]
    label = preds_orig.data.cpu().numpy().flatten().argmax()

    input_shape = x.detach().cpu().numpy().shape
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    preds_pert = net(x_pert)[0]
    k_i = label
    
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        preds_pert[label].backward(retain_graph=True)
        grad_orig = x_pert.grad.data.cpu().numpy().copy()
        
        for k in range(num_classes):
            if k == label:
                continue
                
            x_pert.grad.data.zero_()

            preds_pert[k].backward(retain_graph=True)
            cur_grad = x_pert.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (preds_pert[k] - preds_pert[label]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        x_pert = x.to(device) + torch.from_numpy(r_tot).to(device)
        x_pert = torch.clone(x_pert).detach().to(device)
        x_pert.requires_grad_()
        
        preds_pert = net(x_pert)[0]
        k_i = np.argmax(preds_pert.data.cpu().numpy().flatten())

        loop_i += 1
        
    return (r_tot * r_tot).sum()

class UeEstimatorHybrid:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset
        
    def __call__(self, X, y):
        return self._predict_with_fitted_cov(X, y)
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Start fitting covariance and centroids **************")
        
        if y is None:
            y = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features = self._exctract_features(X)
            
        self.class_cond_centroids = self._fit_centroids(X_features, y)
        self.class_cond_covarince = self._fit_covariance(X_features, y)
    
        log.info("**************Done.**********************")
        
    def _fit_covariance(self, X, y, class_cond=True):
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)
        
    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)
      
    def _replace_model_head(self): 
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        self.old_head = copy.deepcopy(model.classifier)
        
        if is_custom_head(model):
            model.classifier = ElectraClassificationHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)
        
    def _return_head(self):
        self.cls._auto_model.classifier = self.old_head
        log.info("Change Identity Pooler to classifier")
    
    def _exctract_labels(self, X):    
        return np.asarray([example["label"] for example in X])
    
    def _exctract_features(self, X):
        cls = self.cls
        model = self.cls._auto_model
        
        try:
            X = X.remove_columns("label")
        except:
            X.dataset = X.dataset.remove_columns("label")
            
        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        return X_features
        
    def _predict_with_fitted_cov(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute MD with fitted covariance and centroids **************")

        if self.ue_args.id_uncertainty == 'entropy':
            self._return_head()
            preds, probs = cls.predict(X, apply_softmax=False, return_preds=False)[:2]
            id_uncertainty = entropy(probs)
            self._replace_model_head()
        elif self.ue_args.id_uncertainty == 'adversarial':
            log.info("****************Compute DeepFool dists**************")
            head_copy = copy.deepcopy(self.old_head)
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)
            X_encoder_features = self._exctract_features(X)
            
            self._return_head()
            self._replace_model_head()
            
            id_uncertainty = np.zeros(X_encoder_features.shape[0])
            for i, x in tqdm(enumerate(X_encoder_features)):
                id_uncertainty[i] = deepfool(x[None,:,:], head_copy)
            log.info("****************Done.**************")
                    
        self.md_threshold = self.ue_args.md_threshold if 'md_threshold' in self.ue_args.keys() else 0.2
        
        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        
        
        end = time.time()
        
        eval_results = {}
        
        md, inf_time = mahalanobis_distance(None, None, X_features, 
                                            self.class_cond_centroids, self.class_cond_covarince)
        
        sum_inf_time = inf_time + (end - start)
        
        n_preds = len(y)
        n_lowest = int(n_preds*self.md_threshold)
        
        md_rank = rankdata(md)
        id_uncertainty_rank = rankdata(id_uncertainty[md_rank < n_lowest])
        md_rank[md_rank < n_lowest] = id_uncertainty_rank
        
        
        eval_results["mahalanobis_distance"] = md_rank.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
    
        log.info("**************Done.**********************")
        return eval_results