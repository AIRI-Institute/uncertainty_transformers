from sklearn.metrics import accuracy_score
import numpy as np
import math
from torch.nn import functional as F
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import spectral_norm
from scipy.special import softmax

from utils.utils_inference import (
    is_custom_head,
    unpad_features,
    pad_scores
)

from utils.utils_heads import (
    ElectraClassificationHeadIdentityPooler,
    BertClassificationHeadIdentityPooler,
    ElectraNERHeadIdentityPooler,
)

import time
from tqdm import tqdm
import os

from ue4nlp.mahalanobis_distance import (
    mahalanobis_distance,
    mahalanobis_distance_relative,
    mahalanobis_distance_marginal,
    compute_centroids,
    compute_covariance
)

def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
import logging

log = logging.getLogger()

def iCE(d_disc, d_nondisc, labels, num_classes):
    y = F.one_hot(labels, num_classes=num_classes)
    disc_loss = -F.log_softmax(d_disc, dim=-1) * y
    nondisc_loss = F.log_softmax(d_nondisc, dim=-1) * y
    loss = (disc_loss.sum(axis=1) + nondisc_loss.sum(axis=1)).mean()
    return loss


class DecomposingModel(nn.Module):
    def __init__(self, train_features, train_labels, use_spectral_norm=False):
        super(DecomposingModel, self).__init__()
        self.train_features = train_features
        self.train_labels = train_labels
        self.input_dim = self.train_features.shape[1]
        self.output_dim = torch.unique(self.train_labels).shape[0]
        self.F_linear = nn.Linear(self.input_dim, self.input_dim)
        self.D_linear = nn.Linear(self.input_dim-self.output_dim, self.output_dim)
        if use_spectral_norm:
            self.F_linear = spectral_norm(self.F_linear)
            self.D_linear = spectral_norm(self.D_linear)

    def forward(self, features):
        lin_mapping = self.F_linear(features)
        d_disc = lin_mapping[:, :self.output_dim]
        d_nondisc = self.D_linear(lin_mapping[:, self.output_dim:])
        return d_disc, d_nondisc
    
    def fit(self, lr=1e-5, batch_size=128, n_epochs=5, verbose=True):

        train_dataset = TensorDataset(self.train_features, self.train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size) 
        
        params = list(self.F_linear.parameters()) + list(self.D_linear.parameters())

        optimizer = optim.Adam(params, lr=lr)
        
        log.info('********Start Training Decomposing Representations********')
        
        n_print = max(1, int(0.5*len(train_dataset)//batch_size)) if verbose else -1
        for epoch in range(n_epochs):
            running_loss = 0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                d_disc, d_nondisc = self.forward(inputs)
                loss = iCE(d_disc, d_nondisc, labels, self.output_dim)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % n_print == (n_print-1):
                    log.info(f'epoch: {epoch + 1}, step: {i + 1:5d}, loss: {running_loss / n_print:.3f}')
                    running_loss = 0.0

        log.info('********Finished Training********')
        return self


class UeEstimatorDecomposing:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset
        
    def __call__(self, X, y):
        return self._predict_with_trained_decomposition(X, y)
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Start fitting**************")
        
        if y is None:
            y = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features = self._exctract_features(X)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_features = torch.Tensor(X_features).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        seed_everything(self.config.training.seed)
        
        self.dec_model = DecomposingModel(X_features, y, use_spectral_norm=False).to(self.device)
        self.dec_model.fit(lr=self.ue_args.lr, batch_size=self.ue_args.batch_size, n_epochs=self.ue_args.n_epochs)
        
        X_d_disc, X_d_nondisc = self.dec_model(X_features)
        
        self.disc_class_cond_centroids = self._fit_centroids(X_d_disc.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.disc_class_cond_covariance = self._fit_covariance(X_d_disc.cpu().detach().numpy(), y.cpu().detach().numpy(), 
                                                              centroids=self.disc_class_cond_centroids)
        
        self.nondisc_class_cond_centroids = self._fit_centroids(X_d_nondisc.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.nondisc_class_cond_covariance = self._fit_covariance(X_d_nondisc.cpu().detach().numpy(), y.cpu().detach().numpy(),
                                                                 centroids=self.nondisc_class_cond_centroids)

        log.info("**************Done.**********************")
        
    def _fit_covariance(self, X, y, class_cond=True, centroids=None):
        if class_cond:
            if centroids is None:
                centroids = self.class_cond_centroids
            return compute_covariance(centroids, X, y, class_cond)
    
        if centroids is None:
            centroids = self.train_centroid
        return compute_covariance(centroids, X, y, class_cond)
        
    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)
        
    def _replace_model_head(self): 
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        
        if is_custom_head(model):
            model.classifier = ElectraClassificationHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)
        
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

        
    def _predict_with_trained_decomposition(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        if y is None:
            y = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        X_features = torch.Tensor(X_features).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        
        d_disc, d_nondisc = self.dec_model(X_features)
        end = time.time()
        
        d_disc_md, inf_time1 = mahalanobis_distance(None, None, d_disc.cpu().detach().numpy(), 
                                                    self.disc_class_cond_centroids, 
                                                    self.disc_class_cond_covariance)
        
        d_nondisc_md, inf_time2 = mahalanobis_distance(None, None, d_nondisc.cpu().detach().numpy(), 
                                                       self.nondisc_class_cond_centroids,
                                                       self.nondisc_class_cond_covariance)

        eval_results = {}
        
        sum_inf_time = (end - start) + inf_time1 + inf_time2
        eval_results["disc_md"] = d_disc_md.tolist()
        eval_results["nondisc_md"] = d_nondisc_md.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
    
        log.info("**************Done.**********************")
        return eval_results


class UeEstimatorDecomposingNer:
    def __init__(self, cls, ue_args, config, train_dataset):
        self.cls = cls
        self.ue_args = ue_args
        self.config = config
        self.train_dataset = train_dataset
        
    def __call__(self, X, y):
        return self._predict_with_trained_decomposition(X, y)
    
    def fit_ue(self, X, y=None, X_test=None):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Start fitting**************")
        
        if y is None:
            y, y_shape = self._exctract_labels(X)
            
        self._replace_model_head()
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_features = torch.Tensor(X_features).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        seed_everything(self.config.training.seed)
        
        self.dec_model = DecomposingModel(X_features, y, use_spectral_norm=False).to(self.device)
        self.dec_model.fit(lr=self.ue_args.lr, batch_size=self.ue_args.batch_size, n_epochs=self.ue_args.n_epochs)
        
        X_d_disc, X_d_nondisc = self.dec_model(X_features)
        
        self.disc_class_cond_centroids = self._fit_centroids(X_d_disc.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.disc_class_cond_covariance = self._fit_covariance(X_d_disc.cpu().detach().numpy(), y.cpu().detach().numpy(), 
                                                              centroids=self.disc_class_cond_centroids)
        
        self.nondisc_class_cond_centroids = self._fit_centroids(X_d_nondisc.cpu().detach().numpy(), y.cpu().detach().numpy())
        self.nondisc_class_cond_covariance = self._fit_covariance(X_d_nondisc.cpu().detach().numpy(), y.cpu().detach().numpy(),
                                                                 centroids=self.nondisc_class_cond_centroids)

        log.info("**************Done.**********************")
        
    def _fit_covariance(self, X, y, class_cond=True, centroids=None):
        if class_cond:
            if centroids is None:
                centroids = self.class_cond_centroids
            return compute_covariance(centroids, X, y, class_cond)
    
        if centroids is None:
            centroids = self.train_centroid
        return compute_covariance(centroids, X, y, class_cond)
        
    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)
        
    def _replace_model_head(self): 
        log.info("Change classifier to Identity Pooler")
        cls = self.cls
        model = self.cls._auto_model
        
        if is_custom_head(model):
            model.classifier = ElectraNERHeadIdentityPooler(model.classifier)
        else:
            model.classifier = BertClassificationHeadIdentityPooler(model.classifier)
            
    def _exctract_labels(self, X):    
        y = np.asarray([example["labels"] for example in X])
        y_shape = y.shape
        
        return y.reshape(-1), y_shape
    
    def _exctract_features(self, X, remove_col=True):
        cls = self.cls
        model = self.cls._auto_model
        
        if remove_col:
            try:
                X = X.remove_columns("labels")
            except:
                X.dataset = X.dataset.remove_columns("labels")
            
        X_features = cls.predict(X, apply_softmax=False, return_preds=False)[0]
        X_features = X_features.reshape(-1, X_features.shape[-1])
        
        return X_features
    
    def _predict_with_trained_decomposition(self, X, y):
        cls = self.cls
        model = self.cls._auto_model
        
        log.info("****************Compute MD with fitted covariance and centroids **************")

        start = time.time()
        y_pad, y_shape = self._exctract_labels(X)
        X_features = self._exctract_features(X)
        X_features, y = unpad_features(X_features, y_pad)
        X_features = torch.Tensor(X_features).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        
        d_disc, d_nondisc = self.dec_model(X_features)
        end = time.time()
        
        d_disc_md, inf_time1 = mahalanobis_distance(None, None, d_disc.cpu().detach().numpy(), 
                                                    self.disc_class_cond_centroids, 
                                                    self.disc_class_cond_covariance)
        
        d_nondisc_md, inf_time2 = mahalanobis_distance(None, None, d_nondisc.cpu().detach().numpy(), 
                                                       self.nondisc_class_cond_centroids,
                                                       self.nondisc_class_cond_covariance)
        
        d_disc_md = pad_scores(d_disc_md, np.asarray(y_pad).reshape(y_shape), y_pad)
        d_nondisc_md = pad_scores(d_nondisc_md, np.asarray(y_pad).reshape(y_shape), y_pad)

        eval_results = {}
        
        sum_inf_time = (end - start) + inf_time1 + inf_time2
        eval_results["disc_md"] = d_disc_md.tolist()
        eval_results["nondisc_md"] = d_nondisc_md.tolist()
        eval_results["ue_time"] = sum_inf_time
        log.info(f"UE time: {sum_inf_time}")
    
        log.info("**************Done.**********************")
        return eval_results