from utils.utils_heads import (
    ElectraClassificationHeadCustom,
    ElectraClassificationHeadSN,
    ElectraNERHeadCustom,
    ElectraNERHeadSN,
    ElectraClassificationHS
)
import numpy as np

def is_custom_head(model):
    if not hasattr(model, 'classifier'):
        return False
    if (isinstance(model.classifier, ElectraClassificationHeadCustom) or 
        isinstance(model.classifier, ElectraNERHeadCustom) or 
        isinstance(model.classifier, ElectraClassificationHeadSN) or 
        isinstance(model.classifier, ElectraNERHeadSN) or 
        isinstance(model.classifier, ElectraClassificationHS)):
        return True
    return False

def unpad_features(features, labels):
    true_features = [f for (f, l) in zip(features, labels) if l != -100]
    true_labels = [l for (f, l) in zip(features, labels) if l != -100]

    return np.array(true_features), np.array(true_labels)

def pad_scores(unc, full_labels, flatten_labels):
    flatten_full_ue = np.zeros(shape=flatten_labels.shape)
    flatten_full_ue[flatten_labels!=-100] = unc
    full_ue = flatten_full_ue.reshape(full_labels.shape)
    return full_ue