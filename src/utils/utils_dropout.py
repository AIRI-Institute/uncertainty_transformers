from transformers import (
    ElectraForSequenceClassification,
    RobertaForSequenceClassification,
    ElectraForTokenClassification,
    XLNetForSequenceClassification,
    DistilBertForTokenClassification,
    DebertaForSequenceClassification,
)
from utils.utils_heads import (ElectraClassificationHeadCustom, 
                               ElectraNERHeadCustom,
                               ElectraNERHeadSN,
                               ElectraClassificationHeadSN)


def get_last_dropout(model):
    if (
        isinstance(model, ElectraForSequenceClassification)
        or isinstance(model, ElectraForTokenClassification)
        or isinstance(model, RobertaForSequenceClassification)
        or isinstance(model, DistilBertForTokenClassification)
    ):
        if (isinstance(model.classifier, ElectraClassificationHeadCustom) 
            or isinstance(model.classifier, ElectraNERHeadCustom)
            or isinstance(model.classifier, ElectraClassificationHeadSN)
            or isinstance(model.classifier, ElectraNERHeadSN)):
            return model.classifier.dropout2
        else:
            return model.classifier.dropout
    else:
        return model.dropout


def set_last_dropout(model, dropout):
    if (
        isinstance(model, ElectraForSequenceClassification)
        or isinstance(model, ElectraForTokenClassification)
        or isinstance(model, RobertaForSequenceClassification)
        or isinstance(model, DistilBertForTokenClassification)
    ):
        if (isinstance(model.classifier, ElectraClassificationHeadCustom) 
            or isinstance(model.classifier, ElectraNERHeadCustom)
            or isinstance(model.classifier, ElectraClassificationHeadSN)
            or isinstance(model.classifier, ElectraNERHeadSN)):
            model.classifier.dropout2 = dropout
        else:
            model.classifier.dropout = dropout
    elif (
        isinstance(model, XLNetForSequenceClassification)
    ):
        model.sequence_summary.last_dropout = dropout
    else:
        model.dropout = dropout


def set_last_dropconnect(model, linear_dropconnect):
    if (
        isinstance(model, ElectraForSequenceClassification)
        or isinstance(model, ElectraForTokenClassification)
        or isinstance(model, RobertaForSequenceClassification)
        or isinstance(model, DebertaForSequenceClassification)
    ):
        if (isinstance(model.classifier, ElectraClassificationHeadCustom) 
            or isinstance(model.classifier, ElectraNERHeadCustom)
            or isinstance(model.classifier, ElectraClassificationHeadSN)
            or isinstance(model.classifier, ElectraNERHeadSN)):
            model.classifier.out_proj = linear_dropconnect(
                linear=model.classifier.out_proj, activate=False
            )
        else:
            model.classifier.dense = linear_dropconnect(
                linear=model.classifier.dense, activate=False
            )
    else:
        model.dense = linear_dropconnect(linear=model.dense, activate=False)
