import torch

def accuracy_score(y_pred, y_true, smooth=1e-6):
    """Accuracy calculation"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    tp = (y_pred * y_true).sum()
    tn = ((1 - y_pred) * (1 - y_true)).sum()
    return (tp + tn + smooth) / (len(y_pred) + smooth)

def dice_score(y_pred, y_true, smooth=1e-6):
    """Dice Similarity Coefficient (DSC)"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def iou_score(y_pred, y_true, smooth=1e-6):
    """Intersection over Union (IoU) / Jaccard Index"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(y_pred, y_true, smooth=1e-6):
    """Precision calculation"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(y_pred, y_true, smooth=1e-6):
    """Recall (Sensitivity) calculation"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    tp = (y_pred * y_true).sum()
    fn = ((1 - y_pred) * y_true).sum()
    return (tp + smooth) / (tp + fn + smooth)

def specificity_score(y_pred, y_true, smooth=1e-6):
    """Specificity calculation"""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    tn = ((1 - y_pred) * (1 - y_true)).sum()
    fp = (y_pred * (1 - y_true)).sum()
    return (tn + smooth) / (tn + fp + smooth)

def f_measure_score(y_pred, y_true, smooth=1e-6):
    """F-Measure calculation"""
    precision = precision_score(y_pred, y_true, smooth)
    recall = recall_score(y_pred, y_true, smooth)
    return (2 * precision * recall + smooth) / (precision + recall + smooth)
