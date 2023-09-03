import torch
import torch.nn as nn
from pdb import set_trace as pause 
from torch import nn 

def mil_loss(cls_score, labels):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)

    return loss.mean()
'''
def mil_loss(self, mil_score, onehot_target_im):
        import torch.nn.functional as F
        from torch import nn 
        # refer to code.layers.losses.mil_loss; target_classes: labels, src_logits: class scores
        # src_logits.shape = (2, 100, 92) = (batch_size, n_tokens, n_classes)
        # target_classes.shape = (2, 100)
        
        im_cls_score = mil_score.sum(dim=1) # 2, 92 # sum over proposals -> get img labels
        multi_criterion = nn.MultiLabelSoftMarginLoss(weight=None, reduce=False)
        loss = multi_criterion(im_cls_score, onehot_target_im) # 

        return loss.mean()
'''