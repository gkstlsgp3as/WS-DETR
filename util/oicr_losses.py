import torch
import torch.nn as nn
from pdb import set_trace as pause 

class OICRLosses(nn.Module):

    def forward(self, pcl_probs, labels, cls_loss_weights, gt_assignment, im_labels):
        # shapes
        # cls_loss_weights = (2, 100)
        # pcl_probs = (2, 100, 92) / (6, 2, 100, 92)

        eps = 1e-6
        pcl_probs = pcl_probs.clamp(eps, 1-eps).log()
        
        if pcl_probs.dim() == 4:
        # cls_loss_weights = cls_loss_weights.repeat(pcl_probs.shape[1],1).permute(1,0).cuda()
            cls_loss_weights = cls_loss_weights.unsqueeze(0).unsqueeze(-1).repeat(pcl_probs.shape[0], 1, 1, pcl_probs.shape[-1]).cuda() # 6, 2, 100, 92
            labels = labels.unsqueeze(0).unsqueeze(-1).repeat(pcl_probs.shape[0], 1, 1, pcl_probs.shape[-1]).cuda()
            reap   = torch.arange(pcl_probs.shape[-1])[None,:].repeat(pcl_probs.shape[-3], pcl_probs.shape[-2], 1).unsqueeze(0).repeat(pcl_probs.shape[0],1,1,1).long() # 100, 92
             
            labels = (reap.cuda() - labels == 0).float().cuda() # 6, 2, 100, 92; class 일치하는 경우
            loss = labels * cls_loss_weights * pcl_probs
            loss = -loss.sum(dim=-1).mean() # / pcl_probs.size(0) # 2, 100, 92 > 2, 100 (sum over class) 
                                                        # > 2 (mean over proposals) > 1 (mean over images)
        else:
            cls_loss_weights = cls_loss_weights.unsqueeze(-1).repeat(1, 1, pcl_probs.shape[-1]).cuda() # 2, 100, 92
            labels = labels.unsqueeze(-1).repeat(1, 1, pcl_probs.shape[-1]).cuda() # 2, 100, 92
            reap   = torch.arange(pcl_probs.shape[-1])[None,:].repeat(pcl_probs.shape[-2], 1).unsqueeze(0).repeat(pcl_probs.shape[0],1,1).long() # 100, 92
            labels = (reap.cuda() - labels == 0).float().cuda()
            #print("labels:", labels) # [1, 0, 0, ...]]
            loss = labels * cls_loss_weights * pcl_probs
            loss = -loss.sum(dim=2).mean() # / pcl_probs.size(0) # 2, 100, 92 > 2, 100 (sum over class) 
                                                        # > 2 (mean over proposals) > 1 (mean over images)
        return loss

