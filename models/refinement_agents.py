import torch
import torch.nn as nn
import torch.nn.functional as F

from util.config import cfg

class RefinementAgents(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.refine_score = []
        for i_refine in range(cfg.REFINE_TIMES):
            self.refine_score.append(nn.Linear(dim_in, dim_out))
        self.refine_score = nn.ModuleList(self.refine_score)

        self._init_weights()

    def _init_weights(self):
        for i_refine in range(cfg.REFINE_TIMES):
            nn.init.normal_(self.refine_score[i_refine].weight, std=0.01)
            nn.init.constant_(self.refine_score[i_refine].bias, 0)

    def forward(self, x, aux=False): # 6, 2, 100, 256
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        if aux:
            refine_score = [F.softmax(refine(x), dim=2) for refine in self.refine_score] # 6, 100, 92
        else:
            refine_score = [F.softmax(refine(x[-1]), dim=2) for refine in self.refine_score] # 1, 100, 92

        return refine_score
