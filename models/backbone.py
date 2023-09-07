# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

# wsod
class DINOBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args: str, return_interm_layers: bool):
        super().__init__()
        import models.vision_transformer as vits
        self.arch = args.arch
        self.patch_size = args.patch_size
        self.conv = nn.Conv2d(384, args.hidden_dim, kernel_size=(1,1))
        
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            self.return_layers = True
            self.strides = [8, 16, 32]
            self.num_channels = [64, 128, 256]
            
            self.conv_mult = nn.Conv2d(384, 384, kernel_size=(3,3), stride=(2,2))
            self.conv2 = nn.Conv2d(384, self.num_channels[0], kernel_size=(1,1))
            self.conv3 = nn.Conv2d(384, self.num_channels[1], kernel_size=(1,1))
            self.conv4 = nn.Conv2d(384, self.num_channels[2], kernel_size=(1,1))
        else:
            self.return_layers = False
            self.strides = [32]
            self.num_channels = [args.hidden_dim] # 256

        self.model = vits.__dict__[self.arch](patch_size=self.patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        
        url = self.return_url()
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.model.load_state_dict(state_dict, strict=True)

    def return_url(self):
        if self.arch == "vit_small" and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif self.arch == "vit_small" and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif self.arch == "vit_base" and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif self.arch == "vit_base" and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        return url

    def forward(self, tensor_list: NestedTensor):
        w_featmap = tensor_list.tensors.shape[-2] // self.patch_size # 50 -> 50*8 = 400 (org_w)
        h_featmap = tensor_list.tensors.shape[-1] // self.patch_size # 60 -> 60*8 = 480 (org_h)
        
        # get_intermediate_layers로 중간 layer들 뽑아서 쓰는것도 가능은 할 것 같은데 
        attentions, _x_ctxed, _x_final, head_x = self.model.get_last_selfattention(tensor_list.tensors)
        #breakpoint()
        # height x width = # of tokens
        # attentions : query @ key, [batch, # of heads,  # of tokens, # of tokens] e.g. (2,6,3001,3001)
        # _x_ctxed : attentions @ value, means contextualized, [batch, # of tokens, dimension] # (2,3001,384)
        # _x_final : mlp(attentions), attention block(attention + mlp)을 완전히 통과한 이후의 tokens  # (2,3001,384)
        # -> 즉 frozen extractor에서 뽑은 feature를 쓰려면, _x_final을 써야함 
        # head_x : (2,3001,6,64) => 6*64 = 384 (384: embedding dimension)
        #( 2, 3001, 128 ) * 3 set => (2, w, h, 128) => (2, w, h, 256)
        # nhead = attentions.shape[1]
        # cls_attn = attentions.mean(1).squeeze()[0,1:].reshape(w_featmap, h_featmap)

        # nhead = attentions.shape[1] 
        ntokens = _x_final.shape[2] # 384
        attn = _x_final.transpose(1,2)[:,:,1:].reshape(-1, ntokens, w_featmap, h_featmap) # 2, 384, 8481 (8480+1) > 2, 384, w, h
        # cls_attn = attentions[:,:,0,1:].reshape(-1, nhead, w_featmap, h_featmap) # ex. 2, 6, 62, 75
        # cls_attn = self.conv(cls_attn) # 2, 256, w, h
        # cls_attn = attentions[:,:,0,1:].reshape(-1, nhead, w_featmap, h_featmap) # ex. 2, 6, 62, 75

        ## multi-scale
        if self.return_layers:
            # attn = (2,384,w,h)
            attn_1st = self.conv_mult(attn) # 2, 384, w/2, h/2
            attn_2nd = self.conv_mult(attn_1st) # 2, 384, w/4, h/4
            #attn_3rd = self.conv_mult(attn_2nd) # 2, 384, w/8, h/8
            
            attn = self.conv2(attn) # 2, 64, w, h
            attn_1st = self.conv3(attn_1st) # 2, 128, w/2, h/2
            attn_2nd = self.conv4(attn_2nd) # 2, 256, w/4, h/4
            #cls_attn = self.conv(cls_attn).flatten(2).permute(1,0,1) # ex. torch.Size([2, 256, 9435]) > 4960, 2, 256
            # RuntimeError: Given groups=1, weight of size [256, 256, 1, 1], expected input[1, 9435, 2, 256] to have 256 channels, but got 9435 channels instead
            
            xs = { 'layer0': attn, 'layer1': attn_1st, 'layer2': attn_2nd } 
        
        else: 
            attn = self.conv(attn) # 2, 256, w, h
            xs = { 'last': attn }
        
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    # wsod
    if not args.backbone.startswith('resnet'): 
        backbone = DINOBackbone(args, return_interm_layers)
    else:
        train_backbone = args.lr_backbone > 0
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
