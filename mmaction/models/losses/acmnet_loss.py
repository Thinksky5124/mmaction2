'''
Author: Thyssen Wen
Date: 2021-09-15 15:32:43
LastEditors: Thyssen Wen
LastEditTime: 2021-09-23 10:42:39
Description: file content
FilePath: /mmaction2/mmaction/models/losses/acmnet_loss.py
'''
import torch
import torch.nn as nn

from ..builder import LOSSES
from ...utils import get_root_logger

@LOSSES.register_module()
class ACMNetLoss(nn.Module):
    def __init__(self, lamb1=2e-3, lamb2=5e-5, lamb3=2e-4, feat_margin = 50, dataset="Thumos14Dataset"):
        super().__init__()
        self.dataset = dataset
        self.lamb1 = lamb1 # att_norm_loss param
        self.lamb2 = lamb2
        self.lamb3 = lamb3
        self.feat_margin = feat_margin  #50
        # debug
        self.logger = get_root_logger()

    def cls_criterion(self, inputs, label):
        return - torch.mean(torch.sum(torch.log(inputs) * label, dim=1))
    
    def forward(self , act_inst_cls, act_cont_cls, act_back_cls, label_matix, temp_att=None,\
                act_inst_feat=None, act_cont_feat=None, act_back_feat=None, temp_cas=None, ):
        vid_label = label_matix

        device = act_inst_cls.device
        batch_size = act_inst_cls.shape[0]
        
        act_inst_label = torch.hstack((vid_label, torch.zeros((batch_size, 1), device=device)))
        act_cont_label = torch.hstack((vid_label, torch.ones((batch_size, 1), device=device)))
        act_back_label = torch.hstack((torch.zeros_like(vid_label), torch.ones((batch_size, 1), device=device)))
        
        act_inst_label = act_inst_label / torch.sum(act_inst_label, dim=1, keepdim=True)
        act_cont_label = act_cont_label / torch.sum(act_cont_label, dim=1, keepdim=True)
        act_back_label = act_back_label / torch.sum(act_back_label, dim=1, keepdim=True)
        
        # add a very small number to avoid loss function nan
        epsilon = torch.full(act_back_label.shape, 1.0e-44, device=act_back_label.device, dtype=act_back_cls.dtype)
        
        act_inst_loss = self.cls_criterion(act_inst_cls + epsilon, act_inst_label)
        act_cont_loss = self.cls_criterion(act_cont_cls + epsilon, act_cont_label)
        act_back_loss = self.cls_criterion(act_back_cls + epsilon, act_back_label)
        
        # Guide Loss
        guide_loss = torch.sum(torch.abs(1 - temp_cas[:, :, -1] - temp_att[:, :, 0].detach()), dim=1).mean()

        # Feat Loss
        act_inst_feat_norm = torch.norm(act_inst_feat, p=2, dim=1)
        act_cont_feat_norm = torch.norm(act_cont_feat, p=2, dim=1)
        act_back_feat_norm = torch.norm(act_back_feat, p=2, dim=1)
        
        feat_loss_1 = self.feat_margin - act_inst_feat_norm + act_cont_feat_norm
        feat_loss_1[feat_loss_1 < 0] = 0
        feat_loss_2 = self.feat_margin - act_cont_feat_norm + act_back_feat_norm
        feat_loss_2[feat_loss_2 < 0] = 0
        feat_loss_3 = act_back_feat_norm
        feat_loss = torch.mean((feat_loss_1 + feat_loss_2 + feat_loss_3)**2)

        # Sparse Att Loss
        # att_loss = torch.sum(temp_att[:, :, 0], dim=1).mean() + torch.sum(temp_att[:, :, 1], dim=1).mean() 
        sparse_loss = torch.sum(temp_att[:, :, :2], dim=1).mean()
        
        if self.dataset == "Thumos14Dataset":
            cls_loss = 1.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            cls_loss = 5.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
            
        add_loss = self.lamb1 * guide_loss + self.lamb2 * feat_loss + self.lamb3 * sparse_loss
        
        loss = (cls_loss + add_loss)

        loss_dict = {}
        loss_dict["act_inst_loss"] = act_inst_loss.cpu().item()
        loss_dict["act_cont_loss"] = act_cont_loss.cpu().item()
        loss_dict["act_back_loss"] = act_back_loss.cpu().item()
        loss_dict["guide_loss"] = guide_loss.cpu().item()
        loss_dict["feat_loss"] = feat_loss.cpu().item()
        loss_dict["sparse_loss"] = sparse_loss.cpu().item()
        
        return loss, loss_dict