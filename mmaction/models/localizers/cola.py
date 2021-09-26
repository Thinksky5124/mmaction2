'''
Author: Thyssen Wen
Date: 2021-09-22 16:43:01
LastEditors: Thyssen Wen
LastEditTime: 2021-09-23 16:03:32
Description: file content
FilePath: /mmaction2/mmaction/models/localizers/cola.py
'''
# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import torch
import torch.nn as nn
import mmcv
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from mmaction.localization import nms

from .. import builder
from ..builder import LOCALIZERS
from .base import BaseLocalizer

# (a) Feature Embedding and (b) Actionness Modeling
class Actionness_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Actionness_Module, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        out = self.f_embed(x)
        embeddings = out.permute(0, 2, 1)
        out = self.dropout(out)
        out = self.f_cls(out)
        cas = out.permute(0, 2, 1)
        actionness = cas.sum(dim=2)
        return embeddings, cas, actionness

# CoLA Pipeline
@LOCALIZERS.register_module()
class CoLA(BaseLocalizer):
    def __init__(self,
                 len_feature = 2048,
                 num_classes = 20,
                 r_easy = 5,
                 r_hard = 20,
                 m = 3,
                 M = 6,
                 dropout = 0.6,
                 class_thres = 0.2,
                 nms_thres = 0.6,
                 up_scale = 24,
                 feats_fps = 25,
                 cas_thres = np.arange(0.0, 0.25, 0.025),
                 aness_thres = np.arange(0.1, 0.925, 0.025),
                 num_segements = 750,
                 loss_cls=dict(type='CoLALoss'),
                 train_cfg=None,
                 test_cfg=None):
        super(CoLA, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.class_thres = class_thres
        self.nms_thres = nms_thres
        self.up_scale = up_scale
        self.feats_fps = feats_fps
        self.cas_thres = cas_thres
        self.aness_thres = aness_thres
        self.num_segements = num_segements

        action_classes_path = train_cfg.action_classes_path
        anno_database = mmcv.load(action_classes_path)
        self.action_classes_list = anno_database['action_classes']
        self.action_class_num = len(self.action_classes_list)

        self.loss_cls = builder.build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.actionness_module = Actionness_Module(self.len_feature, self.num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_easy = r_easy
        self.r_hard = r_hard
        self.m = m
        self.M = M

        self.dropout = nn.Dropout(p=dropout)
    
    def generate_label_matrix(self, gt_bbox, device):
        vid_label = []
        for item_name in gt_bbox:
            item_anns_list = item_name["annotations"]
            item_label = np.zeros(self.action_class_num)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.action_classes_list[ann_label]] = 1.0
            vid_label.append(np.expand_dims(item_label,0))
        
        vid_label = np.concatenate(vid_label,axis=0)
        vid_label_t = torch.as_tensor(vid_label.astype(np.float32),device=device)
        
        return vid_label_t
    def minmax_norm(self, act_map, min_val=None, max_val=None):
        if min_val is None or max_val is None:
            relu = torch.nn.ReLU()
            max_val = relu(torch.max(act_map, dim=1)[0])
            min_val = relu(torch.min(act_map, dim=1)[0])
        delta = max_val - min_val
        delta[delta <= 0] = 1
        ret = (act_map - min_val) / delta
        ret[ret > 1] = 1
        ret[ret < 0] = 0
        return ret
    
    def upgrade_resolution(self, arr, scale):
        x = np.arange(0, arr.shape[0])
        f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
        scale_x = np.arange(0, arr.shape[0], 1 / scale)
        up_scale = f(scale_x)
        return up_scale
        
    def get_pred_activations(self, src, pred):
        src = self.minmax_norm(src)
        if len(src.size()) == 2:
            src = src.repeat((self.num_classes, 1, 1)).permute(1, 2, 0)
        src_pred = src[0].cpu().numpy()[:, pred]
        src_pred = np.reshape(src_pred, (src.size(1), -1, 1))
        src_pred = self.upgrade_resolution(src_pred, self.up_scale)
        return src_pred

    def grouping(self, arr):
        return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

    def get_proposal_oic(self, tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, _lambda=0.25, gamma=0.2):
        t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
        temp = []
        for i in range(len(tList)):
            c_temp = []
            temp_list = np.array(tList[i])[0]
            if temp_list.any():
                grouped_temp_list = self.grouping(temp_list)
                for j in range(len(grouped_temp_list)):
                    if len(grouped_temp_list[j]) < 2:
                        continue           
                    inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])
                    len_proposal = len(grouped_temp_list[j])
                    outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                    outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))
                    outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))               
                    if len(outer_temp_list) == 0:
                        outer_score = 0
                    else:
                        outer_score = np.mean(wtcam[outer_temp_list, i, 0])
                    c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                    t_start = grouped_temp_list[j][0] * t_factor
                    t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                    c_temp.append([c_pred[i], c_score, t_start, t_end])
                temp.append(c_temp)
        return temp
    
    def get_proposal_dict(self, cas_pred, aness_pred, pred, score_np, vid_num_seg):
        prop_dict = {}
        for th in self.cas_thres:
            cas_tmp = cas_pred.copy()
            num_segments = cas_pred.shape[0]//self.up_scale
            cas_tmp[cas_tmp[:, :, 0] < th] = 0
            seg_list = [np.where(cas_tmp[:, c, 0] > 0) for c in range(len(pred))]
            proposals = self.get_proposal_oic(seg_list, cas_tmp, score_np, pred, self.up_scale, \
                            vid_num_seg, self.feats_fps, num_segments)
            for i in range(len(proposals)):
                class_id = proposals[i][0][0]
                prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]

        for th in self.aness_thres:
            aness_tmp = aness_pred.copy()
            num_segments = aness_pred.shape[0]//self.up_scale
            aness_tmp[aness_tmp[:, :, 0] < th] = 0
            seg_list = [np.where(aness_tmp[:, c, 0] > 0) for c in range(len(pred))]
            proposals = self.get_proposal_oic(seg_list, cas_pred, score_np, pred, self.up_scale, \
                            vid_num_seg, self.feats_fps, num_segments)
            for i in range(len(proposals)):
                class_id = proposals[i][0][0]
                prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]
        return prop_dict

    def post_precessing(self, pred_matrix):
        video_scores, _, actionness, cas = pred_matrix
        score_np = video_scores[0].cpu().data.numpy()

        pred = np.where(score_np >= self.class_thres)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])

        return actionness, cas, pred, score_np

    def value2key(self, dicts, value):
        if value in dicts.values():
            for a in range(0,len(dicts)):
                if list(dicts.values())[a]==value:
                    return list(dicts.keys())[a]
        else:
            raise NotImplementedError

    def result2json(self, result):
        result_file = []
        for i in range(len(result)):
            for j in range(len(result[i])):
                line = {'label': self.value2key(self.action_classes_list, int(result[i][j][0])),
                        'score': result[i][j][1],
                        'segment': [result[i][j][2], result[i][j][3]]}
                result_file.append(line)
        return result_file
    
    def _pred_matrix2result(self, pred_label_matrix, video_meta):
        actionness, cas, pred, score_np = pred_label_matrix
        pred_list={}
        for video_info in video_meta:
            video_name = video_info['video_name']
            vid_len = video_info['feature_frame']
            cas_pred = self.get_pred_activations(cas, pred)
            aness_pred = self.get_pred_activations(actionness, pred)
            proposal_dict = self.get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_len)

            final_proposals = [nms(v, self.nms_thres) for _,v in proposal_dict.items()]
            pred_list[video_name] = self.result2json(final_proposals)
        return pred_list

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act = self.select_topk_embeddings(actionness_drop, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)

        return easy_act, easy_bkg

    def hard_snippets_mining(self, actionness, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores

    def _forward(self, x):
        num_segments = x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard

        embeddings, cas, actionness = self.actionness_module(x)

        easy_act, easy_bkg = self.easy_snippets_mining(actionness, embeddings, k_easy)
        hard_act, hard_bkg = self.hard_snippets_mining(actionness, embeddings, k_hard)
        
        video_scores = self.get_video_cls_scores(cas, k_easy)

        contrast_pairs = {
            'EA': easy_act,
            'EB': easy_bkg,
            'HA': hard_act,
            'HB': hard_bkg
        }

        return video_scores, contrast_pairs, actionness, cas

    def forward_train(self, input_data, label_matix):
        """Defines the computation performed at training."""
        video_scores, contrast_pairs, _, _ = self._forward(input_data)
        losses = self.loss_cls(video_scores, label_matix, contrast_pairs)
        return losses

    def forward_test(self, input_data, video_meta):
        """Defines the computation performed at testing."""
        pred_matrix = self._forward(input_data)
            
        pred_label_matrix = self.post_precessing(pred_matrix)
        pred_list = self._pred_matrix2result(pred_label_matrix, video_meta)
        
        result = []
        output = dict()
        for i in range(len(video_meta)):
            output['video_name'] = video_meta[i]['video_name']
            output['proposal_list'] = pred_list[video_meta[i]['video_name']]
            result.append(output)
        
        return result
        
    def forward(self, 
                raw_feature,
                gt_bbox=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            device = raw_feature.device
            label_matix = self.generate_label_matrix(gt_bbox,device)
            return self.forward_train(raw_feature, label_matix)

        return self.forward_test(raw_feature, video_meta)