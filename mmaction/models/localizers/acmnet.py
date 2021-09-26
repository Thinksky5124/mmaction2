'''
Author: Thyssen Wen
Date: 2021-09-13 11:07:35
LastEditors: Thyssen Wen
LastEditTime: 2021-09-23 15:19:24
Description: file content
FilePath: /mmaction2/mmaction/models/localizers/acmnet.py
'''
import torch
import torch.nn as nn
import mmcv
import numpy as np
from scipy import interpolate
from mmaction.localization import nms

from .. import builder
from ..builder import LOCALIZERS
from .base import BaseLocalizer

@LOCALIZERS.register_module()
class ACMNet(BaseLocalizer):
    def __init__(self,
                 dataset_type,
                 dropout = 0.7,
                 feature_dim = 2048,
                 ins_topk_seg = 2,
                 con_topk_seg = 10,
                 bak_topk_seg = 10,
                 num_classes = 200,
                 frames_per_sec = 25,
                 segment_frames_num = 16,
                 cls_threshold = 0.10,
                 nms_thresh = 0.90,
                 test_upgrade_scale = 20,
                 loss_cls=dict(type='ACMNetLoss'),
                 train_cfg=None,
                 test_cfg=None):
        super(ACMNet, self).__init__()
        self.dataset_type = dataset_type
        self.frames_per_sec = frames_per_sec
        self.segment_frames_num = segment_frames_num
        self.loss_cls = builder.build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cls_threshold = cls_threshold
        self.test_upgrade_scale = test_upgrade_scale
        self.nms_thresh = nms_thresh
        self.dropout_thres = dropout
        self.feature_dim = feature_dim
        self.ins_topk_seg = ins_topk_seg
        self.con_topk_seg = con_topk_seg
        self.bak_topk_seg = bak_topk_seg
        self.num_classes= num_classes

        action_classes_path = train_cfg.action_classes_path
        anno_database = mmcv.load(action_classes_path)
        self.action_classes_list = anno_database['action_classes']
        self.action_class_num = len(self.action_classes_list)

        self.init_weights()
        
    def init_weights(self):
        """Weight initialization for model."""
        self.dropout = nn.Dropout(self.dropout_thres)
        if self.dataset_type == "Thumos14Dataset":
            self.feature_embedding = nn.Sequential(
                # nn.Dropout(self.dropout_thres),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(self.dropout_thres),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        
        # We introduce three-branch attention, action instance, action context and the irrelevant backgrounds.
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=3, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.num_classes + 1))
        
    def value2key(self, dicts, value):
        if value in dicts.values():
            for a in range(0,len(dicts)):
                if list(dicts.values())[a]==value:
                    return list(dicts.keys())[a]
        else:
            raise NotImplementedError

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
    
    def post_precessing(self, pred_matrix):
        act_inst_cls, _, _, _, _, _, temp_att, temp_cas, _, _, _ = pred_matrix
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        fg_score = act_inst_cls[:, :self.action_class_num]
        score_np = fg_score.cpu().numpy()
        pred_np = np.zeros_like(score_np)
        pred_np[score_np >= self.cls_threshold] = 1
        pred_np[score_np < self.cls_threshold] = 0
        #--------------------------------------------------------------------------#
        #--------------------------------------------------------------------------#
        # GENERATE PROPORALS.
        temp_cls_score_np = temp_cas[:, :, :self.action_class_num].cpu().numpy()
        temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], self.action_class_num, 1))
        temp_att_ins_score_np = temp_att[:, :, 0].unsqueeze(2).expand([-1, -1, self.action_class_num]).cpu().numpy()
        temp_att_con_score_np = temp_att[:, :, 1].unsqueeze(2).expand([-1, -1, self.action_class_num]).cpu().numpy()
        temp_att_ins_score_np = np.reshape(temp_att_ins_score_np, (temp_cas.shape[1], self.action_class_num, 1))
        temp_att_con_score_np = np.reshape(temp_att_con_score_np, (temp_cas.shape[1], self.action_class_num, 1))
        
        score_np = np.reshape(score_np, (-1))
        if score_np.max() > self.cls_threshold:
            cls_prediction = np.array(np.where(score_np > self.cls_threshold)[0])
        else:
            cls_prediction = np.array([np.argmax(score_np)], dtype=np.int)
            
        temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
        temp_att_ins_score_np = temp_att_ins_score_np[:, cls_prediction]
        temp_att_con_score_np = temp_att_con_score_np[:, cls_prediction]
        
        return temp_cls_score_np, temp_att_ins_score_np, temp_att_con_score_np, cls_prediction, score_np
    
    def upgrade_resolution(self, arr, scale):
        x = np.arange(0, arr.shape[0])
        f = interpolate.interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
        scale_x = np.arange(0, arr.shape[0], 1 / scale)
        up_scale = f(scale_x)
        return up_scale

    def grouping(self, arr):
        """
        Group the connected results
        """
        return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

    def get_proposal_oic(self, tempseg_list, int_temp_scores, c_pred, c_pred_scores, t_factor, lamb=0.25, gamma=0.20): # [0.25, 0.20]
        temp = []
        for i in range(len(tempseg_list)):
            c_temp = []
            temp_list = np.array(tempseg_list[i])[0]
            if temp_list.any():
                grouped_temp_list = self.grouping(temp_list)
                for j in range(len(grouped_temp_list)):
                    if len(grouped_temp_list[j]) < 2:
                        continue
                    
                    inner_score = np.mean(int_temp_scores[grouped_temp_list[j], i])

                    len_proposal = len(grouped_temp_list[j])
                    outer_s = max(0, int(grouped_temp_list[j][0] - lamb * len_proposal))
                    outer_e = min(int(int_temp_scores.shape[0] - 1), int(grouped_temp_list[j][-1] + lamb * len_proposal))

                    outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                    
                    if len(outer_temp_list) == 0:
                        outer_score = 0
                    else:
                        outer_score = np.mean(int_temp_scores[outer_temp_list, i])

                    c_score = inner_score - outer_score + gamma * c_pred_scores[c_pred[i]]
                    t_start = (grouped_temp_list[j][0]  + 0) * t_factor
                    t_end =   (grouped_temp_list[j][-1] + 1) * t_factor
                    c_temp.append([c_pred[i], c_score, t_start, t_end])
                        
                temp.append(c_temp)
        return temp
    
    def result2json(self, temp_prop_lst):
        result = []
        for i in range(len(temp_prop_lst)):
            for j in range(len(temp_prop_lst[i])):
                line = {'label': self.value2key(self.action_classes_list, int(temp_prop_lst[i][j][0])),
                        'score': temp_prop_lst[i][j][1],
                        'segment': [temp_prop_lst[i][j][2], temp_prop_lst[i][j][3]]}
                result.append(line)

        return result
    
    def _pred_matrix2result(self, pred_label_matrix, video_meta, input_data_shape_2):
        pred_list={}
        for video_info in video_meta:
            video_name = video_info['video_name']
            vid_len = video_info['feature_frame']
            t_factor = (self.segment_frames_num * vid_len) / (self.frames_per_sec * self.test_upgrade_scale  * input_data_shape_2)
            temp_cls_score_np, temp_att_ins_score_np, _, cls_prediction, score_np = pred_label_matrix
            int_temp_cls_scores = self.upgrade_resolution(temp_cls_score_np, self.test_upgrade_scale)
            int_temp_att_ins_score_np = self.upgrade_resolution(temp_att_ins_score_np, self.test_upgrade_scale)
            # int_temp_att_con_score_np = self.upgrade_resolution(temp_att_con_score_np, self.test_upgrade_scale)
            
            if self.dataset_type == "Thumos14Dataset":
                cas_act_thresh = np.arange(0.15, 0.25, 0.05)
                att_act_thresh = np.arange(0.15, 1.00, 0.05)
            else:
                cas_act_thresh = [0.005, 0.01, 0.015, 0.02]
                att_act_thresh = [0.005, 0.01, 0.015, 0.02]
            
            proposal_dict = {}
            # CAS based proposal generation
            # cas_act_thresh = []
            for act_thresh in cas_act_thresh:

                tmp_int_cas = int_temp_cls_scores.copy()
                zero_location = np.where(tmp_int_cas < act_thresh)
                tmp_int_cas[zero_location] = 0
                
                tmp_seg_list = []
                for c_idx in range(len(cls_prediction)):
                    pos = np.where(tmp_int_cas[:, c_idx] >= act_thresh)
                    tmp_seg_list.append(pos)
                
                if self.dataset_type == "Thumos14Dataset":
                    props_list = self.get_proposal_oic(tmp_seg_list, (1.0*tmp_int_cas + 0.0*int_temp_att_ins_score_np), cls_prediction, score_np, t_factor, lamb=0.2, gamma=0.0)
                else:
                    props_list = self.get_proposal_oic(tmp_seg_list, (0.70*tmp_int_cas + 0.30*int_temp_att_ins_score_np), cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.0)
                
                for i in range(len(props_list)):
                    if len(props_list[i]) == 0:
                        continue
                    class_id = props_list[i][0][0]
                    
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    
                    proposal_dict[class_id] += props_list[i]
            
            # att_act_thresh = []
            for att_thresh in att_act_thresh:

                tmp_int_att = int_temp_att_ins_score_np.copy()
                zero_location = np.where(tmp_int_att < att_thresh)
                tmp_int_att[zero_location] = 0
                
                tmp_seg_list = []
                for c_idx in range(len(cls_prediction)):
                    pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
                    tmp_seg_list.append(pos)
                
                if self.dataset_type == "Thumos14Dataset":
                    props_list = self.get_proposal_oic(tmp_seg_list, (1.0*int_temp_cls_scores + 0.0*tmp_int_att), cls_prediction, score_np, t_factor, lamb=0.2, gamma=0.0)
                else:
                    props_list = self.get_proposal_oic(tmp_seg_list, (0.70*int_temp_cls_scores + 0.30*tmp_int_att), cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.250)
                
                for i in range(len(props_list)):
                    if len(props_list[i]) == 0:
                        continue
                    class_id = props_list[i][0][0]
                    
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    
                    proposal_dict[class_id] += props_list[i]
            
            # NMS 
            final_proposals = []
            
            for class_id in proposal_dict.keys():
                final_proposals.append(nms(proposal_dict[class_id], self.nms_thresh))
                    
            pred_result= self.result2json(final_proposals)
            pred_list[video_name] = pred_result
        return pred_list

    def _forward(self,input_features):
        device = input_features.device
        input_features = input_features.permute(0, 2, 1)
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]
        
        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)
        
        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        
        if self.dataset_type == "Thumos14Dataset":
            temp_att = self.att_branch((embeded_feature))
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            temp_att = self.att_branch(self.dropout(embeded_feature))
        
        temp_att = temp_att.permute(0, 2, 1)
        temp_att = torch.softmax(temp_att, dim=2)
        
        act_inst_att = temp_att[:, :, 0].unsqueeze(2)
        act_cont_att = temp_att[:, :, 1].unsqueeze(2)
        act_back_att = temp_att[:, :, 2].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1)
        embeded_feature_rev = embeded_feature
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx

        act_cas = self.snippet_cls(self.dropout(embeded_feature))
        act_inst_cas = act_cas * act_inst_att
        act_cont_cas = act_cas * act_cont_att
        act_back_cas = act_cas * act_back_att
        
        sorted_inst_cas, _ = torch.sort(act_inst_cas, dim=1, descending=True)
        sorted_cont_cas, _ = torch.sort(act_cont_cas, dim=1, descending=True)
        sorted_back_cas, _ = torch.sort(act_back_cas, dim=1, descending=True)
        
        act_inst_cls = torch.mean(sorted_inst_cas[:, :inst_topk_num, :], dim=1)
        act_cont_cls = torch.mean(sorted_cont_cas[:, :cont_topk_num, :], dim=1)
        act_back_cls = torch.mean(sorted_back_cas[:, :back_topk_num, :], dim=1)
        act_inst_cls = torch.softmax(act_inst_cls, dim=1)
        act_cont_cls = torch.softmax(act_cont_cls, dim=1)
        act_back_cls = torch.softmax(act_back_cls, dim=1)
        
        act_inst_cas = torch.softmax(act_inst_cas, dim=2)
        act_cont_cas = torch.softmax(act_cont_cas, dim=2)
        act_back_cas = torch.softmax(act_back_cas, dim=2)
        
        act_cas = torch.softmax(act_cas, dim=2)
        
        _, sorted_act_inst_att_idx = torch.sort(act_inst_att, dim=1, descending=True)
        _, sorted_act_cont_att_idx = torch.sort(act_cont_att, dim=1, descending=True)
        _, sorted_act_back_att_idx = torch.sort(act_back_att, dim=1, descending=True)
        act_inst_feat_idx = sorted_act_inst_att_idx[:, :inst_topk_num, :].expand([-1, -1, self.feature_dim])
        act_cont_feat_idx = sorted_act_cont_att_idx[:, :cont_topk_num, :].expand([-1, -1, self.feature_dim])
        act_back_feat_idx = sorted_act_back_att_idx[:, :back_topk_num, :].expand([-1, -1, self.feature_dim])
        act_inst_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_inst_feat_idx), dim=1)
        act_cont_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_cont_feat_idx), dim=1)
        act_back_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_back_feat_idx), dim=1)
        
        return act_inst_cls, act_cont_cls, act_back_cls,\
               act_inst_feat, act_cont_feat, act_back_feat,\
               temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas

    def forward_train(self, input_data, label_matix):
        """Defines the computation performed at training."""
        act_inst_cls, act_cont_cls, act_back_cls,\
        act_inst_feat, act_cont_feat, act_back_feat,\
        temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas = self._forward(input_data)
        losses = self.loss_cls(act_inst_cls, act_cont_cls, act_back_cls, label_matix, temp_att, \
                             act_inst_feat, act_cont_feat, act_back_feat, act_inst_cas)
        return losses

    def forward_test(self, input_data, video_meta):
        """Defines the computation performed at testing."""
        input_data_shape_2 = input_data.shape[2]
        pred_matrix = self._forward(input_data)
            
        pred_label_matrix = self.post_precessing(pred_matrix)
        pred_list = self._pred_matrix2result(pred_label_matrix, video_meta, input_data_shape_2)
        
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