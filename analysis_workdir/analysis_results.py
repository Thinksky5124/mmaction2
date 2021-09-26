'''
Author: Thyssen Wen
Date: 2021-09-23 19:44:01
LastEditors: Thyssen Wen
LastEditTime: 2021-09-25 18:41:02
Description: file content
FilePath: /mmaction2/analysis_workdir/analysis_results.py
'''
import numpy as np
import seaborn as sns
import mmcv
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from mmaction.core import average_recall_at_avg_proposals

pred_result_path = 'analysis_workdir/LACP/temp_result.json'
groundtrue_path = 'data/THUMOS-14/test.json'
fig_path = 'analysis_workdir/img'
analysis_num = 10

def label_visualize(gt_boxes, pred_boxes, video_name, video_duration):
    # data prepare
    bar_length = 100
    gt_hit_vector = np.zeros((1,bar_length))
    gt_labeling = {}
    pred_hit_vector = np.zeros((1,bar_length))
    pred_labeling = {}
    for i in range(len(gt_boxes['annotations'])):
        info = {}
        t_gt_hit_vector = np.zeros((1,bar_length))
        start_time = float(gt_boxes['annotations'][i]["segment"][0])
        end_time = float(gt_boxes['annotations'][i]["segment"][1])
        label = gt_boxes['annotations'][i]["label"]
        start_idx = int((bar_length / video_duration) * start_time)
        end_idx = min(int((bar_length / video_duration) * end_time), bar_length - 1)
        t_gt_hit_vector[0, start_idx:end_idx] = 1
        gt_hit_vector = gt_hit_vector + t_gt_hit_vector
        info['position'] = (start_idx + end_idx) // 2
        info['label'] = label
        gt_labeling[i] = info

    for i in range(len(pred_boxes)):
        info = {}
        t_pred_hit_vector = np.zeros((1,bar_length))
        start_time = float(pred_boxes[i]["segment"][0])
        end_time = float(pred_boxes[i]["segment"][1])
        label = pred_boxes[i]["label"]
        score = pred_boxes[i]["score"]
        start_idx = int((bar_length / video_duration) * start_time)
        end_idx = min(int((bar_length / video_duration) * end_time), bar_length - 1)
        t_pred_hit_vector[0, start_idx:end_idx] = 1
        pred_hit_vector = pred_hit_vector + t_pred_hit_vector
        info['position'] = (start_idx + end_idx) // 2
        info['label'] = label
        info['score'] = score
        pred_labeling[i] = info
    
    # draw
    fig, axes = plt.subplots(2, 1)
    sns.heatmap(data = gt_hit_vector, ax = axes[0])
    sns.heatmap(data = pred_hit_vector, ax = axes[1])
    axes[0].set_title("GroundTrue")
    axes[1].set_title("Pred_result")
    plt.suptitle(video_name)
    for k, info in gt_labeling.items():
        axes[0].text(info['position'], 1, info['label'])

    for k, info in pred_labeling.items():
        axes[1].text(info['position'], 1, info['label'])
        axes[1].text(info['position'], -5, info['score'])
    figure = fig.get_figure()
    figure.savefig(fig_path + '/' + video_name + '.png', dpi =400)

def import_ground_truth(gtboxes):
    """Read ground truth data from video_infos."""
    ground_truth = {}
    for video_name, video_info in gtboxes.items():
        video_id = video_name
        this_video_ground_truths = []
        for ann in video_info['annotations']:
            t_start, t_end = ann['segment']
            label = ann['label']
            this_video_ground_truths.append([t_start, t_end, label])
        ground_truth[video_id] = np.array(this_video_ground_truths)
    return ground_truth

def import_proposals(results):
    """Read predictions from results."""
    proposals = {}
    num_proposals = 0
    for video_id, result in results.items():
        this_video_proposals = []
        for proposal in result:
            t_start, t_end = proposal['segment']
            score = proposal['score']
            this_video_proposals.append([t_start, t_end, score])
            num_proposals += 1
        proposals[video_id] = np.array(this_video_proposals)
    return proposals, num_proposals

def eval_proposal(results, gtboxes):
    eval_results = OrderedDict()
    ground_truth = import_ground_truth(gtboxes)
    proposal, num_proposals = import_proposals(results)
    temporal_iou_thresholds = np.linspace(0.5, 0.95, 10)
    max_avg_proposals = 2000
    if isinstance(temporal_iou_thresholds, list):
        temporal_iou_thresholds = np.array(temporal_iou_thresholds)

    recall, _, _, auc = (
        average_recall_at_avg_proposals(
            ground_truth,
            proposal,
            num_proposals,
            max_avg_proposals=max_avg_proposals,
            temporal_iou_thresholds=temporal_iou_thresholds))
    eval_results['auc'] = auc
    eval_results['AR@1'] = np.mean(recall[:, 0])
    eval_results['AR@5'] = np.mean(recall[:, 4])
    eval_results['AR@10'] = np.mean(recall[:, 9])
    eval_results['AR@100'] = np.mean(recall[:, 99])
    for metric_name, val in eval_results.items():
        print(f'{metric_name}: {val:.04f}')
    
def main():
    gt_json = mmcv.load(groundtrue_path)
    pred_result_json = mmcv.load(pred_result_path)['results']
    for i in range(analysis_num):
        video_name = random.choice(list(gt_json))
        video_duration = gt_json[video_name]['duration_second']
        label_visualize(gt_json[video_name], pred_result_json[video_name], video_name, video_duration)
    eval_proposal(pred_result_json, gt_json)


if __name__ == '__main__':
    main()