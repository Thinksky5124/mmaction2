'''
Author: Thyssen Wen
Date: 2021-09-15 16:24:47
LastEditors: Thyssen Wen
LastEditTime: 2021-09-15 16:24:47
Description: file content
FilePath: /mmaction2/mmaction/localization/localization_utils.py
'''
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def import_ground_truth(ground_truth, action_classes_list):
    """Reads ground truth file, checks if it is well formatted, and returns
        the ground truth instances and the activity classes.

    Parameters
    ----------
    ground_truthe : dict
        Full path to the ground truth json file.

    Outputs
    -------
    ground_truth : df
        Data frame containing the ground truth instances.
    activity_index : dict
        Dictionary containing class index.
    """
    blocked_videos = []

    data = ground_truth
    
    # Load Ground Truth data.
    video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
    for video_id, v in data.items():
        if video_id in blocked_videos:
            continue
        for ann in v['annotations']:
            video_lst.append(video_id)
            t_start_lst.append(float(ann['segment'][0]))
            t_end_lst.append(float(ann['segment'][1]))
            label_lst.append(action_classes_list[ann['label']])
    
    ground_truth = pd.DataFrame({'video-id':video_lst,
                                    't-start':t_start_lst,
                                    't-end':t_end_lst,
                                    'label':label_lst})
    
    return ground_truth

    
def import_prediction(prediction, action_classes_list):
    """Reads prediction file, checks if it is well formatted, and returns
        the prediction instances.

    Parameters
    ----------
    prediction_filename : str
        Full path to the prediction json file.

    Outputs
    -------
    prediction : df
        Data frame containing the prediction instances.
    """
    blocked_videos = []
    predict_data = prediction

    video_lst, t_start_lst, t_end_lst, label_lst, score_lst = [], [], [], [], []
    for video_id, v in predict_data['results'].items():
        if video_id in blocked_videos:
            continue
        for pred in v:
            label = action_classes_list[pred['label']]
            video_lst.append(video_id)
            t_start_lst.append(float(pred['segment'][0]))
            t_end_lst.append(float(pred['segment'][1]))
            label_lst.append(label)
            score_lst.append(float(pred['score']))
    
    prediction = pd.DataFrame({'video-id':video_lst,
                                't-start':t_start_lst,
                                't-end':t_end_lst,
                                'label':label_lst,
                                'score':score_lst})

    return prediction

def get_predictions_with_label(prediction_by_label, label_name, cidx):
    """Get all predicitons of the given label. Return empty DataFrame if there
    is no predcitions with the given label.
    """
    try:
        return prediction_by_label.get_group(cidx).reset_index(drop=True)
    except:
        # print("Warning: No predictions of label {} were provided".format(label_name))
        return pd.DataFrame

def wrapper_compute_average_precision(detections, all_gts, iou_range, action_classes_list):
    """Computes average precision for each class in the subset.
    """
    ap = np.zeros((len(iou_range), len(action_classes_list)))

    ground_truth = import_ground_truth(all_gts, action_classes_list)
    prediction = import_prediction(detections, action_classes_list)

    ground_truth_by_label = ground_truth.groupby('label')
    prediction_by_label = prediction.groupby('label')
    
    results = Parallel(n_jobs=10)(delayed(compute_average_precision_detection)(
                                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                                prediction=get_predictions_with_label(prediction_by_label, label_name, cidx),
                                tiou_thresholds=iou_range,
                                ) for label_name, cidx in action_classes_list.items())
    
    for i, cidx in enumerate(action_classes_list.values()):
        ap[:, cidx] = results[i]
    
    return ap

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.1, 0.9, 9)):
    """Compute average precision (detection task) between ground truth and predictions data frames.
    If multiple predictions occurs for the same predicted segment, only the one with highest score is
    mathced as positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (df): 
            Data frame containing the ground truth instances.
            Required fields: ['video_id', 't-start', 't-end']
            
        prediction (df): 
            Data frame containing the prediction instances.
            Required fields: ['video-id', 't-start', 't-end', 'score']
        
        tiou_thresholds (1darray, optional):
            Temporal intersection over union threshold.
            Defaults to np.linspace(0.1, 0.9, 9).
            
    Outpus:
    ap: float
        average precision scores.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap
    
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # sort predictions by decreasing score order
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    
    # Initializa true positive and false positive vectors
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))
    
    ground_truth_gbvn = ground_truth.groupby('video-id')
    
    for idx, this_pred in prediction.iterrows():
        
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            # wrong predicted association cls label.
            fp[:, idx] = 1
            continue
        
        this_gt = ground_truth_videoid.reset_index()
        
        tiou_arr = get_segment_iou(this_pred[['t-start', 't-end']].values,
                                   this_gt[['t-start', 't-end']].values)
        
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after filters above
                tp[tidx, idx] = 1
                # for each gt, we only assign the highest iou detection instance.
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx

                break
                
            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1 
                
    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    
    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = get_AP(precision_cumsum[tidx, :], recall_cumsum[tidx, :])
    
    return ap

def get_segment_iou(target_segment, candidate_segment):
    """
    Calculate the t-IOU between target_segments and the candidate_segments.
    
    Args:
        target_segment (1d array): [t_start, t_end]
        candidate_segment (2d array): N X [t_start, t_end]
    Return:
        tIOU
    """
    tt1 = np.maximum(target_segment[0], candidate_segment[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segment[:, 1])
    segment_intersection = (tt2 - tt1).clip(0)
    segment_union = (candidate_segment[:, 1] - candidate_segment[:, 0]) + \
                    (target_segment[1] - target_segment[0]) - segment_intersection
    tIOU = segment_intersection.astype(np.float) / segment_union
    
    return tIOU

def get_AP(prec, rec):
    """
    Calculate the interpolated AP -- VOCdevkit from VOC 2011
    
    Args:
        prec ([type]): [description]
        rec ([type]): [description]

    Returns:
        AP [float]:
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for idx in range(len(mprec) - 1)[::-1]:
        mprec[idx] = max(mprec[idx], mprec[idx + 1])    
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    
    return ap