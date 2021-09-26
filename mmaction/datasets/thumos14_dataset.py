'''
Author: Thyssen Wen
Date: 2021-09-17 19:05:29
LastEditors: Thyssen Wen
LastEditTime: 2021-09-24 14:50:05
Description: file content
FilePath: /mmaction2/mmaction/datasets/thumos14_dataset.py
'''
import copy
import os
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np

from ..core import average_recall_at_avg_proposals
from ..localization import wrapper_compute_average_precision
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Thumos14Dataset(BaseDataset):
    """ActivityNet dataset for temporal action localization.

    The dataset loads raw features and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a json file with multiple objects, and each object has a
    key of the name of a video, and value of total frames of the video, total
    seconds of the video, annotations of a video, feature frames (frames
    covered by features) of the video, fps and rfps. Example of a
    annotation file:

    .. code-block:: JSON

        {
            "v_--1DO2V4K74":  {
                "duration_second": 211.53,
                "duration_frame": 6337,
                "annotations": [
                    {
                        "segment": [
                            30.025882995319815,
                            205.2318595943838
                        ],
                        "label": "Rock climbing"
                    }
                ],
                "feature_frame": 6336,
                "fps": 30.0,
                "rfps": 29.9579255898
            },
            "v_--6bJUbfpnQ": {
                "duration_second": 26.75,
                "duration_frame": 647,
                "annotations": [
                    {
                        "segment": [
                            2.578755070202808,
                            24.914101404056165
                        ],
                        "label": "Drinking beer"
                    }
                ],
                "feature_frame": 624,
                "fps": 24.0,
                "rfps": 24.1869158879
            },
            ...
        }


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False, test_cfg=None):
        super().__init__(ann_file, pipeline, data_prefix, test_mode)
        action_classes_path = test_cfg.action_classes_path
        self.evaluater = test_cfg.evaluater
        if action_classes_path is not None:
            _action_classes_path = action_classes_path
            anno_database = mmcv.load(_action_classes_path)
            self.action_classes_list = anno_database['action_classes']

    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        video_infos = []
        anno_database = mmcv.load(self.ann_file)
        for video_name in anno_database:
            video_info = anno_database[video_name]
            video_info['video_name'] = video_name
            video_infos.append(video_info)
        return video_infos

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['data_prefix'] = self.data_prefix
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def _import_ground_truth(self):
        """Read ground truth data from video_infos."""
        ground_truth = {}
        for video_info in self.video_infos:
            video_id = video_info['video_name']
            this_video_ground_truths = []
            for ann in video_info['annotations']:
                t_start, t_end = ann['segment']
                label = ann['label']
                this_video_ground_truths.append([t_start, t_end, label])
            ground_truth[video_id] = np.array(this_video_ground_truths)
        return ground_truth

    @staticmethod
    def proposals2json(results, show_progress=False):
        """Convert all proposals to a final dict(json) format.

        Args:
            results (list[dict]): All proposals.
            show_progress (bool): Whether to show the progress bar.
                Defaults: False.

        Returns:
            dict: The final result dict. E.g.

            .. code-block:: Python

                dict(video-1=[dict(segment=[1.1,2.0]. score=0.9),
                              dict(segment=[50.1, 129.3], score=0.6)])
        """
        result_dict = {}
        print('Convert proposals to json format')
        if show_progress:
            prog_bar = mmcv.ProgressBar(len(results))
        for result in results:
            video_name =  result['video_name']
            result_dict[video_name] = result['proposal_list']
            if show_progress:
                prog_bar.update()
        return result_dict

    @staticmethod
    def _import_proposals(results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name']
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals

    def dump_results(self, results, out, output_format, version='VERSION 1.3'):
        """Dump data to json/csv files."""
        if output_format == 'json':
            result_dict = self.proposals2json(results)
            output_dict = {
                'version': version,
                'results': result_dict,
                'external_data': {}
            }
            mmcv.dump(output_dict, out)
        elif output_format == 'csv':
            # TODO: add csv handler to mmcv and use mmcv.dump
            os.makedirs(out, exist_ok=True)
            header = 'action,start,end,tmin,tmax'
            for result in results:
                video_name, outputs = result
                output_path = osp.join(out, video_name + '.csv')
                np.savetxt(
                    output_path,
                    outputs,
                    header=header,
                    delimiter=',',
                    comments='')
        else:
            raise ValueError(
                f'The output format {output_format} is not supported.')

    def results_to_detections(self, results, version='VERSION 1.3', top_k=2000, **kwargs):
        """Fetch results instances of the entire dataset."""
        result_dict = self.proposals2json(results)
        output_dict = {
            'version': version,
            'results': result_dict,
            'external_data': {}
        }
        return output_dict

    def get_all_gts(self):
        """Fetch groundtruth instances of the entire dataset."""
        anno_dict = mmcv.load(self.ann_file)
        return anno_dict

    def evaluate(
            self,
            results,
            metrics='AR@AN',
            metric_options={
                'AR@AN':
                dict(
                    max_avg_proposals=100,
                    temporal_iou_thresholds=np.linspace(0.5, 0.95, 10)),
                'mAP':
                dict(
                    mAP=dict(eval_dataset='anet'))
            },
            logger=None,
            **deprecated_kwargs):
        """Evaluation in feature dataset.

        Args:
            results (list[dict]): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'AR@AN'.
            metric_options (dict): Dict for metric options. Options are
                ``max_avg_proposals``, ``temporal_iou_thresholds`` for
                ``AR@AN``.
                default: ``{'AR@AN': dict(max_avg_proposals=100,
                temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))}``.
            logger (logging.Logger | None): Training logger. Defaults: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results for evaluation metrics.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['AR@AN'] = dict(metric_options['AR@AN'],
                                           **deprecated_kwargs)
            metric_options['mAP'] = dict(metric_options['mAP'],
                                         **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['AR@AN','mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        

        for metric in metrics:
            if metric == 'AR@AN':
                ground_truth = self._import_ground_truth()
                proposal, num_proposals = self._import_proposals(results)
                temporal_iou_thresholds = metric_options.setdefault(
                    'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                            np.linspace(0.5, 0.95, 10))
                max_avg_proposals = metric_options.setdefault(
                    'AR@AN', {}).setdefault('max_avg_proposals', 100)
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
            elif metric == 'mAP':
                detections = self.results_to_detections(results, **self.evaluater)
                # get gts
                all_gts = self.get_all_gts()
                
                eval_dataset = metric_options[metric][metric]['eval_dataset']
                if eval_dataset == 'anet':
                    iou_range = np.arange(0.5, 1.0, .05)
                    ap_values = wrapper_compute_average_precision(detections, all_gts, iou_range, self.action_classes_list)
                    map_ious = ap_values.mean(axis=1)

                    for iou, map_iou in zip(iou_range, map_ious):
                        eval_results[f'mAP@{iou:.02f}'] = map_iou

        return eval_results