_base_ = [
    '../../_base_/models/bmn_400x100.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
data_root_val = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
ann_file_train = 'data/ActivityNet/cuhk_annotataions/anet_anno_train.json'
ann_file_val = 'data/ActivityNet/cuhk_annotataions/anet_anno_val.json'
ann_file_test = 'data/ActivityNet/cuhk_annotataions/anet_anno_val.json'
action_classes_path = 'data/ActivityNet/action_classes.json'

train_cfg_ = dict(
    action_classes_path=action_classes_path
)

test_cfg_ = dict(
    action_classes_path=action_classes_path,
    evaluater=dict(
            top_k=2000)
)

test_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature']),
]
train_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels', gt_bbox_model='Proposal'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels', gt_bbox_model='Proposal'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root_val,
        test_cfg=test_cfg_),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val,
        test_cfg=test_cfg_),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        test_cfg=test_cfg_))
evaluation = dict(interval=1, metrics=['AR@AN'])

# optimizer
optimizer = dict(
    type='Adam', lr=0.001, weight_decay=0.0001)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=7)
total_epochs = 9

# runtime settings
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
