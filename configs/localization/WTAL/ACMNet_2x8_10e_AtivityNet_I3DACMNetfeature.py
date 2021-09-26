_base_ = [
    '../../_base_/long_save_checkpoint_runtime.py'
]
# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = 'data/ActivityNet-13/train'
data_root_val = 'data/ActivityNet-13/test'
ann_file_train = 'data/ActivityNet-13/train.json'
ann_file_val = 'data/ActivityNet-13/test.json'
ann_file_test = 'data/ActivityNet-13/test.json'
action_classes_path = 'data/ActivityNet/action_classes.json'

# model settings
sample_model_train = 'random' 
sample_model_test = 'uniform'
temporal_dim = 75
train_cfg_ = dict(
    action_classes_path=action_classes_path
)

test_cfg_ = dict(
    action_classes_path=action_classes_path,
    evaluater=dict(
            top_k=2000)
)
model = dict(
    type='ACMNet',
    dropout = 0.7,
    feature_dim = 2048,
    ins_topk_seg = 2,
    con_topk_seg = 10,
    bak_topk_seg = 10,
    num_classes = 200,
    cls_threshold = 0.10,
    dataset_type = dataset_type,
    frames_per_sec = 25,
    segment_frames_num = 16,
    test_upgrade_scale = 20,
    nms_thresh = 0.90,
    loss_cls=dict(type='ACMNetLoss',
                  dataset = dataset_type,
                  lamb1 = 5e-3,
                  lamb2 = 5e-5,
                  lamb3 = 0e-4,
                  feat_margin = 50
                  ),
    train_cfg=train_cfg_,
    test_cfg=test_cfg_)

test_pipeline = [
    dict(type='LoadLocalizationI3DFeature', dataset = 'AtivityNet_ACMNet', temporal_dim = temporal_dim),
    dict(type='FeatureFormatShape', sample_len = temporal_dim, sample_model = sample_model_test),
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
    dict(type='LoadLocalizationI3DFeature', dataset = 'AtivityNet_ACMNet', temporal_dim = temporal_dim),
    dict(type='FeatureFormatShape', sample_len = temporal_dim, sample_model = sample_model_train),
    dict(
        type='GenerateLocalizationLabels', gt_bbox_model='Localization'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
val_pipeline = [
    dict(type='LoadLocalizationI3DFeature', dataset = 'AtivityNet_ACMNet', temporal_dim = temporal_dim),
    dict(type='FeatureFormatShape', sample_len = temporal_dim, sample_model = sample_model_test),
    dict(
        type='GenerateLocalizationLabels', gt_bbox_model='Localization'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
data = dict(
    videos_per_gpu=64,
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
evaluation = dict(interval=5, metrics=['mAP'])

# optimizer
optimizer = dict(
    type='Adam', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=1000)
total_epochs = 1000

# runtime settings
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/ACMNet_2x8_10e_AtivityNet_I3DACMNetfeature/'
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
