_base_ = ['../_base_/default_runtime.py']

# wandb logger
visualizer = dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend',
                                                        init_kwargs=dict(project='imuposer_mask'))])

# runtime
train_cfg = dict(max_epochs=30, val_interval=1)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-4),
)

# learning policy
param_scheduler = [
    dict(type='StepLR', step_size=100000, gamma=0.96, end=80, by_epoch=False)
]

# auto_scale_lr = dict(base_batch_size=512)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='P-MPJPE', rule='less'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5)
)

# hooks
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='SyncBuffersHook')
]

# model settings
pretrained_transformer_path = '/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/save_weights/update_design/withEmaReset_stage3.pth'
pretrained_vqvae_path = '/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/save_weights_vq/best_model_epoch_hml_emaReset.pth'
model = dict(
    type='IMUPoserEncoder',
    input_dim=3 + 6,
    sensor_num=6,
    seq_len=196,
    drop_out=0.1,
    num_emb=128,
    text_mask_rate=1,
    recon_loss_weight=1,
    acc_weight=1,
    pretrained_transformer=pretrained_transformer_path,
    pretrained_vqvae=pretrained_vqvae_path,
    with_post_optimization=True,
    init_cfg=None
)

# pipelines
train_pipeline = [
    dict(type='InitAlignIMUMotion',
         imu_acc_name='imu_acc',
         imu_ori_name='imu_ori',
         joint_name='joints',
         ),
    dict(type='RotationMatrixTo6D',
         rotation_name='init_aligned_imu_ori'),
    dict(type='RotationMatrixTo6D',
         rotation_name='imu_ori'),
    dict(type='HMLMotionRepresentation',
         joint_name='init_aligned_global_smpl_joints',
         drop_last_pose_name_list=('init_aligned_imu_acc',
                                   'init_aligned_imu_ori',
                                   'init_aligned_global_smpl_joints',
                                   'smpl_pose')),
    # dict(type='NormalizeIMUMotion', imu_acc_name='init_aligned_imu_acc', imu_ori_name='init_aligned_imu_ori',
    #      imu_acc_mean_std_path='/CT/EgoMocap/work/EgoOmniMocap/data/imu_acc_mean_std.pt',
    #      imu_ori_mean_std_path='/CT/EgoMocap/work/EgoOmniMocap/data/imu_ori_mean_std.pt',),
    dict(type='NormalizeHMLMotion', hml_motion_name='motion_hml',
         hml_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         hml_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt'),
    dict(type='PadMotion', seq_len=196,
         pad_name_list=('init_aligned_imu_acc', 'init_aligned_imu_ori', 'motion_hml',
                        'init_aligned_global_smpl_joints', 'smpl_pose'),
         resize_input_sequence=True),
    dict(type='ChangeHMLShape', hml_motion_name='motion_hml'),
    dict(type='ToTensor',
         keys=['init_aligned_imu_acc', 'init_aligned_imu_ori', 'imu_acc', 'imu_ori',
               'init_aligned_global_smpl_joints', 'motion_hml',
               'smpl_pose', 'joints', 'shape', 'transl']),
    dict(
        type='Collect',
        keys=['motion_hml', 'lengths', 'init_aligned_imu_acc', 'init_aligned_imu_ori'],
        meta_keys=['sent_len', 'init_aligned_global_smpl_joints', 'smpl_pose'],
        meta_name='data_samples'
    )
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='IMUPoserDataset',
        imuposer_root_dir='/CT/EgoMocap/work/IMUPoser',
        pipeline=train_pipeline,
        seq_len=197,
        min_seq_len=197,
        split='train',
        random_mask=True,
        signal_num=6,
        tlcontrol_joint_sequence=True,
    )
)

val_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='IMUPoserDataset',
        imuposer_root_dir='/CT/EgoMocap/work/IMUPoser',
        pipeline=train_pipeline,
        seq_len=197,
        min_seq_len=30,
        split='test',
        random_mask=True,
        combo_name='wo_global',
        signal_num=6,
        tlcontrol_joint_sequence=True,
    )
)
test_dataloader = val_dataloader

# evaluators
# val_evaluator = [
#     dict(type='AgrolMPJPE', mode='c-mpjpe',
#          motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
#          motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt',
#          save_path='/CT/EgoMocap/work/EgoOmniMocap/work_dirs/test_imuposer_dataset_finetune_post_optim_acc/results_save_all_w_optim.pkl'),
#     dict(type='AgrolMPJPE', mode='p-mpjpe',
#          motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
#          motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt'),
#     # dict(type='DumpResults',
#     #      out_file_path='/CT/EgoMocap/work/EgoOmniMocap/work_dirs/test_imuposer_dataset_mask/results_raw.pkl')
# ]

val_evaluator = [
    dict(type='EgoOmniMocapError', mode=('pa-mpjpe', 'c-mpjpe', 'jitter-pred', 'jitter-gt', 'global-angle'),
         motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt',
         save_path='/CT/EgoMocap/work/EgoOmniMocap/work_dirs/test_imuposer_dataset_finetune_post_optim_acc_non_global/results_save_all.pkl',
         max_save_length=10),
]

test_evaluator = val_evaluator
