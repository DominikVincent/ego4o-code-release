_base_ = ['../_base_/default_runtime.py']

# wandb logger
visualizer = dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend')])

# runtime
train_cfg = dict(max_epochs=300, val_interval=2)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-5),
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
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='MPJPE', rule='less'),
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
    type='EgoMotionMaskTransformer',
    input_dim=3 + 6,
    drop_out=0.1,
    num_emb=128,
    text_mask_rate=0.9,
    motion_mask_rate=(0, 0.2, 0.4, 0.6),
    epoch_gap=(0, 100, 150, 200),
    pretrained_transformer=pretrained_transformer_path,
    pretrained_vqvae=pretrained_vqvae_path,
    online_prediction=True,
    init_cfg=None
)

# pipelines
train_pipeline = [
    dict(
        type='ToTensor',
        keys=['traj_data', 'motion', 'lengths']
    ),
    dict(
        type='Collect',
        keys=['traj_data', 'motion', 'lengths', 'text'],
        meta_keys=['sent_len'],
        meta_name='data_samples'
    )
]

traj_mean_path = '/CT/EgoMocap/work/EgoOmniMocap/work_dirs/save_tmp/traj_mean.pt'
traj_std_path = '/CT/EgoMocap/work/EgoOmniMocap/work_dirs/save_tmp/traj_std.pt'

normalize_trajectory = dict(type='NormalizeTrajectory', key='traj_data',
                            traj_mean_path=traj_mean_path,
                            traj_std_path=traj_std_path)

val_pipeline = [
    dict(type='ZUp2YUp', joint_name='global_smpl_motion'),
    dict(type='InitAlignGlobalSMPLJoints',
         use_default_floor_height=False),
    dict(type='SMPLJoint2Trajectory',
         joint_ids=(15, 20, 21),
         joint_name='init_aligned_global_smpl_joints',
         with_joint_orientation=True,
         ),
    # dict(type='PadMotionSequence', motion_name='joint_trajectory', pad_length=196),
    dict(type='Rename', source_name='joint_trajectory', target_name='traj_data', copy=False),
    # dict(type='PadMotionSequence', motion_name='traj_data', pad_length=196),
    dict(type='ToFloatTensor',
         keys=['traj_data', 'init_aligned_global_smpl_joints']
         ),
    normalize_trajectory,
    dict(
        type='Collect',
        keys=['traj_data', 'lengths'],  # todo: pad the data
        meta_keys=['init_aligned_global_smpl_joints', 'lengths'],
        meta_name='data_samples'
    )
]
test_pipeline = val_pipeline

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='HMLTrainDataset',
        mode='train',
        data_root='/CT/EgoMocap/work/OmniControl',
        data_opt_path='./dataset/humanml_opt.txt',
        split="train",
        traj_with_rot=True,
        traj_mean_path=traj_mean_path,
        traj_std_path=traj_std_path,
        dim=3 + 6,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='AgorlTestDataset',
        dataset_root='/CT/EgoMocap/work/AGRoL/dataset/AMASS',
        pipeline=val_pipeline,
        add_last=False,
    )
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='AgrolMPJPE', mode='mpjpe',
         motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt',
         save_path='/CT/EgoMocap/work/EgoOmniMocap/work_dirs/ego_omni_mocap_6loc_rot_recurrent_test/results.pkl'),
    dict(type='AgrolMPJPE', mode='p-mpjpe',
         motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt')
]
test_evaluator = val_evaluator
