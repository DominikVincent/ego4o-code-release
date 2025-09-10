_base_ = ['../_base_/default_runtime.py']

# wandb logger
visualizer = dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend',
                                                        init_kwargs=dict(project='nymeria_vq_vae'))])

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
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='C-MPJPE', rule='less', max_keep_ckpts=5),
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
    type='HumanVQVAE',
    args=None,
    nb_code=512,
    code_dim=512,
    output_emb_width=512,
    down_t=2,
    stride_t=2,
    width=512,
    depth=3,
    dilation_growth_rate=3,
    activation='relu',
    norm=None,
    init_cfg=dict(type='Pretrained', checkpoint=pretrained_vqvae_path)
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
                                   'init_aligned_global_smpl_joints')),
    # dict(type='NormalizeIMUMotion', imu_acc_name='init_aligned_imu_acc', imu_ori_name='init_aligned_imu_ori',
    #      imu_acc_mean_std_path='/CT/EgoMocap/work/EgoOmniMocap/data/imu_acc_mean_std.pt',
    #      imu_ori_mean_std_path='/CT/EgoMocap/work/EgoOmniMocap/data/imu_ori_mean_std.pt',),
    dict(type='NormalizeHMLMotion', hml_motion_name='motion_hml',
         hml_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         hml_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt'),
    dict(type='PadMotion', seq_len=148,
         pad_name_list=(
             'init_aligned_imu_acc', 'init_aligned_imu_ori', 'motion_hml', 'init_aligned_global_smpl_joints'),
         resize_input_sequence=True),
    dict(type='ChangeHMLShape', hml_motion_name='motion_hml'),

    # dict(type='AddDummyText', text_name='text', dummy_text=''),
    dict(type='ToTensor',
         keys=['init_aligned_imu_acc', 'init_aligned_imu_ori', 'imu_acc', 'imu_ori',
               'init_aligned_global_smpl_joints', 'motion_hml',
               'joints']),
    dict(
        type='Collect',
        keys=['motion_hml'],
        meta_keys=['init_aligned_global_smpl_joints', 'motion_file', 'motion_id', ],
        meta_name='data_samples'
    )
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

imuposer_dataset_w_text = dict(
    type='NymeriaDataset',
    dataset_dir='/scratch/inf0/user/jianwang/nymeria',
    pipeline=train_pipeline,
    seq_len=150,
    min_seq_len=150,
    split='train',
    random_mask=True,
    signal_num=6,
    tlcontrol_joint_sequence=True,
    with_text=True,
)

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            imuposer_dataset_w_text,
        ]
    ),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='NymeriaDataset',
        dataset_dir='/scratch/inf0/user/jianwang/nymeria',
        pipeline=train_pipeline,
        seq_len=150,
        min_seq_len=150,
        split='test',
        random_mask=False,
        combo_name='global',
        signal_num=6,
        tlcontrol_joint_sequence=True,
        with_text=True,
    )
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='AgrolMPJPE', mode='c-mpjpe',
         motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt',
         # save_path='/CT/EgoMocap/work/EgoOmniMocap/work_dirs/train_nymeria_random_text/results_save.pkl',
         nymeria_mask=True,
         ),
    dict(type='AgrolMPJPE', mode='p-mpjpe',
         motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt',
         nymeria_mask=True,
         ),
]
test_evaluator = val_evaluator
