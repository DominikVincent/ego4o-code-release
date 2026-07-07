_base_ = ['../_base_/default_runtime.py']

# ---- local paths ----
TLC = '/home/dominik/Documents/ego4o_data/TLControl'
IMUPOSER_ROOT = '/home/dominik/Documents/ego4o_data/IMUPoser'
MOTION_MEAN = f'{TLC}/demo/info_motion_mean.pt'
MOTION_STD = f'{TLC}/demo/info_motion_std.pt'
pretrained_transformer_path = f'{TLC}/save_weights/update_design/withEmaReset_stage3.pth'
pretrained_vqvae_path = f'{TLC}/save_weights_vq/best_model_epoch_hml_emaReset.pth'
WORK_DIR = '/home/dominik/Documents/ego4o_data/work_dirs/imuposer_mask_local'

# no wandb (local-only)
visualizer = dict(type='Visualizer', vis_backends=[dict(type='LocalVisBackend')])

# runtime
train_cfg = dict(max_epochs=30, val_interval=1)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-4))

param_scheduler = [
    dict(type='StepLR', step_size=100000, gamma=0.96, end=80, by_epoch=False)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='C-MPJPE', rule='less'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
    badcase=dict(type='BadCaseAnalysisHook', enable=False, out_dir='badcase',
                 metric_type='loss', badcase_thr=5)
)

custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='SyncBuffersHook')
]

# model (num_emb=128 matches the released TLControl VQ-VAE)
model = dict(
    type='IMUPoserEncoder',
    input_dim=3 + 6,
    sensor_num=6,
    seq_len=196,
    drop_out=0.1,
    num_emb=128,
    text_mask_rate=0.5,
    recon_loss_weight=1,
    pretrained_transformer=pretrained_transformer_path,
    pretrained_vqvae=pretrained_vqvae_path,
    with_post_optimization=False,
    init_cfg=None
)

train_pipeline = [
    dict(type='InitAlignIMUMotion', imu_acc_name='imu_acc', imu_ori_name='imu_ori', joint_name='joints'),
    dict(type='RotationMatrixTo6D', rotation_name='init_aligned_imu_ori'),
    dict(type='RotationMatrixTo6D', rotation_name='imu_ori'),
    dict(type='HMLMotionRepresentation', joint_name='init_aligned_global_smpl_joints',
         drop_last_pose_name_list=('init_aligned_imu_acc', 'init_aligned_imu_ori',
                                   'init_aligned_global_smpl_joints')),
    dict(type='NormalizeHMLMotion', hml_motion_name='motion_hml',
         hml_mean_path=MOTION_MEAN, hml_std_path=MOTION_STD),
    dict(type='PadMotion', seq_len=196,
         pad_name_list=('init_aligned_imu_acc', 'init_aligned_imu_ori', 'motion_hml'),
         resize_input_sequence=True),
    dict(type='ChangeHMLShape', hml_motion_name='motion_hml'),
    dict(type='ToTensor',
         keys=['init_aligned_imu_acc', 'init_aligned_imu_ori', 'imu_acc', 'imu_ori',
               'init_aligned_global_smpl_joints', 'motion_hml',
               'smpl_pose', 'joints', 'shape', 'transl']),
    dict(type='Collect',
         keys=['motion_hml', 'lengths', 'init_aligned_imu_acc', 'init_aligned_imu_ori'],
         meta_keys=['sent_len', 'init_aligned_global_smpl_joints'],
         meta_name='data_samples')
]
val_pipeline = train_pipeline
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=128, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(type='IMUPoserDataset', imuposer_root_dir=IMUPOSER_ROOT,
                 pipeline=train_pipeline, seq_len=197, min_seq_len=197,
                 split='train', hold_out_val=True, random_mask=True,
                 signal_num=6, tlcontrol_joint_sequence=True))

val_dataloader = dict(
    batch_size=64, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(type='IMUPoserDataset', imuposer_root_dir=IMUPOSER_ROOT,
                 pipeline=val_pipeline, seq_len=197, min_seq_len=197,
                 split='val', random_mask=True,
                 signal_num=6, tlcontrol_joint_sequence=True))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='AgrolMPJPE', mode='c-mpjpe', motion_mean_path=MOTION_MEAN, motion_std_path=MOTION_STD,
         save_path=f'{WORK_DIR}/results_save.pkl'),
    dict(type='AgrolMPJPE', mode='p-mpjpe', motion_mean_path=MOTION_MEAN, motion_std_path=MOTION_STD),
]
test_evaluator = val_evaluator

work_dir = WORK_DIR
