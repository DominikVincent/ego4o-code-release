_base_ = ['../_base_/default_runtime.py']

# wandb logger
visualizer = dict(type='Visualizer', vis_backends=[dict(type='WandbVisBackend',
                                                        init_kwargs=dict(project='test_nymeria_speed'))])

# runtime
train_cfg = dict(max_epochs=25, val_interval=1)

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
pretrained_vqvae_path = '/home/jianwang/EgoMocap/work/EgoOmniMocap/work_dirs/train_nymeria_vqvae_4096_64/best_C-MPJPE_epoch_30.pth'
vqvae_model = dict(
    type='HumanVQVAE',
    args=None,
    nb_code=4096,
    code_dim=64,
    output_emb_width=64,
    down_t=2,
    stride_t=2,
    width=512,
    depth=3,
    dilation_growth_rate=3,
    activation='relu',
    norm=None,
    init_cfg=dict(type='Pretrained', checkpoint=pretrained_vqvae_path)
)

model = dict(
    type='IMUPoserEncoder',
    input_dim=3 + 6,
    sensor_num=6,
    seq_len=148,
    drop_out=0.1,
    num_emb=4096,
    text_mask_rate=0,
    image_mask_rate=0,
    recon_loss_weight=1,
    pretrained_transformer=pretrained_transformer_path,
    pretrained_vqvae=pretrained_vqvae_path,
    with_post_optimization=True,
    init_cfg=None,
    max_text_len=None,  # this is important !!!
    vqvae_dict=vqvae_model,
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
    dict(type='AddDummyText', text_name='text', dummy_text=''),

    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(224, 224)),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor',
         keys=['init_aligned_imu_acc', 'init_aligned_imu_ori', 'imu_acc', 'imu_ori',
               'init_aligned_global_smpl_joints', 'motion_hml',
               'joints']),
    # dict(type='AddGeneratedText',
    #      text_json_file='/CT/EgoMocap/work/LLaVA/eval_out/test_nymeria_all10_29_00_12_29/result.json'
    #      ),

    dict(
        type='Collect',
        keys=['motion_hml', 'lengths', 'init_aligned_imu_acc', 'init_aligned_imu_ori', 'text', 'img'],
        meta_keys=['sent_len', 'init_aligned_global_smpl_joints', 'motion_file', 'motion_id', 'combo_name'],
        meta_name='data_samples'
    )
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

# imuposer_dataset_wo_text = dict(
#     type='NymeriaDataset',
#     dataset_dir='/scratch/inf0/user/jianwang/nymeria',
#     pipeline=train_pipeline,
#     seq_len=197,
#     min_seq_len=40,
#     split='train',
#     random_mask=True,
#     signal_num=6,
#     tlcontrol_joint_sequence=True,
#     with_text=False,
# )

imuposer_dataset_w_text = dict(
    type='NymeriaDataset',
    dataset_dir='/HPS/EgoSyn/static00/nymeria',
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

tlcontrol_test_combos = {  # root 0, head 1 ,left wrist 2 , right wrist 3 , left hip 4 , right hip 5
    'lw_rw_h': [1, 2, 3],
    'rw_lp_rp': [3, 4, 5],
    'lw_rw_rp': [2, 3, 5],
    'lw_rp_h': [1, 2, 5],
    'rw_rp_h': [3, 5, 1],
    'lw_lp_rp': [2, 4, 5],
    'lw_rw_lp': [2, 3, 4],
    'lw_lp_h': [2, 4, 1],
    'rw_lp_h': [3, 4, 1],
    'lw_rw': [2, 3],
    'lw_lp': [2, 4],
    'lw_rp': [2, 5],
    'lw_h': [2, 1],
    'rw_lp': [3, 4],
    'rw_rp': [3, 5],
    'rw_h': [3, 1],
    'lp_rp': [4, 5],
    'lp_h': [4, 1],
    'rp_h': [5, 1],
    'lw': [2],
    'rw': [3],
    'lp': [4],
    'rp': [5],
    'h': [1]
}

val_dataset = dict(
    type='NymeriaDataset',
    dataset_dir='/HPS/EgoSyn/static00/nymeria',
    pipeline=train_pipeline,
    seq_len=150,
    min_seq_len=150,
    split='test',
    random_mask=True,
    combo_name='wo_global',
    signal_num=6,
    tlcontrol_joint_sequence=True,
    with_text=True,
    test_data_len=100,
)

val_dataset_list = []
for tl_control_key in tlcontrol_test_combos.keys():
    val_dataset_list.append(
        dict(
            type='NymeriaDataset',
            dataset_dir='/HPS/EgoSyn/static00/nymeria',
            pipeline=train_pipeline,
            seq_len=150,
            min_seq_len=150,
            split='test',
            random_mask=False,
            combo_name=tl_control_key,
            signal_num=6,
            tlcontrol_joint_sequence=True,
            with_text=True,
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='ConcatDataset',
        # datasets=val_dataset_list
        datasets=[val_dataset]
    ),
)
test_dataloader = val_dataloader

# evaluators

val_evaluator = [
    dict(type='EgoOmniMocapError', mode=('pa-mpjpe', 'c-mpjpe', 'jitter-pred', 'jitter-gt',  'global-angle'),
         motion_mean_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_mean.pt',
         motion_std_path='/CT/EgoMocap/work/EgoOmniMocap/projects/TLControl/demo/info_motion_std.pt',
         # save_path='/CT/EgoMocap/work/EgoOmniMocap/work_dirs/for_visualization/results_save_lp_h_text_image.pkl'
         ),
]

test_evaluator = val_evaluator
