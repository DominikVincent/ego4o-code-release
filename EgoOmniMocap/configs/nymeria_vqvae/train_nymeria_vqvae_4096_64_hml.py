# Nymeria VQ-VAE (4096 codes / 64 dim) trained on precomputed 30 fps HumanML3D
# features (GT-motion variant, no IMU). Local adaptation of
# train_nymeria_vqvae_4096_64.py: NymeriaHMLDataset + local paths + recomputed
# normalization stats. Stage C3 of the Ego4o reproduction; the resulting
# best_C-MPJPE checkpoint is what the LLM's motion encoder loads.
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

# ---------------- local paths ----------------
dataset_dir = '/local/home/dhollidt/data/ego4o_nymeria'
hml_mean_path = dataset_dir + '/info_motion_mean.pt'
hml_std_path = dataset_dir + '/info_motion_std.pt'
# TLControl release weights (handoff C1) — download with
#   gdown (see handoff/HANDOFF.md) into this directory before training.
pretrained_vqvae_path = '/local/home/dhollidt/data/ego4o_weights/TLControl/save_weights_vq/best_model_epoch_hml_emaReset.pth'

# model settings
model = dict(
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

# pipelines — features are precomputed; only normalize/pad/reshape here
train_pipeline = [
    dict(type='NormalizeHMLMotion', hml_motion_name='motion_hml',
         hml_mean_path=hml_mean_path,
         hml_std_path=hml_std_path),
    dict(type='PadMotion', seq_len=148,
         pad_name_list=('motion_hml', 'init_aligned_global_smpl_joints'),
         resize_input_sequence=True),
    dict(type='ChangeHMLShape', hml_motion_name='motion_hml'),
    dict(type='ToTensor',
         keys=['motion_hml', 'init_aligned_global_smpl_joints']),
    dict(
        type='Collect',
        keys=['motion_hml'],
        meta_keys=['init_aligned_global_smpl_joints', 'motion_file', 'motion_id', ],
        meta_name='data_samples'
    )
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

# data loaders
train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='NymeriaHMLDataset',
        dataset_dir=dataset_dir,
        pipeline=train_pipeline,
        seq_len=148,
        split='train',
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
        type='NymeriaHMLDataset',
        dataset_dir=dataset_dir,
        pipeline=val_pipeline,
        seq_len=148,
        split='val',
    )
)
test_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='NymeriaHMLDataset',
        dataset_dir=dataset_dir,
        pipeline=test_pipeline,
        seq_len=148,
        split='test',
    )
)

# evaluators
val_evaluator = [
    dict(type='AgrolMPJPE', mode='c-mpjpe',
         motion_mean_path=hml_mean_path,
         motion_std_path=hml_std_path,
         nymeria_mask=True,
         ),
    dict(type='AgrolMPJPE', mode='p-mpjpe',
         motion_mean_path=hml_mean_path,
         motion_std_path=hml_std_path,
         nymeria_mask=True,
         ),
]
test_evaluator = val_evaluator
