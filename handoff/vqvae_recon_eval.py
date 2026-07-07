"""Pure VQ-VAE reconstruction error (encoder-decoder ceiling), no IMU.
GT motion_hml -> encode -> quantize -> decode -> joints, vs GT joints.
Compares to paper suppl: part-aware VQ-VAE 44.93mm MPJPE / 32.72mm PA-MPJPE.
"""
import numpy as np
import torch
from mmengine.config import Config
from mmpose.registry import DATASETS, MODELS
import mmpose.datasets, mmpose.models
from mmpose.datasets.datasets.ego_omni_mocap.humanml_utils.motion_process import recover_from_ric
from mmpose.evaluation.functional import keypoint_mpjpe

TLC = '/home/dominik/Documents/ego4o_data/TLControl'
cfg = Config.fromfile('/home/dominik/Documents/repos/ego4o-code-release/EgoOmniMocap/'
                      'configs/imuposer/train_imuposer_dataset_mask_local.py')
mean = torch.load(f'{TLC}/demo/info_motion_mean.pt')
std = torch.load(f'{TLC}/demo/info_motion_std.pt')

ds = DATASETS.build(cfg.val_dataloader['dataset'])
model = MODELS.build(cfg.model).cuda().eval()
vq = model.vq_net

def denorm(x):  # x: (263,1,T) -> (1,T,263) denormalized
    return (x.permute(1, 2, 0).float().cpu() * std + mean)

none_e, cen_e, pa_e = [], [], []
N = min(400, len(ds))
with torch.no_grad():
    for i in range(0, N, 64):
        batch = [ds[j] for j in range(i, min(i + 64, N))]
        mh = torch.stack([b['motion_hml'] for b in batch]).cuda()          # (B,263,1,T)
        x_out, _, _ = vq.forward_feature(mh)                                # reconstruction
        for k, b in enumerate(batch):
            L = int(b['lengths'])
            pred = recover_from_ric(denorm(x_out[k]), 22)[0][:L]            # (L,22,3)
            gt = b['data_samples']['init_aligned_global_smpl_joints'][:L]
            gt = torch.as_tensor(gt).float().numpy() if not torch.is_tensor(gt) else gt.float().numpy()
            pred = pred.numpy() if torch.is_tensor(pred) else np.asarray(pred)
            m = np.ones((L, 22), bool)
            none_e.append(keypoint_mpjpe(pred, gt, m, 'none'))
            cen_e.append(keypoint_mpjpe(pred, gt, m, 'center'))
            pa_e.append(keypoint_mpjpe(pred, gt, m, 'procrustes'))

print(f'VQ-VAE reconstruction over {len(none_e)} val clips (held-out ACCAD):')
print(f'  MPJPE   (none)      : {np.mean(none_e)*1000:.1f} mm')
print(f'  C-MPJPE (center)    : {np.mean(cen_e)*1000:.1f} mm')
print(f'  P-MPJPE (procrustes): {np.mean(pa_e)*1000:.1f} mm')
print(f'paper part-aware VQ-VAE: 44.93 mm MPJPE / 32.72 mm PA-MPJPE')
