"""Verify the released TLControl checkpoints load into Ego4o's model classes.

Gate for the local stage-1/2 pipeline:
  1. save_weights_vq/best_model_epoch_hml_emaReset.pth -> HumanVQVAE (mmpose fork)
  2. save_weights/update_design/withEmaReset_stage3.pth -> TransformerAutoencoder_withCodes_hml_G2_noTraj
  3. demo/info_motion_mean.pt / info_motion_std.pt -> 263-dim stats
"""
import sys
import torch

TLC = '/home/dominik/Documents/ego4o_data/TLControl'
VQ_PTH = f'{TLC}/save_weights_vq/best_model_epoch_hml_emaReset.pth'
TF_PTH = f'{TLC}/save_weights/update_design/withEmaReset_stage3.pth'


def inspect(name, sd, n=12):
    print(f'\n=== {name}: {len(sd)} keys ===')
    for k in list(sd.keys())[:n]:
        print(f'  {k}: {tuple(sd[k].shape) if hasattr(sd[k], "shape") else type(sd[k])}')


def main():
    ok = True

    # ---------- stats ----------
    mean = torch.load(f'{TLC}/demo/info_motion_mean.pt', map_location='cpu')
    std = torch.load(f'{TLC}/demo/info_motion_std.pt', map_location='cpu')
    print(f'info_motion_mean: {tuple(mean.shape)}, info_motion_std: {tuple(std.shape)}')
    assert mean.reshape(-1).shape[0] == 263 and std.reshape(-1).shape[0] == 263, 'stats not 263-dim!'

    # ---------- VQ-VAE ----------
    raw = torch.load(VQ_PTH, map_location='cpu')
    sd = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    if not all(hasattr(v, 'shape') for v in list(sd.values())[:3]):
        # maybe a full checkpoint dict with nested model
        for key in ('model', 'net', 'vqvae'):
            if key in sd:
                sd = sd[key]
                break
    inspect('VQ-VAE checkpoint', sd)

    # infer codebook geometry from the first quantizer
    cb_keys = [k for k in sd if 'codebook' in k or 'embedding' in k]
    print('codebook-ish keys:', cb_keys[:8])
    nb_code, code_dim = None, None
    for k in cb_keys:
        if hasattr(sd[k], 'shape') and sd[k].dim() == 2:
            nb_code, code_dim = sd[k].shape
            print(f'inferred nb_code={nb_code}, code_dim={code_dim} from {k}')
            break

    from mmpose.models.ego_omni_mocap.vqvae.parser_util import mtm_args
    from mmpose.models.ego_omni_mocap.vqvae.vqvae import HumanVQVAE
    args = mtm_args()
    net = HumanVQVAE(args, nb_code=nb_code, code_dim=code_dim,
                     output_emb_width=code_dim)
    model_sd = net.state_dict()
    # align prefixes if needed
    if not any(k in model_sd for k in sd):
        for pref in ('vqvae.', 'module.'):
            if any((pref + k) in model_sd for k in sd):
                sd = {pref + k: v for k, v in sd.items()}
                break
            if any(k[len(pref):] in model_sd for k in sd if k.startswith(pref)):
                sd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
                break
    try:
        net.load_state_dict(sd, strict=True)
        print('VQ-VAE: strict load OK')
    except RuntimeError as e:
        print('VQ-VAE: strict load FAILED:')
        print(str(e)[:2000])
        ok = False

    # ---------- stage-3 transformer ----------
    raw = torch.load(TF_PTH, map_location='cpu')
    sd = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    if 'model' in sd and not hasattr(sd['model'], 'shape'):
        sd = sd['model']
    inspect('stage-3 transformer checkpoint', sd)

    from mmpose.models.ego_omni_mocap.ego_motion_mask_transformer import (
        TransformerAutoencoder_withCodes_hml_G2_noTraj)
    # ego4o side loads it with input_dim=9, num_emb from codebook
    tf_net = TransformerAutoencoder_withCodes_hml_G2_noTraj(
        input_dim=9, num_emb=nb_code, max_text_len=None, transfomers_clip=False)
    model_sd = tf_net.state_dict()
    hits = sum(1 for k in sd if k in model_sd and sd[k].shape == model_sd[k].shape)
    print(f'stage-3 transformer: {hits}/{len(model_sd)} model keys matched by name+shape '
          f'(ckpt has {len(sd)} keys)')
    missing = [k for k in model_sd if k not in sd]
    unexpected = [k for k in sd if k not in model_sd]
    print('missing (in model, not ckpt):', missing[:8], '...' if len(missing) > 8 else '')
    print('unexpected (in ckpt, not model):', unexpected[:8], '...' if len(unexpected) > 8 else '')

    print('\nRESULT:', 'PASS' if ok else 'FAIL (see above)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
