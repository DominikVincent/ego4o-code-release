"""Bridge the format gap between the released 25fps script and IMUPoserDataset.

imuposer_process_25fps.py writes each file as a list-of-dicts (one dict per sequence).
IMUPoserDataset.load_data expects a dict-of-lists: fdata["acc"][i], fdata["ori"][i], etc.
This transposes in place. Idempotent (skips already-transposed files).
"""
import os
import torch

D = '/home/dominik/Documents/ego4o_data/IMUPoser/data/processed_imuposer_25fps'
KEYS = ['acc', 'ori', 'pose', 'joint', 'shape', 'tran']

for fname in sorted(os.listdir(D)):
    if not fname.endswith('.pt'):
        continue
    fpath = os.path.join(D, fname)
    data = torch.load(fpath)
    if isinstance(data, dict):
        print(f'{fname}: already dict-of-lists, skip')
        continue
    # list-of-dicts -> dict-of-lists
    out = {k: [seq[k] for seq in data] for k in KEYS}
    torch.save(out, fpath)
    print(f'{fname}: transposed {len(data)} sequences')
print('done')
