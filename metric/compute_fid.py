
import os
import torch
import random
import numpy as np

from helper.fid_score import calculate_fid_given_paths

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_fid(path1, path2, data_type='gen'):
    seed_everything(seed=42)
    fid_value = calculate_fid_given_paths(
        dims=2048,
        data_type=data_type,
        paths=[path1, path2],
        batch_size=(90 if data_type == 'gen' else 300),
        num_workers=min(len(os.sched_getaffinity(0)), 8),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    return fid_value

if __name__ == '__main__':
    # sample usage
    fid_gen = compute_fid(
        path1='../src/output_gen/[others_to_38765]',
        path2='../data/gemrec/38765',
        data_type='gen'
    )
    print(f'FID to target style images (generated): {fid_gen:.4f}')
    fid_real = compute_fid(
        path1='../src/output_real/[real_to_38765]',
        path2='../data/coco_pseudo_target/38765',
        data_type='real'
    )
    print(f'FID to target style images (real): {fid_real:.4f}')
