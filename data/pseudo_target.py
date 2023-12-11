
import os
import torch
import random
import argparse
import numpy as np

from tqdm import tqdm
from diffusers import DiffusionPipeline

METADATA_PATH = './coco/metadata.pt'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def generate_single_image(pipeline, height, width, prompt, steps=49, seed=42):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    image = pipeline(
        prompt=prompt,
        negative_prompt='',
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    return image

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mvid', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=49)
    parser.add_argument('--save_dir', type=str, default='./coco_pseudo_target')
    parser.add_argument('--model_dir', type=str, default='../model/output')
    opt = parser.parse_args()

    # init random seed
    seed_everything(opt.seed)

    # init model pipeline
    model_path = os.path.join(opt.model_dir, opt.mvid)
    pipeline = DiffusionPipeline.from_pretrained(model_path, safety_checker=None, custom_pipeline='lpw_stable_diffusion')
    pipeline.to('cuda')

    # init metadata
    metadata = torch.load(METADATA_PATH)

    # generate images
    proc = lambda x: x - (x % 8)
    output_dir = os.path.join(opt.save_dir, opt.mvid)
    os.makedirs(output_dir, exist_ok=True)
    for i, d in enumerate(tqdm(metadata)):
        image = generate_single_image(
            pipeline=pipeline,
            width=proc(d['width']),
            height=proc(d['height']),
            prompt=d['caption'],
            steps=opt.steps,
            seed=opt.seed
        )
        image.save(os.path.join(output_dir, f'{i+1}.png'))