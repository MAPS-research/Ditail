import os
import yaml
import argparse
import warnings
import pandas as pd
from PIL import Image
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from ditail_utils import *
from ditail_metrics import *

# filter warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore', message='.*deprecated.*')

class DitailBatch(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        for k, v in vars(args).items():
            setattr(self, k, v)

    def load_inv_model(self):
        self.scheduler = DDIMScheduler.from_pretrained(self.inv_model, subfolder='scheduler')
        self.scheduler.set_timesteps(self.inv_steps, device=self.device)
        print(f'[INFO] Loading inversion model: {self.inv_model}')
        pipe = StableDiffusionPipeline.from_pretrained(
            self.inv_model, torch_dtype=torch.float16,
            use_safetensors=self.inv_model.endswith('.safetensors')
        ).to(self.device)
        pipe.enable_xformers_memory_efficient_attention()
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.tokenizer_kwargs = dict(
            truncation=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.tokenizer.model_max_length
        )
    
    def load_spl_model(self):
        self.scheduler = DDIMScheduler.from_pretrained(self.spl_model, subfolder='scheduler')
        self.scheduler.set_timesteps(self.spl_steps, device=self.device)
        print(f'[INFO] Loading sampling model: {self.spl_model} (LoRA = {self.lora})')
        if (self.lora != 'none') or (self.inv_model != self.spl_model):
            pipe = StableDiffusionPipeline.from_pretrained(
                self.spl_model, torch_dtype=torch.float16,
                use_safetensors=self.inv_model.endswith('.safetensors')
            ).to(self.device)
            if self.lora != 'none':
                pipe.load_lora_weights(self.lora_dir, weight_name=f'{self.lora}.safetensors')
                pipe.fuse_lora(lora_scale=self.lora_scale)
            pipe.enable_xformers_memory_efficient_attention()
            self.text_encoder = pipe.text_encoder
            self.tokenizer = pipe.tokenizer
            self.unet = pipe.unet
            self.vae = pipe.vae
            self.tokenizer_kwargs = dict(
                truncation=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.tokenizer.model_max_length
            )

    @torch.no_grad()
    def encode_image(self, img_path):
        image_pil = T.Resize(512)(Image.open(img_path).convert('RGB'))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            image = 2 * image - 1
            posterior = self.vae.encode(image).latent_dist
            latent = posterior.mean * 0.18215
        return latent

    @torch.no_grad()
    def invert_image(self, cond, latent):
        self.latents = {}
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i-1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample
                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                self.latents[t.item()] = latent
        self.latents['noisy'] = latent
        torch.save(self.latents, os.path.join(self.latent_dir, f'{self.pid}.pt'))

    @torch.no_grad()
    def extract_latents(self):
        # get the embeddings for pos & neg prompts
        if self.lora != 'none':
            self.pos_prompt += f', {TRIGGER_WORD[self.lora]}'
        text_pos = self.tokenizer(self.pos_prompt, **self.tokenizer_kwargs)
        text_neg = self.tokenizer(self.neg_prompt, **self.tokenizer_kwargs)
        self.emb_pos = self.text_encoder(text_pos.input_ids.to(self.device))[0]
        self.emb_neg = self.text_encoder(text_neg.input_ids.to(self.device))[0]
        # apply condition scaling
        cond = self.alpha * self.emb_pos - self.beta * self.emb_neg
        # encode source image & apply DDIM inversion
        self.invert_image(cond, self.encode_image(self.img_path))

    @torch.no_grad()
    def latent_to_image(self, latent, save_path):
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            latent = 1 / 0.18215 * latent
            image = self.vae.decode(latent).sample[0]
            image = (image / 2 + 0.5).clamp(0, 1)
        T.ToPILImage()(image).save(save_path)

    def init_injection(self, attn_ratio=0.5, conv_ratio=0.8):
        attn_thresh = int(attn_ratio * self.spl_steps)
        conv_thresh = int(conv_ratio * self.spl_steps)
        self.attn_inj_timesteps = self.scheduler.timesteps[:attn_thresh]
        self.conv_inj_timesteps = self.scheduler.timesteps[:conv_thresh]
        register_attn_inj(self, self.attn_inj_timesteps)
        register_conv_inj(self, self.conv_inj_timesteps)

    @torch.no_grad()
    def sampling_loop(self):
        # init text embeddings
        if self.lora != 'none':
            self.pos_prompt += f', {TRIGGER_WORD[self.lora]}'
        text_ept = self.tokenizer('', **self.tokenizer_kwargs)
        text_pos = self.tokenizer(self.pos_prompt, **self.tokenizer_kwargs)
        text_neg = self.tokenizer(self.neg_prompt, **self.tokenizer_kwargs)
        self.emb_ept = self.text_encoder(text_ept.input_ids.to(self.device))[0]
        self.emb_pos = self.text_encoder(text_pos.input_ids.to(self.device))[0]
        self.emb_neg = self.text_encoder(text_neg.input_ids.to(self.device))[0]
        self.emb_spl = torch.cat([self.emb_ept, self.emb_pos, self.emb_neg], dim=0)
        # init injection mask (optional)
        register_mask(self, self.mask, self.latents['noisy'])
        # noise sampling loop
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            # use noisy latent as starting point
            x = self.latents[self.scheduler.timesteps[0].item()]
            # sampling loop
            for t in self.scheduler.timesteps:
                # concat latents & register timestep
                src_latent = self.latents[t.item()]
                latents = torch.cat([src_latent, x, x])
                register_time(self, t.item())
                # apply U-Net for denoising
                noise_pred = self.unet(latents, t, encoder_hidden_states=self.emb_spl).sample
                # classifier-free guidance
                _, noise_pred_pos, noise_pred_neg = noise_pred.chunk(3)
                noise_pred = noise_pred_neg + self.omega * (noise_pred_pos - noise_pred_neg)
                # denoise step
                x = self.scheduler.step(noise_pred, t, x).prev_sample
            # save output latent
            self.output_latent = x

def main(args):
    # init experiment
    seed_everything(args.seed)
    ditail = DitailBatch(args)
    latent_dir = f'./latent/{args.data_type}_{args.latent_id}'
    output_dir = f'./output/{args.data_type}_{args.exp_id}'
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    setattr(ditail, 'latent_dir', latent_dir)
    setattr(ditail, 'output_dir', output_dir)
    # save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(vars(args)))
    # main loop
    if args.data_type == 'real':
        # init metadata
        metadata = torch.load(os.path.join(args.data_dir, 'metadata.pt'))
        ditail.neg_prompt = 'worst quality, blurry, NSFW'
        # step 1: inversion stage
        ditail.load_inv_model()
        for i in trange(len(metadata)):
            ditail.pid = i+1
            ditail.img_path = metadata[i]['path']
            ditail.pos_prompt = metadata[i]['caption']
            if not os.path.exists(os.path.join(latent_dir, f'{i+1}.pt')):
                ditail.extract_latents()
        # step 2: sampling stage
        ditail.load_spl_model()
        if not ditail.no_injection:
            ditail.init_injection()
        for i in trange(len(metadata)):
            ditail.pid = i+1
            ditail.pos_prompt = metadata[i]['caption']
            ditail.latents = torch.load(os.path.join(latent_dir, f'{i+1}.pt'))
            ditail.sampling_loop()
            ditail.latent_to_image(
                latent=ditail.output_latent,
                save_path=os.path.join(output_dir, f'{i+1}.png')
            )
    elif args.data_type == 'gen':
        # init metadata
        df = pd.read_csv(os.path.join(args.data_dir, 'promptbook.csv'))
        # step 1: inversion stage
        ditail.load_inv_model()
        for row in tqdm(list(df.itertuples())):
            ditail.pid = row.prompt_id
            ditail.pos_prompt = row.prompt
            ditail.neg_prompt = row.negativePrompt
            ditail.img_path = os.path.join(args.data_dir, f'{row.prompt_id}.png')
            ditail.extract_latents()
        # step 2: sampling stage
        ditail.load_spl_model()
        if not ditail.no_injection:
            ditail.init_injection()
        for row in tqdm(list(df.itertuples())):
            ditail.pid = row.prompt_id
            ditail.pos_prompt = row.prompt
            ditail.neg_prompt = row.negativePrompt
            ditail.latents = torch.load(os.path.join(latent_dir, f'{row.prompt_id}.pt'))
            ditail.sampling_loop()
            ditail.latent_to_image(
                latent=ditail.output_latent,
                save_path=os.path.join(output_dir, f'{row.prompt_id}.png')
            )
    else:
        raise ValueError(f'Invalid data type: {args.data_type}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_type', type=str, required=True, choices=['gen', 'real'],
                        help='Data type for batch image manipulation')
    parser.add_argument('--inv_model', type=str, default='stablediffusionapi/realistic-vision-v51',
                        help='Pre-trained inversion model name or path (step 1)')
    parser.add_argument('--spl_model', type=str, default='stablediffusionapi/realistic-vision-v51',
                        help='Pre-trained sampling model name or path (step 2)')
    parser.add_argument('--inv_steps', type=int, default=50,
                        help='Number of inversion steps (step 1)')
    parser.add_argument('--spl_steps', type=int, default=50,
                        help='Number of sampling steps (step 2)')
    parser.add_argument('--alpha', type=float, default=3.0,
                        help='Positive prompt scaling factor')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Negative prompt scaling factor')
    parser.add_argument('--omega', type=float, default=7.5,
                        help='Classifier-free guidance factor')
    parser.add_argument('--mask', type=str, default='full',
                        help='Optional mask for regional injection')
    parser.add_argument('--lora', type=str, default='none',
                        help='Optional LoRA for the sampling stage')
    parser.add_argument('--lora_dir', type=str, default='./lora',
                        help='Optional LoRA storing directory')
    parser.add_argument('--lora_scale', type=float, default=0.7,
                        help='Optional LoRA scaling weight')
    parser.add_argument('--no_injection', action="store_true",
                        help='Do not use PnP injection')
    clip = BatchCLIP()
    dino = DINO()
    for alpha in [1, 2, 3, 4, 8]:
        for omega in [7.5, 15.0]:
            # init config
            args = parser.parse_args()
            src_mvid = os.path.basename(args.inv_model)
            tgt_mvid = os.path.basename(args.spl_model)
            if args.data_type == 'gen':
                args.data_dir = f'./data/gemrec/{src_mvid}'
            else:
                args.data_dir = f'./data/coco'
            a, b, w = int(alpha), int(args.beta), int(omega)
            args.alpha, args.beta, args.omega = a, b, w
            args.exp_id = f'{src_mvid}_{tgt_mvid}_{a}_{b}_{w}'
            args.latent_id = f'{src_mvid}_{a}_{b}'
            # run experiment
            print('\n[Exp]:', args.exp_id)
            main(args)
            # evaluation
            output_dir = f'./output/{args.data_type}_{args.exp_id}'
            if args.data_type == 'gen':
                clip_src = clip.compute_clip_scores(args.data_dir, data_type=args.data_type)
                clip_out = clip.compute_clip_scores(output_dir, data_type=args.data_type)
            else:
                clip_src = clip.compute_clip_scores(args.data_dir, data_type=args.data_type + '_src')
                clip_out = clip.compute_clip_scores(output_dir, data_type=args.data_type + '_tgt')
            print(f'\n===> CLIP Score: src {clip_src:.4f} out {clip_out:.4f}')
            dino_out = dino.compute_ssim_loss(
                src_dir=args.data_dir,
                tgt_dir=output_dir,
                data_type=args.data_type
            )
            print(f'\n===> DINO Score: out {dino_out:.4f}')
            if args.data_type == 'gen':
                tgt_dir = f'./data/gemrec/{tgt_mvid}'
            else:
                tgt_dir = f'./data/coco_pseudo_target/{tgt_mvid}'
            fid_src = compute_fid(args.data_dir, tgt_dir, args.data_type)
            fid_out = compute_fid(output_dir, tgt_dir, args.data_type)
            print(f'\n===> FID Score: src {fid_src:.2f} out {fid_out:.2f}')
