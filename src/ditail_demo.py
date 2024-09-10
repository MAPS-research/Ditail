import os
import yaml
import argparse
import warnings
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from ditail_utils import *

# filter warnings
logging.set_verbosity_error()
warnings.filterwarnings('ignore', message='.*deprecated.*')

class DitailDemo(nn.Module):
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
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            image = 2 * image - 1
            posterior = self.vae.encode(image).latent_dist
            latent = posterior.mean * 0.18215
        return latent

    @torch.no_grad()
    def invert_image(self, cond, latent):
        self.latents = {}
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
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
        self.noisy_latent = latent

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
        with torch.autocast(device_type=self.device, dtype=torch.float32):
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
        register_mask(self, self.mask, self.noisy_latent)

    @torch.no_grad()
    def sampling_loop(self):
        # init text embeddings
        text_ept = self.tokenizer('', **self.tokenizer_kwargs)
        self.emb_ept = self.text_encoder(text_ept.input_ids.to(self.device))[0]
        self.emb_spl = torch.cat([self.emb_ept, self.emb_pos, self.emb_neg], dim=0)
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            # use noisy latent as starting point
            x = self.latents[self.scheduler.timesteps[0].item()]
            # sampling loop
            for t in tqdm(self.scheduler.timesteps):
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

    def run_ditail(self):
        # init output dir & dump config
        self.save_dir = get_save_dir(self.output_dir, self.img_path)
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(vars(self.args)))
        # step 1: inversion stage
        self.load_inv_model()
        self.extract_latents()
        self.latent_to_image(
            latent=self.noisy_latent,
            save_path=os.path.join(self.save_dir, 'noise.png')
        )
        # step 2: sampling stage
        self.load_spl_model()
        if not self.no_injection:
            self.init_injection()
        self.sampling_loop()
        self.latent_to_image(
            latent=self.output_latent,
            save_path=os.path.join(self.save_dir, 'output.png')
        )

def main(args):
    seed_everything(args.seed)
    ditail = DitailDemo(args)
    ditail.run_ditail()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./output/demo')
    parser.add_argument('--inv_model', type=str, default='stablediffusionapi/realistic-vision-v51',
                        help='Pre-trained inversion model name or path (step 1)')
    parser.add_argument('--spl_model', type=str, default='stablediffusionapi/realistic-vision-v51',
                        help='Pre-trained sampling model name or path (step 2)')
    parser.add_argument('--inv_steps', type=int, default=50,
                        help='Number of inversion steps (step 1)')
    parser.add_argument('--spl_steps', type=int, default=50,
                        help='Number of sampling steps (step 2)')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to the source image')
    parser.add_argument('--pos_prompt', type=str, required=True,
                        help='Positive prompt for inversion')
    parser.add_argument('--neg_prompt', type=str, default='worst quality, blurry, NSFW',
                        help='Negative prompt for inversion')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Positive prompt scaling factor')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Negative prompt scaling factor')
    parser.add_argument('--omega', type=float, default=15,
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
    args = parser.parse_args()
    main(args)