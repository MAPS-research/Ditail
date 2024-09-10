import os
import torch
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from ditail_utils import seed_everything
from helper.extractor import VitExtractor
from helper.fid_score import calculate_fid_given_paths

COCO_METADATA = './data/coco/metadata.pt'
GEMREC_METADATA = './data/gemrec/38765/promptbook.csv'

class BatchCLIP:
    def __init__(self, clip_id='openai/clip-vit-large-patch14', device='cuda'):
        self.device = torch.device(device)
        print('[INFO] Loading CLIP model:', clip_id)
        self.processor = CLIPProcessor.from_pretrained(clip_id)
        self.resize = transforms.Resize((224, 224), interpolation=3)
        self.model = CLIPModel.from_pretrained(clip_id).to(self.device)
        self.model.eval()
        print('[INFO] CLIP model loaded')
        self.init_prompts()

    def init_prompts(self):
        self.coco_data = torch.load(COCO_METADATA)
        self.coco_prompts = [self.coco_data[i]['caption'] for i in range(300)]
        self.gemrec_data = pd.read_csv(GEMREC_METADATA)
        self.gemrec_prompts = [row.prompt for row in self.gemrec_data.itertuples()]

    def compute_clip_scores(self, data_dir, data_type='gen'):
        if data_type == 'gen':
            prompts = self.gemrec_prompts
            images = [self.resize(Image.open(os.path.join(data_dir, f'{i}.png'))) \
                      for i in range(1, 91)]
        elif data_type == 'real_src':
            prompts = self.coco_prompts
            images = [self.resize(Image.open(os.path.join(data_dir, f'{i}.jpg'))) \
                      for i in range(1, 301)]
        elif data_type == 'real_tgt':
            prompts = self.coco_prompts
            images = [self.resize(Image.open(os.path.join(data_dir, f'{i}.png'))) \
                      for i in range(1, 301)]
        else:
            raise ValueError(f'Invalid data type: {data_type}')
        with torch.no_grad():
            inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
            output = self.model(**{k:v.to(self.device) for k, v in inputs.items()})
            scores = torch.diag(output.logits_per_text).detach().cpu()
        return scores.mean().item() / 100

class DINO:
    def __init__(self, dino_id='dino_vitb8', device='cuda'):
        self.device = torch.device(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        print('[INFO] Loading DINO model:', dino_id)
        self.extractor = VitExtractor(model_name=dino_id, device=device)
        print('[INFO] DINO model loaded')

    def load_img(self, img_dir, img_name):
        return self.transform(Image.open(os.path.join(img_dir, img_name)))

    def compute_ssim_loss(self, src_dir, tgt_dir, data_type='gen'):
        if data_type == 'gen':
            src_images = [self.load_img(src_dir, f'{i}.png').to(self.device) \
                          for i in range(1, 91)]
            tgt_images = [self.load_img(tgt_dir, f'{i}.png').to(self.device) \
                          for i in range(1, 91)]
        elif data_type == 'real':
            src_images = [self.load_img(src_dir, f'{i}.jpg').to(self.device) \
                          for i in range(1, 301)]
            tgt_images = [self.load_img(tgt_dir, f'{i}.png').to(self.device) \
                          for i in range(1, 301)]
        else:
            raise ValueError(f'Invalid data type: {data_type}')
        ssim_loss = 0.0
        get_keys = self.extractor.get_keys_self_sim_from_input
        for src_img, tgt_img in zip(src_images, tgt_images):
            with torch.no_grad():
                src_keys = get_keys(src_img.unsqueeze(0), layer_num=11)
                tgt_keys = get_keys(tgt_img.unsqueeze(0), layer_num=11)
            ssim_loss += F.mse_loss(src_keys, tgt_keys)
        return (ssim_loss / len(src_images)).item()

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
