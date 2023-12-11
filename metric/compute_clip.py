
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

COCO_METADATA = '../data/coco/metadata.pt'
GEMREC_METADATA = '../data/gemrec/38765/promptbook.csv'

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
            scores = output.logits_per_text[0].detach().cpu()
        return scores.mean().item() / 100

if __name__ == '__main__':
    # sample usage
    clip = BatchCLIP()
    clip_gen = clip.compute_clip_scores(
        data_dir='../data/gemrec/38765',
        data_type='gen'
    )
    print(f'CLIP cosine similarity on generated images: {clip_gen:.4f}')
    clip_real = clip.compute_clip_scores(
        data_dir='../data/coco',
        data_type='real_src'
    )
    print(f'CLIP cosine similarity on real images: {clip_real:.4f}')
