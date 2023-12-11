
import os
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from helper.extractor import VitExtractor

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

if __name__ == '__main__':
    # sample usage
    dino = DINO()
    ssim_gen = dino.compute_ssim_loss(
        src_dir='../data/gemrec/38765',
        tgt_dir='../src/output_gen/[38765_to_others]',
        data_type='gen'
    )
    print(f'DINO self-similarity on generated images: {ssim_gen:.4f}')
    ssim_real = dino.compute_ssim_loss(
        src_dir='../data/coco',
        tgt_dir='../src/output_real/[real_to_others]',
        data_type='real'
    )
    print(f'DINO self-similarity on real images: {ssim_real:.4f}')