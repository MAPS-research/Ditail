
import os
import torch
import random
import requests
from pycocotools.coco import COCO

K = 300
SAVE_DIR = './coco'

def fetch_image(url, save_path):
    # Send a GET request to the image URL
    response = requests.get(url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the save_path as a file with write-binary ('wb') mode
        with open(save_path, 'wb') as f:
            # Iterate over the response data stream and write to the file
            for chunk in response.iter_content(chunk_size=8192):
                # If chunk is not None or empty, write to file
                if chunk:
                    f.write(chunk)
        return
    # Cast error if any step fails
    raise ValueError("Fetching failed")

def main():
    # init coco tools
    # !wget https://huggingface.co/datasets/merve/coco/resolve/main/annotations/captions_train2017.json
    coco = COCO('./captions_train2017.json')
    # sample a subset of images with metadata
    imgIds = []
    random.seed(0)
    raw_imgIds = coco.getImgIds()
    random.shuffle(raw_imgIds)
    raw_metadata = coco.loadImgs(raw_imgIds[:K*10])
    for i in range(len(raw_metadata)):
        if raw_metadata[i]['license'] in (2, 4):
            imgIds.append(raw_imgIds[i])
        if len(imgIds) == K:
            break
    metadata = coco.loadImgs(imgIds)
    # add captions and fetch images
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i, iid in enumerate(imgIds):
        metadata[i]['caption'] = coco.loadAnns(coco.getAnnIds(iid))[0]['caption'].strip()
        save_path = os.path.join(SAVE_DIR, f'{i+1}.jpg')
        print(f"{i+1}: {metadata[i]['caption']}")
        if not os.path.exists(save_path):
            fetch_image(metadata[i]['coco_url'], save_path)
        metadata[i]['path'] = f'./data/coco/{i+1}.jpg'
    # save metadata
    torch.save(metadata, os.path.join(SAVE_DIR, 'metadata.pt'))

if __name__ == '__main__':
    main()