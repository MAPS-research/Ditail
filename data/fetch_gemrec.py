
import os
import requests
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login

SAVE_DIR = './gemrec'

def load_hf_dataset():
    login(token=os.environ.get('HF_TOKEN'))
    roster = pd.DataFrame(load_dataset('MAPS-research/GEMRec-Roster', split='train', use_auth_token=True))
    promptBook = pd.DataFrame(load_dataset('MAPS-research/GEMRec-Metadata', split='train', use_auth_token=True))
    return roster, promptBook

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    roster, promptBook = load_hf_dataset()
    target_ids = [38765, 87747, 96373, 97557, 1000005]
    for target_id in tqdm(target_ids):
        if os.path.exists(f'{SAVE_DIR}/{target_id}'):
            continue
        else:
            os.makedirs(f'{SAVE_DIR}/{target_id}')
        df = promptBook[promptBook['modelVersion_id'] == target_id].sort_values(by='prompt_id')
        df.to_csv(f'{SAVE_DIR}/{target_id}/promptbook.csv', index=False)
        for i, iid in enumerate(df['image_id'].tolist()):
            response = requests.get(f'https://modelcofferbucket.s3-accelerate.amazonaws.com/{iid}.png')
            with open(f'{SAVE_DIR}/{target_id}/{i+1}.png', 'wb') as f:
                f.write(response.content)
    roster.to_csv(f'{SAVE_DIR}/roster.csv', index=False)

if __name__ == '__main__':
    main()