# Dataset Preparation

We use [GEMRec-18K](https://huggingface.co/datasets/MAPS-research/GEMRec-PromptBook) for generated images and [COCO Captions](https://cocodataset.org/#captions-2015) for real images.

## GEMRec-18K Subset
Default dir: `./gemrec`
```shell
# Note: set your HF_TOKEN before running the script

# Fetch a subset with 5x90 images (deterministic)
python fetch_gemrec.py
```

## COCO Caption 2017 Subset
Default dir: `./coco`
```shell
# Download metadata
wget https://huggingface.co/datasets/merve/coco/resolve/main/annotations/captions_train2017.json
# Fetch a subset with 300 images (deterministic)
python fetch_coco.py
```

## Pseudo Target for Real Images
Default dir: `./coco_pseudo_target`
```shell
# Generate pseudo targets given target model
python pseudo_target.py --mvid [target_mvid]
```