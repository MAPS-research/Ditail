# Ditail
Official implementation for "Diffusion Cocktail: Fused Generation from Diffusion Models".

<p align="center">
  <img src="./img/header.png" alt="Ditail Header Figure">
</p>


## Quick Links
 - [Project Page](https://MAPS-research.github.io/Ditail)
 - [Paper Preprint](https://arxiv.org/abs/2312.08873)
 - [HuggingFace Demo](https://huggingface.co/spaces/MAPS-research/Diffusion-Cocktail)
 - Stable Diffusion WebUI Extension (coming soon)

## Abstract
Diffusion models excel at generating high-quality images and are easy to extend, making them extremely popular among active users who have created an extensive collection of diffusion models with various styles by fine-tuning base models such as Stable Diffusion. Recent work has focused on uncovering semantic and visual information encoded in various components of a diffusion model, enabling better generation quality and more fine-grained control. However, those methods target improving a single model and overlook the vastly available collection of fine-tuned diffusion models. In this work, we study the combinations of diffusion models. We propose Diffusion Cocktail (Ditail), a training-free method that can accurately transfer content information between two diffusion models. This allows us to perform diverse generations using a set of diffusion models, resulting in novel images that are unlikely to be obtained by a single model alone. We also explore utilizing Ditail for style transfer, with the target style set by a diffusion model instead of an image. Ditail offers a more detailed manipulation of the diffusion generation, thereby enabling the vast community to integrate various styles and contents seamlessly and generate any content of any style.

**TL;DR:** Ditail offers a training-free method for novel image generations and fine-grained manipulations of content/style, enabling flexible integrations of existing pre-trained Diffusion models and LoRAs.

## Environment Setup
```bash
# Clone the repo
git clone https://github.com/MAPS-research/Ditail.git && cd Ditail

# Create a new conda environment
conda env create -f env.yml

# Activate and verify
conda activate ditail
conda list
```

## Ditail Demo
Ditail enpowers flexible image style transfer and content manipulation. You may want to play with:
- Content manipulation: `--pos_prompt`, `--neg_prompt`, `--alpha`, `--beta`, `--no_injection`.
- Style transfer: `--inv_model`, `--spl_model`, `--lora`, `lora_scale`.
- Granularity: `--inv_steps`, `--spl_steps`, `--omega`.
```bash
# Sample usage with editing prompt
python src/ditail_demo.py --img_path ./img/watch.jpg \
    --pos_prompt "a golden leather watch with dial and crystals"

# Sample usage with multiple model checkpoints
# Note: See ./model/README.md for more setup details
python src/ditail_demo.py --img_path ./img/watch.jpg \
    --pos_prompt "a golden leather watch with dial and crystals" \
    --spl_model "stablediffusionapi/pastel-mix-stylized-anime"

# Sample usage with LoRA
# Note: See ./lora/README.md for more setup details
python src/ditail_demo.py --img_path ./img/watch.jpg --lora pop \
    --pos_prompt "a golden leather watch with dial and crystals"
```

## Ditail for Batch Image Manipulation
**Note:** Make sure the dataset is ready, see `./data/README.md`.
```bash
# Sample usage on generated images
# Note: See ./model/README.md for more setup details
python src/ditail_batch.py --data_type gen --exp_id test \
    --inv_model ./model/output/38765 \
    --spl_model ./model/output/97557 \
    --data_dir ./data/gemrec/38765
# => Output dir: ./output/gen_test

# Sample usage on real images with LoRA
# Note: See ./lora/README.md for more setup details
python src/ditail_batch.py --data_type real --exp_id test \
    --data_dir ./data/coco \
    --lora pop
# => Output dir: ./output/real_test
```

## Acknowledgement
This work is supported in part by the Shanghai Frontiers Science Center of Artificial Intelligence and Deep Learning at NYU Shanghai, NYU Shanghai Boost Fund, and NYU HPC resources and services.

## Citation
If you find our work helpful, please consider cite it as follows:
```bibtex
@article{liu2023ditail,
  title={Diffusion Cocktail: Fused Generation from Diffusion Models},
  author={Liu, Haoming and Guo, Yuanhe and Wang, Shengjie and Wen, Hongyi},
  journal={arXiv preprint arXiv:2312.08873},
  year={2023}
}
```
