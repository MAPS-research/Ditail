# Model Preparation

Download DM checkpoints from [Civitai](https://civitai.com/) and convert them to [Diffusers](https://huggingface.co/docs/diffusers/index) format.

## Conversion Script Sample Usage
```shell
cd ./model
python convert_model.py --mvid [civitai_model_version_id]
```

## Dir Explanations
`./model/meta`: Metadata for the checkpoints. \
`./model/output`: Converted models in [Diffusers](https://huggingface.co/docs/diffusers/index) format. \
`./model/download`: Checkpoint cache downloaded from [Civitai](https://civitai.com/).

## Models Used in Ditail Paper & Demo
**Note:** All the non-SD checkpoints are finetuned from Stable Diffusion 1.5.
|          Model Name          |             Model String             | Style Keyword | Need Conversion |
|:----------------------------:|:------------------------------------:|:-------------:|:---------------:|
| [BluePastel](https://civitai.com/models/32333?modelVersionId=38765) | ./model/output/38765 | Anime | ✓ |
| [DiaMix](https://civitai.com/models/75949?modelVersionId=87747) | ./model/output/87747 | Fantasy | ✓ |
| [Little Illustration](https://civitai.com/models/15250?modelVersionId=96373) | ./model/output/96373 | Illustration | ✓ |
| [Chaos3.0](https://civitai.com/models/91534?modelVersionId=97557) | ./model/output/97557 | Abstract | ✓ |
| [RealisticVision](https://civitai.com/models/4201?modelVersionId=130072) | ./model/output/130072 | Realistic | ✓ |
| [Chaos3.0](https://huggingface.co/MAPS-research/Chaos3.0) | MAPS-research/Chaos3.0 | Abstract | ✗ |
| [PastelMix](https://huggingface.co/stablediffusionapi/pastel-mix-stylized-anime) | stablediffusionapi/pastel-mix-stylized-anime | Anime | ✗ |
| [RealisticVision](https://huggingface.co/stablediffusionapi/realistic-vision-v51) | stablediffusionapi/realistic-vision-v51 | Realistic | ✗ |
| [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | runwayml/stable-diffusion-v1-5 | Generic | ✗ |

## Add More Models
 - Convert it to Diffusers format (if needed) or use existing repo id
 - Set `--inv_model` or `--spl_model` to that model dir or repo id