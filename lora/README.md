# LoRA Preparation

## LoRAs Used in Ditail Paper
We collected a few LoRAs from [Civitai](https://civitai.com/) and [Liblib](https://www.liblib.ai/):
 - [Pop](https://civitai.com/models/161450/pop-art) (rename as `pop.safetensors`)
 - [Flat](https://www.liblib.art/modelinfo/76dcb8b59d814960b0244849f2747a15) (rename as `flat.safetensors`)
 - [Snow](https://www.liblib.art/modelinfo/f732b23b02f041bdb7f8f3f8a256ca8b) (rename as `snow.safetensors`)

## Add More LoRAs
 - Place them under `./lora` and rename as `[lora_key].safetensors`
 - Update the TRIGGER_WORD dict in `./src/ditail_utils.py`: {[lora_key]: [trigger_word]}
 - Use it at inference time: `--lora [lora_key]`

