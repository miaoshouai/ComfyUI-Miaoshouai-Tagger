# MiaoshouAI Tagger for ComfyUI

[English](README.md) / [中文](README_CN.md) / [日本語](README_JP.md)

MiaoshouAI Tagger for ComfyUI is an advanced image captioning tool based on the Microsoft Florence-2 Model Fine-tuned to perfection. This tool offers highly accurate and contextually relevant image tagging for your projects.

## Update Note
2026/07/31 Compatibility with transformers 5.x. The node crashed on load with `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`; fixing that uncovered four further breakages, three of which were silent — the model loaded without any error and produced garbage, looping output, or captions computed from unresized and unnormalized images. See [the details below](#transformers-5x-compatibility). transformers 4.x behaviour is unchanged.</br>
2025/12/16 Compatibility with transformers 4.51+, which no longer provides `generate()` on `PreTrainedModel`. Florence-2's modeling and configuration files are now vendored in this repository (`modeling_florence2.py`, `configuration_florence2.py`) and patched to inherit `GenerationMixin`, so the model is loaded locally instead of through `trust_remote_code`.</br>
2024/11/05 v1.4 A new release to support [Florence-2-base-PromptGen-v2.0](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0) and [Florence-2-large-PromptGen-v2.0](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v2.0)</br>
2024/09/28 v1.31 fix the configuration error rated to [this issue](https://github.com/miaoshouai/ComfyUI-Miaoshouai-Tagger/issues/15), try to delete your existing model from models\LLM folder and run again. It will automatically download the new configurations for you. Or you can download the model from the [baidu drive](https://pan.baidu.com/s/1h8kLNmukfcUitM7mKRE89w?pwd=4xwc) folder.</br>
2024/09/07 v1.2 updated to support [Florence-2-large-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5), a random prompt widget is added to Tagger node so that if you want to get a different prompt everytime, then just switch it to "always". <br>
2024/09/05 v1.1 updated to support [Florence-2-base-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5), 2 new prompt mode is added; a new node for flux clip text encoder is added to add easy support for flux model clips.

## Why Another Tagger?
While current taggers like WD14 perform reasonably well, they often produce errors that require manual correction. MiaoshouAI/Florence-2-base-PromptGen is fine-tuned on Microsoft's latest Florence2 model using a curated dataset from Civitai images and tags. This ensures that the tagging results are more aligned with the typical prompts used for generating images, enhancing accuracy and relevance.

## Why ComfyUI?
ComfyUI has emerged as one of the most popular node-based tools for Stable Diffusion workers. It offers various nodes and models, such as LLava and Ollama Vision nodes, for generating image captions and passing them to text encoders. However, these vision models are not specifically trained for prompting and image tagging. By using MiaoshouAI Tagger, you can see a clear improvement in results.

## Key Features
#### High Accuracy: 
Fine-tuned on selected high quality Civitai images and clean tags to produce highly accurate and contextually relevant tags.
Node-Based System: Leverages the power of ComfyUI's node-based system to concatenate tagging nodes, combining description captioning and keyword tagging for optimal results.
#### Versatile Integration: 
Can be combined with other nodes, such as text encoding, to achieve excellent results for automatic image processing.
#### Enhanced Image Training: 
Provides the best results for image training captioning by using advanced tagging and description methods.


## Installation:

Clone this repository to 'ComfyUI/custom_nodes` folder.

Install the dependencies in requirements.txt, transformers version 4.38.0 minimum is required:

`pip install -r requirements.txt`

or if you use portable (run this in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Miaoshouai-Tagger\requirements.txt`

## transformers 5.x compatibility

The node runs on both transformers 4.x and 5.x. Note that `requirements.txt` still pins
`transformers==4.49.0`; installing it will downgrade transformers for *every* custom node in
that ComfyUI environment, so if something else needs 5.x, skip the pin — the code does not
need it.

The PromptGen checkpoints are transformers-4.x artifacts, and v5 changed five things that
affect them. Only the first one raises an error; the other three marked below are silent,
which is why they are documented here rather than left to be rediscovered.

| Symptom | Cause |
| --- | --- |
| `AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'` on load | v5's `PretrainedConfig` is a dataclass that pops every generation parameter out of the config. |
| `AttributeError: 'list' object has no attribute 'keys'` on load | `_tied_weights_keys` must be a `{target: source}` dict in v5, not a list. |
| **Silent** — captions are word salad (`motivesadeshadesh Cypadesh Cyp...`) | v5 runs `initialize_weights()` *after* the checkpoint is loaded and skips only modules flagged `_is_hf_initialized`. Florence-2's `_init_weights` writes in place 4.x-style and sets no flags, so every language-model weight was overwritten with a fresh random init. No error, no missing key, no warning. |
| **Silent** — the caption loops one phrase for the whole token budget (`multiple boys, multiple boys, ...`) | Generation happens in the sub-model, whose config comes from `config.text_config` — which under v5 no longer carries `no_repeat_ngram_size`, `forced_bos_token_id`, `early_stopping`. |
| **Silent** — captions are plausible but wrong, describing an image the model never really saw | The processor forwards `None` for preprocessing options it was not given. Under 4.x `None` meant "use the value from `preprocessor_config.json`"; under v5 it means "off", which skipped the 768×768 resize and the normalization (measured: `(1,3,256,256)` and mean `0.0` instead of `(1,3,768,768)` and mean `-1.986`). |

One further v5 detail worth knowing if you modify the model files: `lm_head` is deliberately
**not** tied to `shared`. These checkpoints ship an `lm_head` that differs from `shared` in
all 51289 rows — it is a separately trained head, not a stale copy — and tying it makes
`<GENERATE_TAGS>` emit prose instead of tags.

## Workflows

Use as single image captioning
![miaoshouai_tagger_single_node_workflow.png](examples/miaoshouai_tagger_single_node_workflow.png)
Combine simple caption with tag caption and save to output files
![image](examples/miaoshouai_tagger_combined_workflow.png)

(Save image and grag to ComfyUI to try)

## Huggingface model
Model should be automatically downloaded the first time when you use the node. In any case that didn't happen, you can manually download it.
[MiaoshouAI/Florence-2-base-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5)
The downloaded model will be placed under`ComfyUI/LLM` folder
If you want to use a new version of PromptGen, you can simply delete the model folder and relaunch the ComfyUI workflow. It will auto download the model for you.

## Windows Tagger Program
For anyone who wants to use PromptGen model outside comfyui to batch tag their images, you can use this tag tool created by TTPlant.
His program uses my model and works in a Windows enviroment. Access to the [download link](https://github.com/TTPlanetPig/Florence_2_tagger).

