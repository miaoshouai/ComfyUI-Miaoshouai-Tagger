# MiaoshouAI Tagger for ComfyUI

[English](README.md) / [中文](README_CN.md)

MiaoshouAI Tagger for ComfyUI is an advanced image captioning tool based on the Microsoft Florence-2 Model Fine-tuned to perfection. This tool offers highly accurate and contextually relevant image tagging for your projects.

## Update Note
2024/09/28 v1.3 fix the configuration error rated to [this issue](https://github.com/miaoshouai/ComfyUI-Miaoshouai-Tagger/issues/15), try to delete your existing model from models\LLM folder and run again. It will automatically download the new configurations for you. Or you can download the model from the [baidu drive](https://pan.baidu.com/s/1h8kLNmukfcUitM7mKRE89w?pwd=4xwc) folder.</br>
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

## Workflows

Use as single image captioning
![miaoshouai_tagger_single_node_workflow.png](examples/miaoshouai_tagger_single_node_workflow.png)
Combine simple caption with tag caption and save to output files
![image](examples/miaoshouai_tagger_combined_workflow.png)

(Save image and grag to ComfyUI to try)

## Huggingface model
Model should be automatically downloaded the first time when you use the node. In any case that didn't happen, you can manually download it.
[MiaoshouAI/Florence-2-base-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5)
The downloaded model will be placed under`ComfyUI/models/LLM` folder
If you want to use a new version of PromptGen, you can simply delete the model folder and relaunch the ComfyUI workflow. It will auto download the model for you.

## Windows Tagger Program
For anyone who wants to use PromptGen model outside comfyui to batch tag their images, you can use this tag tool created by TTPlant.
His program uses my model and works in a Windows enviroment. Access to the [download link](https://github.com/TTPlanetPig/Florence_2_tagger).

