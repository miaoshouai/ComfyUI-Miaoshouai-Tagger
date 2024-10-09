# ComfyUI 的 MiaoshouAI Tagger

[English](README.md) / [中文](README_CN.md)

MiaoshouAI Tagger 是一个基于微软 Florence-2 模型的高级图像标注工具，经过精细调优。该工具为您的项目提供高精度和上下文相关的图像标注。

## 版本更新
2024/09/28 v1.31 修复部分用户碰到的[模型配置文件缺失](https://github.com/miaoshouai/ComfyUI-Miaoshouai-Tagger/issues/15)问题, 更新节点之后，删除models\LLM文件夹下面的模型，重新运行工作流模型会自动下载。或者你也可以手动从 [度盘](https://pan.baidu.com/s/1h8kLNmukfcUitM7mKRE89w?pwd=4xwc) 文件夹下载模型.</br>
2024/09/07 v1.2 更新支持 [Florence-2-large-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5), 为Tagger节点增加了一个随机widget，如果选择Always，Tagger将在每一次运行生成一套新的提示词。<br>
2024/09/05 v1.1 更新支持 [Florence-2-base-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5)，新增了两种提示模式；新增了一个用于 flux clip 文本编码器的节点，以便更轻松地支持 flux 模型片段。

## 为什么需要另一个标注工具？[
尽管目前有许多](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5)标注工具如 WD14 表现相当不错，但它们在实用中尝尝各有各的问题。MiaoshouAI/Florence-2-base-PromptGen 是基于微软最新的 Florence2 模型，并使用精心挑选的 Civitai 图像和标签进行训练，专门为生成和标注提示词而训练。因此，其标注结果更加符合我们通常用于生成图像的提示，提高了准确性和相关性。

## 为什么基于ComfyUI？
ComfyUI 已成为 Stable Diffusion 工作者中最受欢迎的基于节点的工具之一。它提供了各种节点和模型，例如 LLava 和 Ollama Vision 节点，用于生成图像打标并将其传递给文本编码器。然而，这些视觉模型并不是专门为提示和图像标注训练的。使用 MiaoshouAI Tagger，您可以看到明显的结果改进。

## 主要功能
#### 高精度：
基于精选的 Civitai 图像和清洗标签数据集进行微调，生成高度精确和上下文相关的标签。

#### 基于节点的系统：
利用 ComfyUI 的节点系统的强大功能，将标注节点连接起来，结合描述性打标和关键词标注以获得最佳效果。

#### 多功能集成：
可以与其他节点（如Prompt Text Encoder）结合，达到出色的自动图像处理效果。

#### 增强的图像训练：
通过使用先进的标注和描述方法，为图像训练打标提供最佳结果。

## 安装
将此存储库克隆到 `ComfyUI/custom_nodes` 文件夹中。

安装 `requirements.txt` 中的依赖项，至少需要 transformers 版本 4.38.0：

bash
复制代码
`pip install -r requirements.txt`
或者，如果您使用便携版本（在 ComfyUI_windows_portable 文件夹中运行此命令）：

bash
复制代码
`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Miaoshouai-Tagger\requirements.txt`

# 工作流
单图像标注使用：
![miaoshouai_tagger_single_node_workflow.png](examples/miaoshouai_tagger_single_node_workflow.png)

结合简单的描述性打标和标签打标并保存到输出文件：
![image](examples/miaoshouai_tagger_combined_workflow.png)

（保存图像并拖动到 ComfyUI 中使用）

## Huggingface 模型
首次使用节点时，模型应自动下载。如果没有发生这种情况，您可以手动下载。
[MiaoshouAI/Florence-2-base-PromptGen-v1.5](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5)
下载的模型将放置在 `ComfyUI/LLM` 文件夹下。
如果你想要更新PromptGen的最新版本，你可以在此文件中删除你原有的模型，然后重新运行工作流，新模型会自动下载。

## Windows打标工具
对于任何希望在 ComfyUI 之外使用 PromptGen 模型对其图像进行批量标记的人，可以使用汤团猪创建的这个标记工具。
他的程序使用了我的模型，并支持在Windows环境中运行。获取[下载链接](https://github.com/TTPlanetPig/Florence_2_tagger)。