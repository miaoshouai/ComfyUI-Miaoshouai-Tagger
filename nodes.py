import os
from PIL import Image
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import torchvision.transforms.functional as F
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoProcessor

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


class Tagger:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Path to your image folder"
                }),
                "caption_method": (['tags', 'simple', 'detailed'], {
                    "default": "tags"
                }),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64})
            },
            "optional": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"forceInput": True}),
                "captions": ("STRING", {"forceInput": True}),
                "prefix_caption": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "suffix_caption": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "INT", )
    RETURN_NAMES = ("images", "filenames", "captions", "folder_path", "batch_size", )
    OUTPUT_IS_LIST = (True, True, True, False, False, )
    FUNCTION = "start_tag"
    #OUTPUT_NODE = True
    CATEGORY = "MiaoshouAI Tagger"

    def tag_image(self, image, caption_method, model, processor, device, dtype, max_new_tokens, num_beams):

        if caption_method == 'tags':
            prompt = "<GENERATE_PROMPT>"
        elif caption_method == 'simple':
            prompt = "<DETAILED_CAPTION>"
        else:
            prompt = "<MORE_DETAILED_CAPTION>"

        inputs = processor(text=prompt, images=image, return_tensors="pt", do_rescale=False).to(dtype).to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            early_stopping=False,
            do_sample=False,
            num_beams=num_beams,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height))

        return parsed_answer[prompt]

    def start_tag(self, folder_path, caption_method, max_new_tokens, num_beams, images=None, filenames=None, captions=None, prefix_caption="", suffix_caption=""):
        file_names = []
        tag_contents = []
        pil_images = []
        tensor_images = []
        attention = 'sdpa'
        precision = 'fp16'
        batch_size = 0

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        print(f"using {attention} for attention")

        # Download model if it does not exist
        hg_model = 'MiaoshouAI/Florence-2-base-PromptGen'
        model_name = hg_model.rsplit('/', 1)[-1]
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)
        if not os.path.exists(model_path):
            print(f"Downloading Lumina model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=hg_model,
                              local_dir=model_path,
                              local_dir_use_symlinks=False)

        with patch("transformers.dynamic_module_utils.get_imports",
                   fixed_get_imports):  # workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device,
                                                         torch_dtype=dtype, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if images is None:
            for filename in os.listdir(folder_path):
                image_types = ['png', 'jpg', 'jpeg']
                if filename.split(".")[-1] in image_types:
                    img_path = os.path.join(folder_path, filename)
                    cap_filename = '.'.join(filename.split('.')[:-1]) + '.txt'
                    image = Image.open(img_path)
                    pil_images.append(image)

                    tensor_image = F.to_tensor(image)
                    tensor_image = tensor_image[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    tensor_images.append(tensor_image)

                    file_names.append(cap_filename)
        else:
            if len(images) == 1:
                images = [images]

            to_pil = transforms.ToPILImage()
            max_digits = max(3, len(str(len(images))))

            for i, img in enumerate(images, start=1):
                if img.ndim == 4:
                    # Convert (N, H, W, C) to (N, C, H, W)
                    img = img.permute(0, 3, 1, 2).squeeze(0)

                if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                    pil_img = to_pil(img.cpu())
                    pil_images.append(pil_img)

                    tensor_img = F.to_tensor(pil_img)
                    tensor_img = tensor_img[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    tensor_images.append(tensor_img)

                if filenames is None:
                    cap_filename = f"{i:0{max_digits}d}.txt"
                    file_names.append(cap_filename)
                else:
                    file_names.append(filenames)

        pbar = ProgressBar(len(pil_images))

        for i, image in enumerate(pil_images):
            tags = self.tag_image(image, caption_method, model, processor, device, dtype, max_new_tokens, num_beams)
            tags = prefix_caption + tags + suffix_caption
            # when two tagger nodes and their captions are connected
            if captions is not None:
                tags = captions + tags

            print(i, caption_method, tags)
            tag_contents.append(tags)

            pbar.update(1)

        # Print tensor image details for debugging
        for i, img in enumerate(tensor_images):
            print(f"Tensor image {i + 1} shape: {img.shape}")
            if img.shape[0] == 3:
                print("Color image")
            elif img.shape[0] == 1:
                print("Grayscale image")
            else:
                print("Unexpected number of channels")

        batch_size = len(tensor_images)

        return (tensor_images, file_names, tag_contents, folder_path, batch_size,)


class SaveTags:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filenames": ("STRING", {"forceInput": True}),
                "captions": ("STRING", {"forceInput": True}),
                "save_folder": ("STRING", {"default": "Your save directory"}),
                "filename_prefix": ("STRING", {"default": ""}),
                "mode": (['overwrite', 'append'], {
                    "default": "overwrite"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('captions',)
    OUTPUT_IS_LIST = (False, )
    FUNCTION = "save_tag"
    OUTPUT_NODE = True
    CATEGORY = "MiaoshouAI Tagger"

    def save_tag(self, filenames, captions, save_folder, filename_prefix, mode):

        wmode = 'w' if mode == 'overwrite' else 'a'
        with open(os.path.join(save_folder, filename_prefix + filenames), wmode) as f:
            f.write(captions)

        print("Captions Saved")

        return (captions,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Miaoshouai_Tagger": Tagger,
    "Miaoshouai_SaveTags": SaveTags
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Miaoshouai_Tagger": "🐾MiaoshouAI Tagger",
    "Miaoshouai_SaveTags": "🐾MiaoshouAI Save Tags"
}
