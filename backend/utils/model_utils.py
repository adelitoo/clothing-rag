import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from functools import cache
from config import DEVICE, IMAGE_TEXT_MODEL, IMAGE_CAPTION_MODEL


@cache
def load_clip_model_and_processor(model_name: str = IMAGE_TEXT_MODEL):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.to(DEVICE)
    model.eval()
    return model, processor


@cache
def load_captioning_model_and_processor(device: torch.device = DEVICE):
    processor = BlipProcessor.from_pretrained(IMAGE_CAPTION_MODEL)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = BlipForConditionalGeneration.from_pretrained(
        IMAGE_CAPTION_MODEL, torch_dtype=dtype
    ).to(device)

    model.eval()
    return processor, model
