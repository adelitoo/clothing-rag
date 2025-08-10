import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from functools import cache
from .config import settings

@cache
def load_clip_model_and_processor(model_name: str = settings.IMAGE_TEXT_MODEL):
    model = CLIPModel.from_pretrained(model_name).to(settings.DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

@cache
def load_captioning_model_and_processor(model_name: str = settings.IMAGE_CAPTION_MODEL):
    processor = BlipProcessor.from_pretrained(model_name)
    dtype = torch.float16 if settings.DEVICE.type == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(settings.DEVICE).eval()
    return processor, model