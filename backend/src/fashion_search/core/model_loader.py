import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from functools import cache
from .config import settings

@cache
def load_clip_model_and_processor(model_name: str = settings.IMAGE_TEXT_MODEL):
    """Loads and caches the CLIP model and processor."""
    print(f"⏳ Loading CLIP model: {model_name}...")
    model = CLIPModel.from_pretrained(model_name).to(settings.DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"✅ CLIP model loaded on device: {settings.DEVICE}")
    return model, processor

@cache
def load_captioning_model_and_processor(model_name: str = settings.IMAGE_CAPTION_MODEL):
    """Loads and caches the BLIP captioning model and processor."""
    print(f"⏳ Loading Captioning model: {model_name}...")
    processor = BlipProcessor.from_pretrained(model_name)
    dtype = torch.float16 if settings.DEVICE.type == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(settings.DEVICE).eval()
    print(f"✅ Captioning model loaded on device: {settings.DEVICE}")
    return processor, model