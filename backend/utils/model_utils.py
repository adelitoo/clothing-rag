import torch
from transformers import CLIPProcessor, CLIPModel  
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)

from config import DEVICE, IMAGE_TEXT_MODEL, IMAGE_CAPTION_MODEL


def load_clip_model_and_processor(model_name=IMAGE_TEXT_MODEL):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = torch.device(DEVICE)
    model.to(device).eval()
    return model, processor


def load_captioning_model_and_processor(device=DEVICE):
    processor = BlipProcessor.from_pretrained(IMAGE_CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(
        IMAGE_CAPTION_MODEL,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    model.eval()
    return processor, model
