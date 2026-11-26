"""Expose the fixed CLIP encoder separately from editable retrieval postprocessing."""
import clip
from iso_core import encode_image_noproj_openai

def load_encoder(model_dir,checkpoint):
    model,preprocess=clip.load(str(checkpoint),device='cpu')
    return model.eval().requires_grad_(False),preprocess

def encode(model,images):
    return encode_image_noproj_openai(model,images)
