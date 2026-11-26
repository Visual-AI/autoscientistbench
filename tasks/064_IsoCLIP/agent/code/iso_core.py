"""Unmodified OpenAI image encoder and IsoCLIP projector functions from the official source."""

import torch

def apply_iso(W_text, W_image, iso_ktop=0, iso_kbottom=0 ):
    
    # inter-modal operator
    Psi = W_image.T @ W_text
        
    U, S, V = torch.linalg.svd(Psi, full_matrices=False)
    V = V.T

    print("Manual filtering: k_top = {}, k_bottom = {}".format(iso_ktop, iso_kbottom))
    r = S.shape[0]

    start = iso_ktop
    end = r - iso_kbottom
    # Sanity check
    if iso_ktop + iso_kbottom >= r:
        raise ValueError(f"Cannot remove {iso_ktop} top and {iso_kbottom} bottom components from {r}")

    U_k = U[:, start:end]
    V_k = V[:, start:end]
    
    W_text_iso = W_text @ V_k @ V_k.T
    W_image_iso = W_image @ U_k @ U_k.T 
 
    # Perform the transpose to match the original shape of the features 
    W_text_iso = W_text_iso.T
    W_image_iso = W_image_iso.T
 
    return W_text_iso, W_image_iso 

@torch.no_grad()
def encode_image_noproj_openai(clip_model, images):
    x = clip_model.visual.conv1(images)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + clip_model.visual.positional_embedding.to(x.dtype)
    x = clip_model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.visual.ln_post(x[:, 0, :]) # removed projection
    return x 
