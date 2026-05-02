import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from rembg import remove
from diffusers import AutoPipelineForInpainting
from ollama_refine import refine_prompt

# -------------------------
# Paths
# -------------------------
ROOT = Path(".")
INPUT_DIR  = ROOT / "0_input"
MASK_DIR   = ROOT / "1_mask_debug"
OUT_DIR    = ROOT / "2_diffusion_output"
LOG_DIR    = ROOT / "logs"

MASK_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Stable Diffusion inpainting model (lighter than SDXL)
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return arr
    # RGB -> BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    if cv_img.ndim == 2:
        return Image.fromarray(cv_img)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def segment_foreground_rgba(pil_img: Image.Image) -> Image.Image:
    """
    Uses rembg to remove background. Returns RGBA where alpha ~ foreground.
    """
    rgba = remove(pil_img)  # output is usually RGBA
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    return rgba

def build_outpaint_canvas_and_mask(
    pil_img: Image.Image,
    pad_px: int = 128,
    ring_px: int = 35
):
    """
    Creates:
    - expanded canvas (outpaint area)
    - mask for inpainting (white = fill)
    Also creates a small ring around subject boundary for better blending.
    """
    w, h = pil_img.size

    # 1) Segment foreground to get alpha mask
    rgba = segment_foreground_rgba(pil_img)
    rgba_np = np.array(rgba)  # (h,w,4)
    alpha = rgba_np[:, :, 3]  # 0..255

    # Foreground mask (head/person)
    fg = (alpha > 10).astype(np.uint8) * 255

    # Smooth + fill small holes
    fg = cv2.medianBlur(fg, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    # 2) Expand canvas
    new_w = w + 2 * pad_px
    new_h = h + 2 * pad_px

    # Put original image centered on expanded canvas
    canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))  # neutral base
    canvas.paste(pil_img, (pad_px, pad_px))

    # 3) Create inpaint mask
    # Mask starts as: padded region = white
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    mask[:pad_px, :] = 255
    mask[-pad_px:, :] = 255
    mask[:, :pad_px] = 255
    mask[:, -pad_px:] = 255

    # 4) Add a "ring" around the foreground boundary (hair blending)
    # Place fg into expanded space
    fg_big = np.zeros((new_h, new_w), dtype=np.uint8)
    fg_big[pad_px:pad_px + h, pad_px:pad_px + w] = fg

    # Ring = dilate(fg) - erode(fg)
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px*2+1, ring_px*2+1))
    dil = cv2.dilate(fg_big, dil_k, iterations=1)

    er_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, ring_px//2)*2+1, max(3, ring_px//2)*2+1))
    ero = cv2.erode(fg_big, er_k, iterations=1)

    ring = cv2.subtract(dil, ero)  # ring area around subject
    # We want diffusion to "paint" the ring area too (helps hair/back head continuity)
    mask = cv2.bitwise_or(mask, ring)

    # Slight blur on mask edge for smoother transitions
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    return canvas, Image.fromarray(mask).convert("L"), fg_big, ring

def run_inpainting(
    image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    seed: int = 42,
    steps: int = 30,
    guidance: float = 7.0,
    strength: float = 0.92
) -> Image.Image:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)

    g = torch.Generator(device=device).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=g
    ).images[0]

    return out

def main():
    print("\n=== Auto-mask Outpaint + Diffusion Inpainting (Beginner) ===\n")

    img_name = input("Enter input image filename inside 0_input (default: input.jpg): ").strip()
    if not img_name:
        img_name = "input.jpg"

    in_path = INPUT_DIR / img_name
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    user_request = input(
        "\nDescribe what you want (example: 'generate back of head with natural hair, extend hair length, realistic'): \n> "
    ).strip()
    if not user_request:
        user_request = "Generate missing back of head and natural hair texture, photorealistic, same identity."

    pad_px = input("\nOutpaint padding in pixels (recommended 96 to 160, default 128): ").strip()
    pad_px = int(pad_px) if pad_px else 128

    ring_px = input("\nBlend ring size around head boundary (default 35): ").strip()
    ring_px = int(ring_px) if ring_px else 35

    seed = input("\nSeed (default 42): ").strip()
    seed = int(seed) if seed else 42

    print("\n[1/3] Loading image...")
    pil_img = Image.open(in_path).convert("RGB")

    print("[2/3] Refining prompt using Ollama (llama3:8b)...")
    refined = refine_prompt(user_request)

    # Add strong “keep identity” constraints (helps a lot)
    final_prompt = (
        refined
        + ", photorealistic portrait, consistent identity, realistic hair strands, natural lighting, "
          "high detail texture, no distortion, no cartoon, no anime, no painting"
    )

    print("\nFinal Prompt:\n", final_prompt)

    print("\n[3/3] Building auto outpaint mask + running diffusion inpainting...")
    canvas, mask_pil, fg_big, ring = build_outpaint_canvas_and_mask(pil_img, pad_px=pad_px, ring_px=ring_px)

    # Save debug masks
    (MASK_DIR / "mask.png").write_bytes(mask_pil.tobytes())  # not viewable, just placeholder

    mask_np = np.array(mask_pil)
    canvas_np = pil_to_cv2(canvas)

    cv2.imwrite(str(MASK_DIR / "mask.png"), mask_np)
    cv2.imwrite(str(MASK_DIR / "canvas.png"), canvas_np)
    cv2.imwrite(str(MASK_DIR / "fg_big.png"), fg_big)
    cv2.imwrite(str(MASK_DIR / "ring.png"), ring)

    # Overlay for easy viewing
    overlay = canvas_np.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    alpha = (mask_np.astype(np.float32) / 255.0) * 0.45
    overlay = (overlay * (1 - alpha[..., None]) + green * alpha[..., None]).astype(np.uint8)
    cv2.imwrite(str(MASK_DIR / "mask_overlay.png"), overlay)

    # Run diffusion
    result = run_inpainting(
        image=canvas,
        mask_image=mask_pil,
        prompt=final_prompt,
        seed=seed,
        steps=30,
        guidance=7.0,
        strength=0.92
    )

    out_path = OUT_DIR / "clean_outpaint.png"
    result.save(out_path)

    print("\n✅ DONE!")
    print("Debug mask outputs in:", MASK_DIR)
    print("Final completed image:", out_path)

if __name__ == "__main__":
    main()
