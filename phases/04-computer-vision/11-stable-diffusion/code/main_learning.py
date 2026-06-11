"""
Stable Diffusion usage examples. Requires `diffusers`, `transformers`, and a GPU
for any real inference. Running this on CPU without the model is a no-op summary.
"""

import os
import shutil
import subprocess
import sys


_PIPELINE_CACHE = {}


def import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def has_diffusers():
    try:
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def describe_pipeline():
    print("[stable diffusion pipeline]")
    print("  text_encoder:   CLIP-L (SD 1.5) / CLIP-L+G (SDXL) / T5-XXL (SD3, FLUX)")
    print("  unet_params:    860M (SD 1.5) / 2.6B (SDXL) / 12B (FLUX)")
    print("  vae_latent:     4 x 64 x 64 for 512x512 input, 4 x 128 x 128 for 1024x1024")
    print("  vae_scale:      0.18215 (SD 1.5/2), 0.13025 (SDXL)")
    print("  default_cfg:    7.5") # 这个gradient scale越高，生成的图像越贴合文本提示，但过高可能导致过度锐化和失真。常见范围是5-10，具体值取决于提示的复杂性和所需的创造力水平。


def cfg_sweep_demo():
    values = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
    print("\n[cfg sweep values to try on a real pipeline]")
    for w in values:
        effect = (
            "unconditional" if w <= 1.0
            else "creative but weak prompt adherence" if w < 5.0
            else "standard" if w <= 8.0
            else "strong adherence, possible oversaturation" if w <= 12.0
            else "heavy artefacts"
        )
        print(f"  w={w:5.1f}  expected: {effect}")


def is_sdxl_model(model_id):
    name = model_id.lower()
    return "sdxl" in name or "stable-diffusion-xl" in name or "/xl-" in name


def open_image_file(path):
    path = os.path.abspath(os.path.expanduser(path))
    if sys.platform == "darwin":
        command = ["open", path]
    elif sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606
        print(f"  opened: {path}")
        return True
    elif shutil.which("xdg-open"):
        command = ["xdg-open", path]
    elif shutil.which("wslview"):
        command = ["wslview", path]
    else:
        print(f"  no desktop opener found; open manually: {path}")
        return False

    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  opened: {path}")
    return True


def set_scheduler(pipe, scheduler_name):
    import diffusers

    dpm_cls = getattr(diffusers, "DPMSolverMultistepScheduler", None)
    euler_cls = getattr(diffusers, "EulerAncestralDiscreteScheduler", None)
    schedulers = {
        "DPMSolverMultistepScheduler": dpm_cls,
        "EulerAncestralDiscreteScheduler": euler_cls,
    }
    if scheduler_name in schedulers and schedulers[scheduler_name] is not None:
        pipe.scheduler = schedulers[scheduler_name].from_config(pipe.scheduler.config)
    else:
        print(f"  unknown scheduler={scheduler_name!r}; using model default scheduler")
    return pipe.scheduler.__class__.__name__


def build_pipeline(model_id, torch, scheduler_name="DPMSolverMultistepScheduler", device="cuda"):
    import diffusers

    cache_key = (model_id, device)
    if cache_key in _PIPELINE_CACHE:
        pipe, pipeline_name = _PIPELINE_CACHE[cache_key]
        resolved_scheduler = set_scheduler(pipe, scheduler_name)
        print(f"  model: {model_id}")
        print(f"  pipeline: {pipeline_name} (cached)")
        print(f"  scheduler: {resolved_scheduler}")
        return pipe

    auto_cls = getattr(diffusers, "AutoPipelineForText2Image", None)
    sd_cls = getattr(diffusers, "StableDiffusionPipeline", None)
    sdxl_cls = getattr(diffusers, "StableDiffusionXLPipeline", None)

    if auto_cls is not None:
        pipeline_cls = auto_cls
    elif is_sdxl_model(model_id):
        if sdxl_cls is None:
            print("  this diffusers install does not provide StableDiffusionXLPipeline.")
            print("  use an SD 1.x model_id here, or run this example in an environment with SDXL support.")
            return None
        pipeline_cls = sdxl_cls
    else:
        if sd_cls is None:
            print("  this diffusers install does not provide StableDiffusionPipeline.")
            return None
        pipeline_cls = sd_cls

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if is_sdxl_model(model_id) and hasattr(pipe, "vae"):
        pipe.vae.to(dtype=torch.float32)
    resolved_scheduler = set_scheduler(pipe, scheduler_name)
    pipeline_name = pipe.__class__.__name__
    _PIPELINE_CACHE[cache_key] = (pipe, pipeline_name)

    print(f"  model: {model_id}")
    print(f"  pipeline: {pipeline_name}")
    print(f"  scheduler: {resolved_scheduler}")
    return pipe


def text_to_image_stub(
    prompt,
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    scheduler="DPMSolverMultistepScheduler",
    guidance_scale=7.5,
    seed=42,
    save_name="sd_demo.png",
    height=None,
    width=None,
    show=True,
):
    print(f"\n[text_to_image] prompt={prompt!r} seed={seed}")
    torch = import_torch()
    if torch is None:
        print("  torch not installed. Install torch in the runtime environment to run real inference.")
        return None
    if not has_diffusers():
        print("  diffusers not installed. `pip install diffusers transformers accelerate` to run.")
        return None
    if not torch.cuda.is_available():
        print("  CUDA not available; running SD on CPU is extremely slow. Skipping real call.")
        return None
    try:
        pipe = build_pipeline(model_id, torch, scheduler_name=scheduler, device="cuda")
    except Exception as exc:
        print(f"  could not load pipeline: {type(exc).__name__}: {exc}")
        return None
    if pipe is None:
        return None

    gen = torch.Generator("cuda").manual_seed(seed)
    if height is None or width is None:
        size = 768 if is_sdxl_model(model_id) else 512
        height = height or size
        width = width or size

    try:
        out = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=25,
            generator=gen,
            height=height,
            width=width,
        ).images[0]
    except RuntimeError as exc:
        print(f"  inference failed: {type(exc).__name__}: {exc}")
        print("  try a smaller height/width, fewer steps, or an SD 1.5 model if VRAM is limited.")
        return None

    path = os.path.expanduser(f"~/{save_name}")
    out.save(path)
    print(f"  saved: {path}")
    if show:
        open_image_file(path)
    return path


def lora_training_sketch():
    print("\n[lora training pseudocode]")
    pseudo = """
            for step, batch in enumerate(dataloader):
                images, prompts = batch
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                t = torch.randint(0, num_train_timesteps, (batch_size,))
                noise = torch.randn_like(latents)
                noisy_latents = scheduler.add_noise(latents, noise, t)
                text_emb = text_encoder(tokenizer(prompts))
                pred_noise = unet(noisy_latents, t, text_emb)       # LoRA weights injected
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()
            """
    print(pseudo)


def main():
    describe_pipeline()
    cfg_sweep_demo()
    text_to_image_stub("a girl standing on the tokyo street, studio ghibli style", model_id="stabilityai/stable-diffusion-xl-base-1.0", scheduler="EulerAncestralDiscreteScheduler", guidance_scale=7.5, save_name="sdxl_demo7.png")
    lora_training_sketch()


if __name__ == "__main__":
    main()
