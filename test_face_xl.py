import os
import sys

sys.path.append(os.getcwd())
import glob
import argparse
import numpy as np
from PIL import Image

import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    EulerDiscreteScheduler,
)
from models.controlnet_xl import ControlNetModel
from diffusers.utils.import_utils import is_xformers_available
from pipelines.pipeline_xl import StableDiffusionXLControlNetPipeline

from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from torchvision import transforms
from transformers import PretrainedConfig, AutoTokenizer

from tqdm import tqdm
import pyiqa
import lpips
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


def gray_world_color_fix(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return image
    strength = max(0.0, min(1.0, float(strength)))
    image_np = np.asarray(image).astype(np.float32)
    mean_rgb = image_np.reshape(-1, 3).mean(axis=0)
    gray = float(mean_rgb.mean())
    scale = gray / (mean_rgb + 1e-6)
    corrected = np.clip(image_np * scale, 0, 255)
    blended = image_np * (1.0 - strength) + corrected * strength
    return Image.fromarray(np.round(blended).astype(np.uint8))


def apply_color_fix(image: Image.Image, source_image: Image.Image, args):
    if args.align_method == "wavelet":
        image, image_tensor = wavelet_color_fix(image, source_image, return_tensor=True)
    elif args.align_method == "adain":
        image, image_tensor = adain_color_fix(image, source_image, return_tensor=True)
    elif args.align_method == "nofix":
        image_tensor = tensor_transforms(image).unsqueeze(0)
    else:
        raise ValueError(f"Unknown align method: {args.align_method}")

    if args.minor_color_fix_strength > 0:
        image = gray_world_color_fix(image, args.minor_color_fix_strength)
        image_tensor = tensor_transforms(image).unsqueeze(0)

    return image, image_tensor


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):

    from models.unet_2d_condition_xl import UNet2DConditionModel

    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    print("================== To load unet ===============")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    print("================== To load Controlnet ===============")
    controlnet = ControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="controlnet"
    )

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    validation_pipeline = StableDiffusionXLControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )

    return validation_pipeline


def resolve_validation_images(validate_image_path):
    try:
        if os.path.isdir(validate_image_path):
            file_list = os.listdir(validate_image_path)
            lq_image_path = None
            gt_image_path = None
            for image_path in file_list:
                if "lq" in image_path:
                    lq_image_path = os.path.join(validate_image_path, image_path)
                else:
                    gt_image_path = os.path.join(validate_image_path, image_path)
            image_names = sorted(glob.glob(f"{lq_image_path}/*.*"))
            gt_image_names = sorted(glob.glob(f"{gt_image_path}/*.*"))
            print("============ Test w GT ============")
            return image_names, gt_image_names
    except Exception:
        pass

    print("============ Test wo GT ============")
    if os.path.isdir(validate_image_path):
        image_names = sorted(glob.glob(f"{validate_image_path}/*.*"))
    else:
        image_names = [validate_image_path]
    return image_names, None


def batch_test(
    args,
    enable_xformers_memory_efficient_attention=True,
):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        accelerator.init_trackers("Authface")

    pipeline = load_pipeline(
        args, accelerator, enable_xformers_memory_efficient_attention
    )

    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        image_names, gt_image_names = resolve_validation_images(
            args.validate_image_path
        )

        for sample_idx in range(args.sample_times):
            os.makedirs(
                f"{args.output_dir}/sample{str(sample_idx).zfill(2)}/",
                exist_ok=True,
            )

        batch_validation_image = []
        batch_validation_prompt = []
        batch_negative_prompt = []
        batch_image_name = []

        def run_pipeline_batch(
            batch_images, batch_prompts, batch_neg_prompts, batch_names, height, width
        ):
            for sample_idx in range(args.sample_times):
                image_dict = pipeline(
                    prompt=batch_prompts,
                    image=batch_images,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    height=height,
                    width=width,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=batch_neg_prompts,
                    controlnet_conditioning_scale=args.conditioning_scale,
                    control_guidance_start=0.0,
                    control_guidance_end=1,
                )

                for i in range(len(batch_images)):
                    image = image_dict.images[i]
                    image, _ = apply_color_fix(image, batch_images[i], args)
                    image = image.resize(
                        (args.process_size, args.process_size),
                        Image.LANCZOS,
                    )
                    name, _ = os.path.splitext(os.path.basename(batch_names[i]))
                    image.save(
                        f"{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png"
                    )

        for image_idx, image_name in tqdm(enumerate(image_names)):
            print(f"================== process {image_idx} imgs... ===================")
            validation_image = Image.open(image_name).convert("RGB")

            validation_prompt = args.added_prompt
            negative_prompt = args.negative_prompt

            print(f"{validation_prompt}")

            rscale = args.upscale

            validation_image = validation_image.resize(
                (validation_image.size[0] * rscale, validation_image.size[1] * rscale)
            )
            validation_image = validation_image.resize(
                (validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8)
            )
            width, height = validation_image.size

            print(f"input size: {height}x{width}")
            if gt_image_names is not None:
                gt_image = Image.open(gt_image_names[image_idx]).convert("RGB")

            batch_validation_image.append(validation_image)
            batch_validation_prompt.append(validation_prompt)
            batch_negative_prompt.append(negative_prompt)
            batch_image_name.append(image_name)
            if len(batch_validation_image) == args.batch_size:
                run_pipeline_batch(
                    batch_validation_image,
                    batch_validation_prompt,
                    batch_negative_prompt,
                    batch_image_name,
                    height,
                    width,
                )
                batch_validation_image = []
                batch_validation_prompt = []
                batch_negative_prompt = []
                batch_image_name = []
        if batch_validation_image:
            run_pipeline_batch(
                batch_validation_image,
                batch_validation_prompt,
                batch_negative_prompt,
                batch_image_name,
                height,
                width,
            )


def cal_metric_only(args):
    image_names, gt_image_names = resolve_validation_images(args.validate_image_path)

    lpips_fn = lpips.LPIPS(net="alex").to(device="cuda")
    MUSIQ_fn = pyiqa.create_metric("musiq")
    ManIQA_fn = pyiqa.create_metric("maniqa")
    TOPIQA_fn = pyiqa.create_metric("topiq_nr-face")
    for repeat_time in range(args.sample_times):
        file_list = sorted(
            os.listdir(
                os.path.join(args.output_dir, f"sample{str(repeat_time).zfill(2)}")
            )
        )
        image_logs = []
        psnr_list = []
        ssim_list = []
        lpips_list = []
        MUSIQ_list = []
        ManIQA_list = []
        TOPIQA_list = []
        for idx, file in tqdm(enumerate(file_list)):
            file_path = os.path.join(
                args.output_dir, f"sample{str(repeat_time).zfill(2)}", file
            )
            validation_image = Image.open(file_path).convert("RGB")
            if gt_image_names is not None:
                gt_image = Image.open(gt_image_names[idx]).convert("RGB")
                gt_image_tensor = tensor_transforms(gt_image).to(dtype=torch.float32)

            image_tensor = tensor_transforms(validation_image).to(dtype=torch.float32)

            if gt_image_names is not None:
                image_np = np.clip(image_tensor.permute(1, 2, 0).numpy(), 0, 1)
                gt_image_np = np.clip(gt_image_tensor.permute(1, 2, 0).numpy(), 0, 1)
                try:
                    psnr_list.append(PSNR(image_np, gt_image_np))
                except Exception as exc:
                    logger.warning("PSNR calculation failed: %s", exc)
                ssim_list.append(
                    SSIM(
                        image_np,
                        gt_image_np,
                        multichannel=True,
                        channel_axis=2,
                        data_range=1,
                    )
                )
                lpips_list.append(
                    lpips_fn(
                        image_tensor.to(device="cuda"),
                        gt_image_tensor.to(device="cuda"),
                    ).item()
                )
            else:
                psnr_list.append(0)
                ssim_list.append(0)
                lpips_list.append(0)
            MUSIQ_list.append(MUSIQ_fn(validation_image).item())
            ManIQA_list.append(ManIQA_fn(validation_image).item())
            try:
                TOPIQA_list.append(
                    TOPIQA_fn(validation_image).item()
                )  # sometimes face can not be detected, so we skip this
            except:
                continue
        image_logs.append(
            {
                "PSNR": float(np.average(np.array(psnr_list))),
                "SSIM": float(np.average(np.array(ssim_list))),
                "LPIPS": float(np.average(np.array(lpips_list))),
                "MUSIQ": np.average(np.array(MUSIQ_list)),
                "ManIQA": np.average(np.array(ManIQA_list)),
                "TOPIQA": np.average(np.array(TOPIQA_list)),
            }
        )
        file_name = args.validate_image_path.split("/")[-1]
        import json

        file_name = f"metric_{file_name}_{repeat_time}.json"
        with open(os.path.join(args.output_dir, file_name), "w") as f:
            json.dump(image_logs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--added_prompt",
        type=str,
        default="hyper-realistic portrait, face close-up, super detailed, Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera,"
        "hyper detailed photo - realistic maximum detail, 32k, ultra HD,"
        "extreme meticulous detailing, skin pore detailing,"
        "hyper sharpness, perfect without deformations",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="unreal face, asymmetrical eyes, oil painting, illustration, drawing, art, sketch,"
        "cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, "
        "worst quality, low quality, frames, watermark, signature, jpeg artifacts, "
        "deformed, lowres, over-smooth, malformed/disproportionate hands/fingers, ugly, easy negative",
    )
    parser.add_argument("--validate_image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--guidance_scale", type=float, default=6.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=320)
    parser.add_argument("--latent_tiled_overlap", type=int, default=4)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument(
        "--align_method",
        type=str,
        choices=["wavelet", "adain", "nofix"],
        default="wavelet",
    )
    parser.add_argument("--minor_color_fix_strength", type=float, default=0.0)
    parser.add_argument("--start_steps", type=int, default=999)
    parser.add_argument(
        "--start_point", type=str, choices=["lr", "noise"], default="lr"
    )
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("-y", "--yaml_path", type=str)
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="num of batch size",
    )

    args = parser.parse_args()
    batch_test(args)
    cal_metric_only(args)
