import argparse, os, sys, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import json

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def generate_images_from_prompts(opt, prompts_with_filenames):
    """
    Generate images using prompts and save them with corresponding filenames.
    """
    # Load configuration and model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Set sampler
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # Prepare output directory
    os.makedirs(opt.outdir, exist_ok=True)

    # Start generation
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompt, filename in tqdm(prompts_with_filenames, desc="Generating Images"):
                    print('Prompt: ', prompt)
                    # Prepare conditioning
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(opt.n_samples * [""])
                    c = model.get_learned_conditioning(prompt)

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                    )

                    # Decode and save image
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_sample = (255.0 * rearrange(x_samples_ddim[0], "h w c -> h w c")).astype(np.uint8)
                    img = Image.fromarray(x_sample)
                    img.save(os.path.join(opt.outdir, filename))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label_file",
        type=str,
        default="label_with_prompt_1.json",
        help="path to the label file containing prompts and filenames",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="outputs/generated-images",
        help="directory to save generated images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="number of samples per prompt",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["full", "autocast"],
        default="autocast",
        help="evaluate at this precision",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["generated_text", "prompt_w_label", "prompt_w_suffix"],
        default="generated_text",
    )
    opt = parser.parse_args()

    # Load prompts and filenames from label file
    with open(opt.label_file, "r") as f:
        label_data = json.load(f)
    prompts_with_filenames = [
        (entry[opt.prompt_type], entry["image"]) for entry in label_data if entry["generated_text"]
    ]

    # Generate images
    generate_images_from_prompts(opt, prompts_with_filenames)


if __name__ == "__main__":
    main()
