import argparse
import os

import sys
sys.path.append(".")

import numpy as np
import torch as th

#th.cuda.set_device(4)

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from datasets.DAVIS import DAVISDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from einops import rearrange

from guided_diffusion.measurements import temporal_blur, generate_random_mask

import random

def set_random_seed(seed):
    # Set the random seed for Python's built-in random module
    random.seed(seed)
    
    # Set the random seed for NumPy
    np.random.seed(seed)
    
    # Set the random seed for PyTorch
    th.manual_seed(seed)
    
    # If you are using CUDA
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    
    # For deterministic operations (not necessary for all use cases)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cycle(dl):
    while True:
        for data in dl:
            yield data

# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration = 250, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    device = dist_util.dev()

    print("creating dataset and dataloader...")
    # Define the directory where the DAVIS dataset is stored
    data_dir = './sample/sample_data'

    # Create the datasets for sampling
    dataset = DAVISDataset(root_dir=f'{data_dir}')

    # Create the dataloaders for sampling
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f'Success loading {len(dataset)} videos')

    dl = cycle(dataloader)

    for i in range(len(dataset)):
        images = next(dl)
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        target = images.to(device)
        #target = target[:,:,:4]

        mask = generate_random_mask(target.shape, 0.5).to(device)

        measurement = temporal_blur(target, 13)

        sample, x_preds = diffusion.time_deconv_sample_loop(
            model,
            (args.batch_size * 16, 3, args.image_size, args.image_size),
            measurement=measurement,
            mask = mask,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs
        )

        ## target save
        video_path = os.path.join('./results/gif', f'{i+1:03d}')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        video_path = os.path.join(video_path, f'target.gif')
        video_tensor_to_gif(target[0], video_path)

        ## measurement save
        video_path = os.path.join('./results/gif', f'{i+1:03d}')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        video_path = os.path.join(video_path, f'measurement.gif')
        video_tensor_to_gif(measurement[0], video_path)

        ## sample save
        sample = th.clamp(sample, -1, 1)
        sample = rearrange(sample, 't c h w -> c t h w')
        print(sample.shape)

        video_path = os.path.join('./results/gif', f'{i+1:03d}')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        video_path = os.path.join(video_path, f'sample.gif')
        video_tensor_to_gif(unnormalize_img(sample), video_path)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # seed = 42
    # set_random_seed(seed)
    main()
