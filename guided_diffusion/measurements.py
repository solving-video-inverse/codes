import torch
import torch.nn.functional as F

def uniform_kernel_1d(kernel_size: int, dtype=torch.float32):
    """Generate a 1D uniform blur kernel."""
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive")
    
    kernel = torch.ones(kernel_size, dtype=dtype)
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_kernel_1d(kernel_size: int, sigma: float, dtype=torch.float32):
    """Generate a 1D Gaussian kernel."""
    ax = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel

def temporal_blur(video_tensor: torch.Tensor, kernel_size_t: int):
    """
    Apply spatio-temporal Gaussian blur to a video tensor.

    Parameters:
    - video_tensor: A tensor of shape (B, C, T, H, W) where B is the batch size,
      T is the number of frames, C is the number of channels, H is the height, and W is the width.
    - kernel_size_t: Size of the Gaussian kernel in the temporal dimension.

    Returns:
    - Blurred video tensor of the same shape as the input.
    """
    device = video_tensor.device
    dtype = video_tensor.dtype
    B, C, T, H, W = video_tensor.shape

    # Generate Gaussian kernels for each dimension
    kernel_t = uniform_kernel_1d(kernel_size_t, dtype=dtype).to(device).view(1, 1, kernel_size_t, 1, 1)

    padding_t = kernel_size_t // 2

    # Apply temporal blur
    video_tensor = F.pad(video_tensor, (0, 0, 0, 0, padding_t, padding_t), mode='circular')
    video_tensor = F.conv3d(video_tensor.view(B * C, 1, T + 2 * padding_t, H, W), kernel_t, padding=0, groups=1).view(B, C, T, H, W)

    return video_tensor

def generate_random_mask(shape, pixel_ratio):
    """
    Generates a random binary mask with the given pixel ratio.

    Args:
        shape (tuple): Shape of the mask (B, C, T, H, W).
        pixel_ratio (float): Ratio of pixels to be set to 1.

    Returns:
        torch.Tensor: Random binary mask.
    """
    B, C, T, H, W = shape
    num_pixels = H * W
    num_ones = int(num_pixels * pixel_ratio)
    
    # Generate a flat array with the appropriate ratio of ones and zeros
    flat_mask = torch.zeros(num_pixels, dtype=torch.float32)
    flat_mask[:num_ones] = 1
    
    # Shuffle to randomize the positions of ones and zeros
    flat_mask = flat_mask[torch.randperm(num_pixels)]
    
    # Reshape to the original spatial dimensions and duplicate across channels
    mask = flat_mask.view(1, H, W)
    mask = mask.expand(B, C, T, H, W)
    
    return mask