"""
Implement the diffusion model for segmentation

Mixture of 
https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py

"""

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

import numpy as np

from MedSegDGSSL.network.segnet.UNet_Diff import diff_unet
from MedSegDGSSL.network.build import NETWORK_REGISTRY

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(loss, optimizer, **kwargs):
    loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape)-1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)


class MedSegDiffusion(nn.Module):
    def __init__(self,
                 model_cfg:nn.Module,
                 *,
                 betas = None):
        super().__init__()

        self.denoise_fn = diff_unet(model_cfg=model_cfg)
        self.num_timesteps = model_cfg.TIME_STEPS
        self.out_channels = model_cfg.OUT_CHANNELS
        self.loss_type = model_cfg.DIFFUSION.LOSS

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(self.num_timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.-alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.-alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        #### y: seg, x: input image

    def q_mean_variance(self, y_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, y_start.shape) * y_start
        variance = extract(1. - self.alphas_cumprod, t, y_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, y_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, y_t.shape) * noise
        )
    
    def q_posterior(self, y_start, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_start +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_variance = extract(self.posterior_variance, t, y_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, y, x, t, clip_denoised: bool):
        # y_recon = self.predict_start_from_noise(y, t=t, noise=self.denoise_fn(y=y, x=x, t=t))
        y_recon = self.denoise_fn(y=y, x=x, t=t)

        if clip_denoised:
            y_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(y_start=y_recon, y_t=y, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, y, x, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *y.shape, y.device
        model_mean, _, model_log_variance = self.p_mean_variance(y=y, x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(y.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(y.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, img, shape):
        # shape for output segmentation map
        device = img.device

        b = shape[0]
        seg = torch.randn(size=shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            seg = self.p_sample(seg, img, torch.full((b,), i, device=device, dtype=torch.long))
        return seg
    
    @torch.no_grad()
    def forward(self, img):
        batch_size, c, *image_size = img.shape
        channels = self.out_channels
        return self.p_sample_loop(img, (batch_size, channels, *image_size))

    def q_sample(self, y_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, y_start.shape) * y_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, y_start.shape) * noise
        )
    
    def p_losses(self, y_start, x, t, noise = None):
        b, c, h, w = y_start.shape
        noise = default(noise, lambda: torch.randn_like(y_start))

        y_noisy = self.q_sample(y_start=y_start, t=t, noise=noise)
        y_pred = self.denoise_fn(y=y_noisy, x=x, t=t)
        target = y_start

        if self.loss_type == 'l1':
            loss = (target - y_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, y_pred)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(target, y_pred)
        else:
            raise NotImplementedError()

        return loss

    def get_p_losses(self, x, y):
        b, device = y.shape[0], y.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(y, x, t)
    
    def get_seg_losses(self, x, y, segloss_func):
        b, device = y.shape[0], y.device
        t = torch.randint(0, self.num_timesteps//10, (b,), device=device).long()
        noisy = self.q_sample(y, t=t)
        return segloss_func(self.denoise_fn(y=noisy, x=x,
                                            t=t, with_context=False), y)

@NETWORK_REGISTRY.register()
def medseg_diff(model_cfg):
    seg_diff = MedSegDiffusion(model_cfg=model_cfg)
    return seg_diff
