import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer

class LGM(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()
        
        # Initialize options
        self.opt = opt
        
        # Define UNet model for feature extraction
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # Define a final convolutional layer
        self.conv = nn.Conv2d(14, 14, kernel_size=1)  # NOTE: Consider removing this if retraining

        # Initialize Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # Activation functions for different attributes
        self.pos_act = lambda x: x.clamp(-1, 1)  # Clamp positions between -1 and 1
        self.scale_act = lambda x: 0.1 * F.softplus(x)  # Apply softplus activation for scale
        self.opacity_act = lambda x: torch.sigmoid(x)  # Sigmoid activation for opacity
        self.rot_act = lambda x: F.normalize(x, dim=-1)  # Normalize rotations
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5  # Tanh activation for RGB colors; consider sigmoid if retraining

        # LPIPS loss for perceptual similarity
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # Remove lpips_loss from the state_dict
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def prepare_default_rays(self, device, elevation=0):
        # Prepare default rays from various camera viewpoints
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        # Generate camera poses from four different viewpoints
        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0)  # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            # Compute rays and Plücker coordinates
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy)  # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device)  # [V, 6, h, w]
        
        return rays_embeddings

    def forward_gaussians(self, images):
        # Process input images to predict Gaussian parameters
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)

        x = self.unet(images)  # [B*4, 14, h, w]
        x = self.conv(x)  # [B*4, 14, h, w]

        x = x.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
        
        # Uncomment below for visualization of Gaussian features
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        # Apply activations to obtain Gaussian parameters
        pos = self.pos_act(x[..., 0:3])  # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        # Concatenate parameters to form final Gaussians
        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)  # [B, N, 14]
        
        return gaussians

    def forward(self, data, step_ratio=1):
        # Perform forward pass and compute losses
        # data: output of the dataloader
        # return: loss and results

        results = {}
        loss = 0

        images = data['input']  # [B, 4, 9, h, W] - Input features
        
        # Predict Gaussians from input images
        gaussians = self.forward_gaussians(images)  # [B, N, 14]
        results['gaussians'] = gaussians

        # Use white background color for rendering
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
        
        # Render the Gaussians and get predictions
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image']  # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha']  # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        # Ground-truth images and masks
        gt_images = data['images_output']  # [B, V, 3, output_size, output_size]
        gt_masks = data['masks_output']  # [B, V, 1, output_size, output_size]

        # Apply masks to ground-truth images
        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        # Compute losses
        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            # Compute LPIPS loss for perceptual similarity
            loss_lpips = self.lpips_loss(
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # Compute PSNR metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
