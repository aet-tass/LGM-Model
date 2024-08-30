import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from core.options import Options

import kiui

class GaussianRenderer:
    def __init__(self, opt: Options):
        # Initialize GaussianRenderer with the given options
        self.opt = opt
        
        # Background color for rendering, set to white
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        # Intrinsics - Calculate and store projection matrix based on the field of view (FOV)
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1
        
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        # Render Gaussian objects from different camera views
        # gaussians: [B, N, 14] - Gaussian parameters
        # cam_view, cam_view_proj: [B, V, 4, 4] - Camera view and projection matrices
        # cam_pos: [B, V, 3] - Camera positions

        device = gaussians.device
        B, V = cam_view.shape[:2]

        images = []
        alphas = []
        for b in range(B):
            # Extract individual Gaussian components: position, opacity, scale, rotation, RGB colors
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float()  # [N, 3]

            for v in range(V):
                # Render novel views using the provided camera parameters
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.opt.output_size,
                    image_width=self.opt.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )

                # Instantiate the rasterizer with the settings
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to the image and calculate radii (on screen)
                rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=None,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )

                # Clamp the rendered image to valid pixel values
                rendered_image = rendered_image.clamp(0, 1)

                # Store the results for each view
                images.append(rendered_image)
                alphas.append(rendered_alpha)

        # Stack images and alphas into tensors for output
        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)

        return {
            "image": images,  # [B, V, 3, H, W] - Rendered images
            "alpha": alphas,  # [B, V, 1, H, W] - Alpha (transparency) channel
        }

    def save_ply(self, gaussians, path, compatible=True):
        # Save Gaussian parameters as a PLY file
        # gaussians: [B, N, 14] - Gaussian parameters
        # compatible: Save pre-activated Gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'Only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float()  # [N, 1, 3]

        # Prune Gaussians based on opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # Invert activation to make it compatible with the original PLY format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        # Convert Gaussian data to numpy arrays for saving as PLY
        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        # Create attribute names for the PLY file
        l = ['x', 'y', 'z']
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        # Define the data type for the PLY file
        dtype_full = [(attribute, 'f4') for attribute in l]

        # Prepare the elements for the PLY file
        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        # Save the elements to a PLY file
        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):
        # Load Gaussian parameters from a PLY file
        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        # Extract 3D positions (x, y, z)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading: ", xyz.shape[0])

        # Extract opacities and SH coefficients
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        # Extract scales and rotations
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if compatible:
            # Apply sigmoid and exponential activation to revert data format
            opacities = torch.sigmoid(torch.tensor(opacities))
            scales = torch.exp(torch.tensor(scales))
            shs = 0.28209479177387814 * torch.tensor(shs) + 0.5

        # Concatenate loaded Gaussian attributes into a single tensor
        loaded_gaussians = torch.cat((
            torch.tensor(xyz),
            opacities,
            scales,
            torch.tensor(rots),
            shs), dim=-1).float()

        return loaded_gaussians.unsqueeze(0)  # [1, N, 14] - Return with batch dimension
