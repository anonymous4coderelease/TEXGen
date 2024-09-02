from dataclasses import dataclass, field
import math
import random
from contextlib import contextmanager
import colorsys

import cv2
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import profiler
from torchvision.transforms import v2 as transform

import spuv
from spuv.utils.misc import get_device
from spuv.systems.base import BaseLossConfig, BaseSystem
from spuv.utils.ops import binary_cross_entropy, get_plucker_rays
from spuv.utils.typing import *
from spuv.models.lpips import LPIPS
from spuv.utils.misc import time_recorder as tr
from spuv.utils.snr_utils import compute_snr_from_scheduler, get_weights_from_timesteps
from spuv.models.perceptual_loss import VGGPerceptualLoss
from spuv.models.renderers.rasterize import NVDiffRasterizerContext
from spuv.utils.mesh_utils import uv_padding
from spuv.utils.nvdiffrast_utils import *
from spuv.utils.lit_ema import LitEma
from spuv.utils.image_metrics import SSIM, PSNR

from diffusers import (
    DDPMScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    )
from spuv.systems.spuv_condition import PointUVDiffusion as SpuvBaseSystem


class PointUVDiffusion(SpuvBaseSystem):
    @dataclass
    class Config(SpuvBaseSystem.Config):
        test_start_i: int = 0

    def forward(self,
                condition: Dict[str, Any],
                diffusion_data: Dict[str, Any],
                condition_drop=None,
                ) -> Dict[str, Any]:
        mask_map = diffusion_data["mask_map"]
        position_map = diffusion_data["position_map"]
        timesteps = diffusion_data["timesteps"]
        input_tensor = diffusion_data["noisy_images"]

        text_embeddings = condition["text_embeddings"]
        image_embeddings = condition["image_embeddings"]
        clip_embeddings = [text_embeddings, image_embeddings]

        mesh = condition["mesh"]

        image_info = {
            'mvp_mtx_cond': condition["mvp_mtx_cond"],
            'rgb_cond': condition["rgb_cond"],
        }

        if condition_drop is None and self.training:
            condition_drop = torch.rand(input_tensor.shape[0], device=input_tensor.device) < self.cfg.condition_drop_rate
            condition_drop = condition_drop.float()
        elif condition_drop is None:
            condition_drop = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
        #spuv.info(f"Condition drop rate: {condition_drop}")
        output, addition_info = self.backbone(
           input_tensor,
           mask_map,
           position_map,
           timesteps,
           clip_embeddings,
           mesh,
           image_info,
           data_normalization=self.cfg.data_normalization,
           condition_drop=condition_drop,
        )

        return output, addition_info

    def prepare_condition_info(self, batch):
        mesh = batch["mesh"]
        mvp_mtx_cond = batch["mvp_mtx_cond"]
        uv_map_gt = batch["uv_map"]
        image_height = batch["height"]
        image_width = batch["width"]

        # Online rendering the condition image
        background_color = self.render_background_color
        rgb_cond = render_batched_meshes(self.ctx, mesh, uv_map_gt, mvp_mtx_cond, image_height, image_width, background_color)

        if self.cfg.cond_rgb_perturb and self.training:
            B, Nv, H, W, C = rgb_cond.shape
            rgb_cond = rearrange(rgb_cond, "B Nv H W C -> (B Nv) C H W")
            rgb_cond = self.data_augmentation(rgb_cond, background_color)
            rgb_cond = rearrange(rgb_cond, "(B Nv) C H W -> B Nv H W C", B=B, Nv=Nv)

        prompt = batch["prompt"]
        #with torch.no_grad():
        text_embeddings = self.image_tokenizer.process_text(prompt).to(dtype=self.dtype)
        image_embeddings = self.image_tokenizer.process_image(rgb_cond).to(dtype=self.dtype)

        condition_info = {
            "mesh": mesh,
            "mvp_mtx_cond": mvp_mtx_cond,
            "rgb_cond": rgb_cond,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "prompt": prompt,
        }

        return condition_info

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if (
                self.true_global_step < self.cfg.recon_warm_up_steps
                or self.cfg.train_regression
        ):
            self.train_regression = True
        else:
            self.train_regression = False

        if batch is None:
            spuv.info("Received None batch, skipping.")
            return None

        try:
            with torch.cuda.amp.autocast(enabled=False):
                if self.use_ema and self.val_with_ema:
                    with self.ema_scope("Validation with ema weights"):
                        texture_map_outputs = self.test_pipeline(batch)
                else:
                    spuv.info("Validation without ema weights")
                    texture_map_outputs = self.test_pipeline(batch)
        except Exception as e:
            spuv.info(f"Error in test pipeline: {e}")
            return None

        render_images = {}
        # background_color = self.render_background_color #if not self.cfg.random_background_color else self.generate_random_color()

        scene_id = batch["scene_id"]
        assert len(scene_id) == 1
        scene_id = scene_id[0]

        background_color = [0.5, 0.5, 0.5]
        # to tensor
        background_color = torch.tensor(background_color, device=get_device())
        for key in ["pred_x0", "gt_x0", "baked_texture"]:
            if key == "pred_x0":
                pos_map = torch.flip(texture_map_outputs["position_map"], dims=[2]) + 0.5
                mask_map = torch.flip(texture_map_outputs["mask_map"], dims=[2])

                mask_format = [{
                    "type": "grayscale",
                    "img": rearrange(mask_map, "B C H W-> (B H) (W C)"),
                    "kwargs": {"cmap": None, "data_range": None},
                }]

                self.save_image_grid(
                    f"it{self.true_global_step}-test/{key}/mask_{scene_id}.jpg",
                    mask_format,
                    name=f"test_step_output_{self.global_rank}_{batch_idx}",
                    step=self.true_global_step,
                )

                pos_format = [{
                    "type": "rgb",
                    "img": rearrange(pos_map, "B C H W-> (B H) W C"),
                    "kwargs": {"data_format": "HWC"},
                }]

                self.save_image_grid(
                    f"it{self.true_global_step}-test/{key}/pos_{scene_id}.jpg",
                    pos_format,
                    name=f"test_step_output_{self.global_rank}_{batch_idx}",
                    step=self.true_global_step,
                )


            value = texture_map_outputs[key]
            if self.cfg.data_normalization:
                img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"]
            else:
                img = value * texture_map_outputs["mask_map"]
            # Important to flip the uv map for possible meshlab loading, for rendering using NvDiffRasterizer, do not flip!
            flip_img = torch.flip(img, dims=[2])

            img_format = [{
                "type": "rgb",
                "img": rearrange(flip_img, "B C H W-> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/{key}/{scene_id}.jpg",
                img_format,
                name=f"test_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

            img = rearrange(img, "B C H W -> B H W C")
            mvp_mtx = batch['mvp_mtx']
            mesh = batch['mesh']
            height = batch['height']
            width = batch['width']
            # TODO: UV padding
            # if key != "pred_x0":
            if True:
                # no need to pad
                #pad_img = img
                pad_img = uv_padding(img.squeeze(0), texture_map_outputs['mask_map'].squeeze(0).squeeze(0),
                                     iterations=2)
            else:
                pad_img = torch.flip(pad_img, dims=[1])
            #pad_img = uv_padding(img.squeeze(0), texture_map_outputs['mask_map'].squeeze(0).squeeze(0), iterations=2)
            render_out = render_batched_meshes(self.ctx, mesh, pad_img, mvp_mtx, height, width, background_color)
            # render_out = render_batched_meshes(self.ctx, mesh, img, mvp_mtx, height, width, background_color)
            img_format = [{
                "type": "rgb",
                "img": rearrange(render_out, "B (V1 V2) H W C -> (B V1 H) (V2 W) C", V1=4),
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/render_{key}/{scene_id}.jpg",
                img_format,
                name=f"test_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

            condition_img_format = [{
                "type": "rgb",
                "img": render_out[0][0],
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/render_{key}/condition_{scene_id}.jpg",
                condition_img_format,
                name=f"test_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

            render_images[key] = torch.clamp(rearrange(render_out, "B V H W C -> (B V) C H W"), min=0, max=1)

        if self.cfg.test_save_mid_result and len(texture_map_outputs["mid_result"]) != 0:
            for i, mid_result in enumerate(texture_map_outputs["mid_result"]):
                for key, value in mid_result.items():
                    if self.cfg.data_normalization:
                        img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"]
                    else:
                        img = value * texture_map_outputs["mask_map"]

                    background_color = self.render_background_color
                    img = rearrange(img, "B C H W -> B H W C")
                    mvp_mtx = batch['mvp_mtx']
                    mesh = batch['mesh']
                    height = batch['height']
                    width = batch['width']
                    # TODO: UV padding
                    pad_img = uv_padding(img.squeeze(0), texture_map_outputs['mask_map'].squeeze(0).squeeze(0),
                                         iterations=2)
                    render_out = render_batched_meshes(self.ctx, mesh, pad_img, mvp_mtx, height, width, background_color)
                    # render_out = render_batched_meshes(self.ctx, mesh, img, mvp_mtx, height, width, background_color)
                    img_format = [{
                        "type": "rgb",
                        "img": rearrange(render_out, "B (V1 V2) H W C -> (B V1 H) (V2 W) C", V1=4),
                        "kwargs": {"data_format": "HWC"},
                    }]

                    self.save_image_grid(
                        f"it{self.true_global_step}-test/{self.global_rank}_{batch_idx}/render_{key}_{i}.jpg",
                        img_format,
                        name=f"test_step_output_{self.global_rank}_{batch_idx}",
                        step=self.true_global_step,
                    )

    def test_pipeline(self, batch):
        diffusion_data = self.prepare_diffusion_data(batch)
        condition_info = self.prepare_condition_info(batch)

        B, C, H, W = diffusion_data["mask_map"].shape
        device = get_device()
        test_num_steps = self.cfg.test_num_steps
        self.test_scheduler.set_timesteps(test_num_steps, device=device)
        timesteps = self.test_scheduler.timesteps

        noise = torch.randn((B, 3, H, W), device=device, dtype=self.dtype)
        noisy_images = noise
        mid_result = []

        start_i = self.cfg.test_start_i
        if start_i > 0:
            timestep = timesteps[start_i].repeat(B)
            sample_images = rearrange(batch["uv_map"], "B H W C -> B C H W").to(dtype=self.dtype)
            if self.cfg.data_normalization:
                sample_images = (sample_images * 2 - 1)
            noisy_images = self.noise_scheduler.add_noise(
                    sample_images * self.cfg.train_image_scaling,
                    noise,
                    timestep
                )
        for i, t in enumerate(timesteps):
            assert not self.train_regression

            if i < start_i:
                continue

            timestep = t.repeat(B)
            diffusion_data["timesteps"] = timestep
            diffusion_data["noisy_images"] = noisy_images * diffusion_data["mask_map"]
            if self.cfg.train_image_scaling != 1.0:
                diffusion_data["noisy_images"] = diffusion_data["noisy_images"] / (
                        diffusion_data["noisy_images"].std(dim=[1, 2, 3], keepdim=True)+1e-6)

            if torch.isnan(diffusion_data["noisy_images"]).any():
                print("Nan in noisy_images")
                breakpoint()

            cond_step_out, addition_info = self(condition_info, diffusion_data)
            if (
                    self.cfg.test_cfg_scale != 0.0
                    and self.cfg.guidance_interval[0] <= i / len(timesteps) <= self.cfg.guidance_interval[1]
            ):
                uncond_step_out, _ = self(condition_info, diffusion_data, condition_drop=torch.ones(B, device=device))
                step_out = uncond_step_out + self.cfg.test_cfg_scale * (cond_step_out - uncond_step_out)
                # Apply guidance rescale. From paper [Common Diffusion Noise Schedules
                # and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf) section 3.4.
                if self.cfg.guidance_rescale != 0:
                    std_pos = cond_step_out.std(dim=list(range(1, cond_step_out.ndim)), keepdim=True)
                    std_cfg = step_out.std(dim=list(range(1, step_out.ndim)), keepdim=True)
                    # Fuse equation 15,16 for more efficient computation.
                    step_out *= self.cfg.guidance_rescale * (std_pos / std_cfg) + (1 - self.cfg.guidance_rescale)
            else:
                step_out = cond_step_out

            noisy_images = self.test_scheduler.step(step_out, t, diffusion_data["noisy_images"]).prev_sample
            x0_pred = self.test_scheduler.step(step_out, t, diffusion_data["noisy_images"]).pred_original_sample
            if torch.isnan(noisy_images).any():
                print("Nan in noisy_images")
                breakpoint()
            mid_result.append(
                {
                    "x0": x0_pred,
                }
            )

        pred_x0 = noisy_images
        texture_map_outputs = {
            "pred_x0": pred_x0,
            "baked_texture": addition_info['baked_texture'],
            "gt_x0": diffusion_data["sample_images"],
            "mask_map": diffusion_data["mask_map"],
            "mid_result": mid_result,
            "position_map": diffusion_data["position_map"],
        }

        return texture_map_outputs
