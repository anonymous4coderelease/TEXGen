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
from spuv.systems.spuv_base import PointUVDiffusion as SpuvBaseSystem


class PointUVDiffusion(SpuvBaseSystem):
    @dataclass
    class Config(SpuvBaseSystem.Config):
        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

    def configure(self):
        super().configure()
        self.image_tokenizer = spuv.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

    def prepare_diffusion_data(self, batch, noisy_images=None):
        device = get_device()
        uv_channel, uv_height, uv_width = batch["uv_channel"][0], batch["uv_height"][0], batch["uv_width"][0]
        batch_size = len(batch["mesh"])
        uv_shape = (batch_size, uv_channel, uv_height, uv_width)
        if self.training or "uv_map" in batch:
            sample_images = rearrange(batch["uv_map"], "B H W C -> B C H W").to(dtype=self.dtype)
            if self.cfg.data_normalization:
                sample_images = (sample_images * 2 - 1)
        else:
            sample_images = None

        if "mask_map" not in batch or "position_map" not in batch:
            position_map_, mask_map_ = rasterize_batched_geometry_maps(
                self.ctx, batch["mesh"],
                uv_height,
                uv_width
            )
            mask_map = rearrange(mask_map_, "B H W C-> B C H W").to(dtype=self.dtype)
            position_map = rearrange(position_map_, "B H W C -> B C H W").to(dtype=self.dtype)
        else:
            mask_map = rearrange(batch["mask_map"], "B H W -> B 1 H W").to(dtype=self.dtype)
            position_map = rearrange(batch["position_map"], "B H W C -> B C H W").to(dtype=self.dtype)

        if self.train_regression or not self.training:
            timesteps = torch.randint(self.num_train_timesteps-1, self.num_train_timesteps, (batch_size,), device=device)
        else:
            timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        timesteps = timesteps.long()

        if noisy_images is not None:
            noisy_images = noisy_images.to(dtype=self.dtype)
        else:
            noise = torch.randn(uv_shape, device=device, dtype=self.dtype)
            if sample_images is not None:
                noisy_images = self.noise_scheduler.add_noise(
                    sample_images * self.cfg.train_image_scaling,
                    noise,
                    timesteps
                )
                if torch.isnan(noisy_images).any():
                    print("Nan in noisy_images")
                    breakpoint()
            else:
                noisy_images = noise

        noisy_images *= mask_map
        if self.cfg.train_image_scaling != 1.0 and not self.train_regression:
            # x_t = x_t / x_t.std(axis=(1,2,3), keepdims=True)
            noisy_images = noisy_images / (noisy_images.std(dim=[1, 2, 3], keepdim=True)+1e-6)

        if (
                self.training
                and self.cfg.loss.use_min_snr_weight
                and not self.train_regression
        ):
            loss_weights = get_weights_from_timesteps(timesteps, self.snr_weights, mode='soft_min_snr', prediction_type=self.prediction_type)
        else:
            loss_weights = torch.ones_like(timesteps, device=device, dtype=self.dtype)

        diffusion_data = {
            "sample_images": sample_images,
            "position_map": position_map,
            "mask_map": mask_map,
            "timesteps": timesteps,
            "noise": noise,
            "noisy_images": noisy_images,
            "batch_loss_weights": loss_weights,
        }

        return diffusion_data

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

    def get_diffusion_loss(self, out, diffusion_data):
        loss = 0.0
        if self.prediction_type == "sample":
            gt = diffusion_data["sample_images"]
        elif self.prediction_type == "epsilon":
            gt = diffusion_data["noise"]
        elif self.prediction_type == "v_prediction":
            gt = self.noise_scheduler.get_velocity(
                diffusion_data["sample_images"],
                diffusion_data["noise"],
                diffusion_data["timesteps"]
            )
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")
        batch_loss_weights = diffusion_data["batch_loss_weights"]

        batch_mse_loss = torch.mean(F.mse_loss(out, gt, reduction="none"), dim=[1, 2, 3])
        mse_loss = torch.mean(batch_mse_loss * batch_loss_weights)
        loss += self.C(self.cfg.loss["diffusion_loss_dict"]["lambda_mse"]) * mse_loss

        batch_l1_loss = torch.mean(F.l1_loss(out, gt, reduction="none"), dim=[1, 2, 3])
        l1_loss = torch.mean(batch_l1_loss * batch_loss_weights)
        loss += self.C(self.cfg.loss["diffusion_loss_dict"]["lambda_l1"]) * l1_loss

        return loss

    def get_batched_pred_x0(self, out, timesteps, noisy_input):
        if self.prediction_type == "sample":
            pred_x0 = out
        elif self.prediction_type == "epsilon" or self.prediction_type == "v_prediction":
            pred_x0 = []
            for bs, t in enumerate(timesteps):
                x0: Float[Tensor, "3 H W"] = self.noise_scheduler.step(
                    out[bs],
                    t,
                    noisy_input[bs]
                ).pred_original_sample
                pred_x0.append(x0)
            pred_x0 = torch.stack(pred_x0, dim=0)
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")

        return pred_x0

    def get_render_loss(self, batch_loss_weights, uv_map_pred, uv_map_gt, mesh, mvp_mtx, height, width):
        loss = 0.0
        num_views = mvp_mtx.shape[1]
        # Online rendering the target image
        background_color = self.render_background_color if not self.cfg.random_background_color else self.generate_random_color()

        render_out = render_batched_meshes(self.ctx, mesh, uv_map_pred, mvp_mtx, height, width, background_color)
        render_gt = render_batched_meshes(self.ctx, mesh, uv_map_gt, mvp_mtx, height, width, background_color)

        rgb_out = rearrange(render_out, "B V H W C -> (B V) C H W")
        rgb_gt = rearrange(render_gt, "B V H W C -> (B V) C H W")

        tr.start("lpips_loss")
        view_lpips_loss = torch.mean(self.lpips_loss_fn(rgb_out, rgb_gt, input_range=(0, 1), ), dim=[1, 2, 3])
        batch_lpips_loss = torch.mean(rearrange(view_lpips_loss, "(B V) -> B V", V=num_views), dim=1)
        lpips_loss = torch.mean(batch_lpips_loss * batch_loss_weights)
        tr.end("lpips_loss")
        view_mse_loss = torch.mean(F.mse_loss(rgb_out, rgb_gt, reduction="none"), dim=[1, 2, 3])
        batch_mse_loss = torch.mean(rearrange(view_mse_loss, "(B V) -> B V", V=num_views), dim=1)
        mse_loss = torch.mean(batch_mse_loss * batch_loss_weights)

        view_l1_loss = torch.mean(F.l1_loss(rgb_out, rgb_gt, reduction="none"), dim=[1, 2, 3])
        batch_l1_loss = torch.mean(rearrange(view_l1_loss, "(B V) -> B V", V=num_views), dim=1)
        l1_loss = torch.mean(batch_l1_loss * batch_loss_weights)

        loss += self.C(self.cfg.loss["render_loss_dict"][f"lambda_render_lpips"]) * lpips_loss
        loss += self.C(self.cfg.loss["render_loss_dict"][f"lambda_render_mse"]) * mse_loss
        loss += self.C(self.cfg.loss["render_loss_dict"][f"lambda_render_l1"]) * l1_loss

        return loss, render_gt, render_out

    def on_check_train(self, batch, outputs):
        if (
                self.true_global_step < self.cfg.recon_warm_up_steps
                or self.cfg.train_regression
        ):
            self.train_regression = True
        else:
            self.train_regression = False

        if (
                self.global_rank == 0
                and self.cfg.check_train_every_n_steps > 0
                and self.true_global_step % (self.cfg.check_train_every_n_steps*10) == 0
        ):
            images = []
            texture_map_outputs = outputs["texture_map_outputs"]

            for key, value in texture_map_outputs.items():
                if self.cfg.data_normalization:
                    img = (value * 0.5 + 0.5) * outputs["mask_map"]
                else:
                    img = value * outputs["mask_map"]
                img_format = {
                    "type": "rgb",
                    "img": rearrange(img, "B C H W -> (B H) W C"),
                    "kwargs": {"data_format": "HWC"},
                }
                images.append(img_format)

            self.save_image_grid(
                f"it{self.true_global_step}-train.jpg",
                images,
                name="train_step_output",
                step=self.true_global_step,
            )

        if outputs['render_out'] is not None:
            images = [
                {
                    "type": "rgb",
                    "img": rearrange(outputs['render_out'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(outputs['render_gt'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": rearrange(outputs['rgb_cond'], "B V H W C -> (B H) (V W) C"),
                    "kwargs": {"data_format": "HWC"},
                }
            ]

            self.save_image_grid(
                f"it{self.true_global_step}-train-render.jpg",
                images,
                name="train_step_output",
                step=self.true_global_step,
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)
        torch.cuda.empty_cache()

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
        background_color = self.render_background_color #if not self.cfg.random_background_color else self.generate_random_color()
        for key in ["pred_x0", "gt_x0", "baked_texture"]:
            value = texture_map_outputs[key]
            if self.cfg.data_normalization:
                img = (value * 0.5 + 0.5) * texture_map_outputs["mask_map"]
            else:
                img = value * texture_map_outputs["mask_map"]
            # Important to flip the uv map for possible meshlab loading, for rendering using NvDiffRasterizer, do not flip!
            flip_img = torch.flip(img, dims=[2])
            # if key == "pred_x0":
            if False:
                hole_mask = 1 - torch.flip(texture_map_outputs["mask_map"], dims=[2]).squeeze(0).squeeze(0)
                #breakpoint()
                inpaint_image = (
                        cv2.inpaint(
                            (flip_img.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                            (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                            2,
                            cv2.INPAINT_TELEA,
                        )
                        / 255.0
                )
                pad_img = torch.from_numpy(inpaint_image).unsqueeze(0).to(flip_img)
                img_format = [{
                    "type": "rgb",
                    "img": rearrange(pad_img, "B H W C-> (B H) W C"),
                    "kwargs": {"data_format": "HWC"},
                }]

                self.save_image_grid(
                    f"it{self.true_global_step}-test/pad_{key}_{self.global_rank}_{batch_idx}.jpg",
                    img_format,
                    name=f"test_step_output_{self.global_rank}_{batch_idx}",
                    step=self.true_global_step,
                )

            img_format = [{
                "type": "rgb",
                "img": rearrange(flip_img, "B C H W-> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            }]

            self.save_image_grid(
                f"it{self.true_global_step}-test/{key}_{self.global_rank}_{batch_idx}.jpg",
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
                f"it{self.true_global_step}-test/render_{key}_{self.global_rank}_{batch_idx}.jpg",
                img_format,
                name=f"test_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
            )

            render_images[key] = torch.clamp(rearrange(render_out, "B V H W C -> (B V) C H W"), min=0, max=1)

        ssim_metric = self.ssim_metric_fn(render_images["pred_x0"], render_images["gt_x0"])
        psnr_metric = self.psnr_metric_fn(render_images["pred_x0"], render_images["gt_x0"])
        lpips_loss = self.lpips_loss_fn(render_images["pred_x0"], render_images["gt_x0"], input_range=(0, 1)).mean()
        self.log('val_ssim', ssim_metric, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_psnr', psnr_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_lpips', lpips_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        if self.cfg.test_save_json:
            save_str = ""
            for scene_id in batch["scene_id"]:
                save_str += str(scene_id) + " "

            self.save_json(
                f"it{self.true_global_step}-test/{self.global_rank}_{batch_idx}.json",
                save_str,
            )

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
        #breakpoint()
        mid_result = []
        for i, t in enumerate(timesteps):
            if self.train_regression:
                t = torch.tensor([self.num_train_timesteps-1], device=device)
                timestep = t.repeat(B)
                diffusion_data["timesteps"] = timestep
                diffusion_data["noisy_images"] = noisy_images * diffusion_data["mask_map"]
                step_out, addition_info = self(condition_info, diffusion_data)
                noisy_images = step_out
                break
            else:
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
                        "x_t": diffusion_data["noisy_images"],
                        "x_t_1": noisy_images,
                    }
                )
            #spuv.info("Test step {timestep}")

        pred_x0 = noisy_images
        texture_map_outputs = {
            "pred_x0": pred_x0,
            "baked_texture": addition_info['baked_texture'],
            "gt_x0": diffusion_data["sample_images"],
            "mask_map": diffusion_data["mask_map"],
            "mid_result": mid_result,
        }

        return texture_map_outputs
