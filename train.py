#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from ogsh import OGSH
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # OG-SH: observability-guided opacity dropout + mode-wise SH shrinkage (replaces DropGaussian's uniform dropout)
    ogsh = None
    if args.ogsh:
        cams = torch.stack([c.camera_center for c in scene.getTrainCameras()], dim=0)
        ogsh = OGSH(cams, p_max=args.ogsh_pmax, weight=args.ogsh_weight, tau=args.ogsh_tau, warmup=args.ogsh_warmup,
                    interval=args.ogsh_interval, max_shrink_pts=args.ogsh_max_shrink_pts, alloc=args.ogsh_alloc)
        print(f"[OG-SH] p_max={args.ogsh_pmax} lambda_OG={args.ogsh_weight} tau={args.ogsh_tau} warmup={args.ogsh_warmup} "
              f"refresh={args.ogsh_interval} alloc={args.ogsh_alloc} | {cams.shape[0]} training cameras")
    elif args.no_drop:
        print("[no_drop] DropGaussian dropout off (plain 3DGS)")

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # OG-SH Steps 1-2: refresh the observability cache if needed, then sample the opacity-dropout multiplier
        opacity_mult = ogsh.dropout(gaussians, iteration) if ogsh else None

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, is_train=True, iteration=iteration,
                            opacity_mult=opacity_mult, dg_dropout=not (args.ogsh or args.no_drop))
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # OG-SH Step 3: mode-wise SH shrinkage
        if ogsh:
            loss = loss + ogsh.shrinkage_loss(gaussians)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if ogsh and iteration in testing_iterations:
                print(f"\n[ITER {iteration}] {ogsh.summary()}")

            # Log and save
            training_report(dataset, tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    # optional (--screen_size_prune): vanilla 3DGS screen-size pruning after the first opacity reset
                    if args.screen_size_prune and iteration > opt.opacity_reset_interval:
                        size_threshold = 20
                    # optional (--white_suppress_strength): DNGaussian-style white-pixel suppression for white-background
                    # scenes -- Gaussians whose centre projects onto a white pixel of the current render get their
                    # opacity multiplied by the strength and their densification gradient reset.
                    if args.white_suppress_strength is not None and iteration >= args.white_suppress_start:
                        white_pix = image.min(dim=0).values > args.white_suppress_threshold  # [H,W]
                        xyz = gaussians.get_xyz
                        p_clip = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1) @ viewpoint_cam.full_proj_transform
                        ndc = p_clip[:, :3] / p_clip[:, 3:4].clamp(min=1e-6)
                        W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
                        px = ((ndc[:, 0] + 1) * 0.5 * W).long().clamp(0, W - 1)
                        py = ((1 - (ndc[:, 1] + 1) * 0.5) * H).long().clamp(0, H - 1)
                        white_mask = white_pix[py, px]
                        gaussians.xyz_gradient_accum[white_mask] = 0.0
                        gaussians._opacity[white_mask] = gaussians.inverse_opacity_activation(
                            gaussians.opacity_activation(gaussians._opacity[white_mask]) * args.white_suppress_strength)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(args, tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            render_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(args.model_path, config['name'], "ours_{}".format(iteration), "gt")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5000, 10000,])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # OG-SH (defaults = paper setting; Blender uses --ogsh_pmax 0.3)
    parser.add_argument("--ogsh", action="store_true", help="enable OG-SH (turns DropGaussian's uniform dropout off)")
    parser.add_argument("--ogsh_pmax", type=float, default=0.2, help="p_max in p_i = p_max (1 - o_i)")
    parser.add_argument("--ogsh_weight", type=float, default=0.01, help="lambda_OG, weight of the mode-wise SH shrinkage loss")
    parser.add_argument("--ogsh_tau", type=float, default=1e-2, help="tau, reference scale of directional support")
    parser.add_argument("--ogsh_warmup", type=int, default=2000, help="OG-SH is inactive up to this iteration")
    parser.add_argument("--ogsh_interval", type=int, default=500, help="observability refresh interval (also refreshed when N or SH degree changes)")
    parser.add_argument("--ogsh_max_shrink_pts", type=int, default=200000, help="size cap of the random Gaussian subset S for the shrinkage loss")
    parser.add_argument("--ogsh_alloc", type=str, default="guided", choices=["guided", "inverted", "uniform"],
                        help="dropout allocation; 'inverted'/'uniform' are ablations with the mean drop rate matched")
    # baselines / optional recipe switches
    parser.add_argument("--no_drop", action="store_true", help="disable DropGaussian's dropout without OG-SH (plain 3DGS)")
    parser.add_argument("--screen_size_prune", action="store_true", help="vanilla 3DGS screen-size pruning (threshold 20) after the first opacity reset")
    parser.add_argument("--white_suppress_strength", type=float, default=None, help="white-pixel opacity suppression strength (None = off)")
    parser.add_argument("--white_suppress_start", type=int, default=2000)
    parser.add_argument("--white_suppress_threshold", type=float, default=253 / 255)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(sys.argv[1:])
    assert not (args.ogsh and args.no_drop), "--ogsh already turns DropGaussian's dropout off"
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    import random, numpy as np   # safe_state() seeds with 0; re-seed with --seed (0 reproduces DropGaussian)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
