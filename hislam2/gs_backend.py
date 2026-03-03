import os
import random
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange
from munch import munchify
from lietorch import SE3, SO3

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from hislam2.util.utils import Log, clone_obj, mask_feature_mean, distinct_colors
from hislam2.gaussian.renderer import render
from hislam2.gaussian.utils.loss_utils import (
    l1_loss,
    ssim,
    separation_loss,
    cohesion_loss,
    kl_regularization_loss,
    prediction_loss,
)
from hislam2.gaussian.scene.gaussian_model import GaussianModel
from hislam2.gaussian.utils.graphics_utils import getProjectionMatrix2
from hislam2.gaussian.utils.slam_utils import (
    update_pose,
    to_se3_vec,
    get_loss_normal,
    get_loss_mapping_rgbd,
)
from hislam2.gaussian.utils.camera_utils import Camera
from hislam2.gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf
from hislam2.gaussian.gui import gui_utils, slam_gui
from hislam2.gaussian.semantics.panoptic_mask_generator import MaskGenerator
from hislam2.gaussian.semantics.mask_reader import (
    read_gt_masks,
    read_sam3_masks,
    sam_masks_semantic_image,
    sam3_dict_to_tensor,
)
from hislam2.gaussian.semantics.predictor import Predictor
from hislam2.gaussian.utils.post_processing import cluster_hdbscan


class GSBackEnd(mp.Process):
    """Gaussian Splatting back-end that runs dense mapping optimization."""

    def __init__(self, config, save_dir, use_gui=False):
        super().__init__()
        self.config = config

        self.iteration_count = 0
        self.optimize_ins_feats_step = config["Training"]["optimize_ins_feats_step"]
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = save_dir
        self.use_gui = use_gui

        self.opt_params = munchify(config["opt_params"])
        self.config["Training"]["monocular"] = False

        # Gaussian model
        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.empty_ins_feats = torch.tensor(
            [0, 0, 0, 0, 0, 0], dtype=torch.float32, device="cuda"
        )

        self.cameras_extent = 6.0
        self.set_hyperparams()

        # Semantic predictor head
        self.predictor = Predictor(
            input_dim=6,
            hidden_dim=256,
            output_dim=self.config["masks"]["no_classes"],
        ).cuda()
        self.predictor.train()
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=0.001
        )
        self.predictor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.predictor_optimizer, gamma=0.9
        )

        # TensorBoard writer
        if SummaryWriter:
            self.writer = SummaryWriter(
                log_dir=os.path.join(save_dir, "tensorboard")
            )
        else:
            self.writer = None

        if self.use_gui:
            self._init_gui()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_gui(self):
        """Spawn the interactive GUI process."""
        self.q_main2vis = mp.Queue()
        self.q_vis2main = mp.Queue()
        self.params_gui = gui_utils.ParamsGUI(
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=self.q_main2vis,
            q_vis2main=self.q_vis2main,
        )
        gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
        gui_process.start()
        time.sleep(3)

    def set_hyperparams(self):
        """Load training hyper-parameters from config into instance attributes."""
        cfg = self.config["Training"]
        self.init_itr_num           = cfg["init_itr_num"]
        self.init_gaussian_update   = cfg["init_gaussian_update"]
        self.init_gaussian_reset    = cfg["init_gaussian_reset"]
        self.init_gaussian_th       = cfg["init_gaussian_th"]
        self.init_gaussian_extent   = self.cameras_extent * cfg["init_gaussian_extent"]
        self.gaussian_update_every  = cfg["gaussian_update_every"]
        self.gaussian_update_offset = cfg["gaussian_update_offset"]
        self.gaussian_th            = cfg["gaussian_th"]
        self.gaussian_extent        = self.cameras_extent * cfg["gaussian_extent"]
        self.gaussian_reset         = cfg["gaussian_reset"]
        self.size_threshold         = cfg["size_threshold"]
        self.window_size            = cfg["window_size"]
        self.lambda_dnormal         = cfg["lambda_dnormal"]

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_scalar(self, tag, value, step):
        """Write a scalar to TensorBoard if a writer is available."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_instance_feats(self, tag, ins_feat, step):
        """Log a 6-channel instance feature map as two normalised RGB images."""
        if not self.writer:
            return
        C, H, W = ins_feat.shape
        assert C == 6, f"Expected 6-channel feature map, got {C}"
        cpu = ins_feat.clone().cpu().detach()
        t1, t2 = torch.split(cpu, 3, dim=0)
        t1 = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-5)
        t2 = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-5)
        self.writer.add_image(f"{tag}1", t1, global_step=step, dataformats="CHW")
        self.writer.add_image(f"{tag}2", t2, global_step=step, dataformats="CHW")
        self.writer.flush()

    def log_rgb_images(self, tag, image, step):
        """Log a 3-channel RGB image to TensorBoard."""
        if not self.writer:
            return
        C, H, W = image.shape
        assert C == 3, f"Expected 3-channel RGB image, got {C}"
        tensor_image = image.clone().cpu().detach()
        self.writer.add_image(tag, tensor_image, global_step=step, dataformats="CHW")
        self.writer.flush()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render(self, viewpoint):
        """Render *viewpoint* with the current Gaussians and return the full package."""
        return render(viewpoint, self.gaussians, self.background, self.empty_ins_feats)

    def _unpack_render_pkg(self, pkg):
        """Unpack a render package into its seven components.

        Returns:
            (image, viewspace_pts, vis_filter, radii, depth, alpha, ins_feat)
        """
        return (
            pkg["render"],
            pkg["viewspace_points"],
            pkg["visibility_filter"],
            pkg["radii"],
            pkg["depth"],
            pkg["alpha"],
            pkg["rendered_features"],
        )

    # ------------------------------------------------------------------
    # Semantic helpers
    # ------------------------------------------------------------------

    def _load_semantic_masks(self, tstamp):
        """Load and GPU-transfer SAM3 + semantic masks for *tstamp*.

        Returns:
            sam_masks    (Tensor): Binary SAM instance masks  [N, H, W].
            sem_masks    (Tensor | None): Semantically grouped masks.
            sem_mask_ids (Tensor | None): Class label per semantic mask.
        """
        masks_dir = self.config["masks"]["sam3_masks_dir"]
        sam_masks_dict = read_sam3_masks(tstamp, masks_dir)
        sem_masks, sem_mask_ids = sam_masks_semantic_image(
            sam_masks_dict, tstamp, masks_dir
        )
        sam_masks = sam3_dict_to_tensor(sam_masks_dict).cuda()
        if sem_masks is not None:
            sem_masks    = sem_masks.cuda()
            sem_mask_ids = sem_mask_ids.cuda()
        return sam_masks, sem_masks, sem_mask_ids

    def _compute_mask_feature_losses(
        self, ins_feat, sam_masks, sem_masks, sem_mask_ids
    ):
        """Compute instance-feature losses and return them individually.

        Returns a dict with keys ``s_loss``, ``c_loss``, ``r_loss``, ``p_loss``.
        ``c_loss`` is ``None`` when *sem_masks* is ``None``;
        ``p_loss`` is ``None`` when *sem_mask_ids* is ``None``.
        """
        # Separation loss over SAM instance masks
        sam_feat_mean = mask_feature_mean(ins_feat, sam_masks)
        s_loss = separation_loss(sam_feat_mean)

        # Cohesion loss over semantic masks (if available)
        c_loss = None
        sem_feat_mean = sam_feat_mean  # fallback used for prediction loss
        if sem_masks is not None:
            sem_feat_mean = mask_feature_mean(ins_feat, sem_masks)
            c_loss = cohesion_loss(sem_masks, ins_feat, sem_feat_mean)

        # Prediction loss (classification head)
        p_loss = None
        if sem_mask_ids is not None:
            predictions = self.predictor(sem_feat_mean)
            p_loss = prediction_loss(predictions, sem_mask_ids)

        # KL regularisation over the full Gaussian field
        r_loss = kl_regularization_loss(
            self.gaussians.get_ins_feat,
            self.gaussians.get_xyz,
            num_of_samples=1000,
            num_of_neighbors=5,
        )

        return dict(s_loss=s_loss, c_loss=c_loss, r_loss=r_loss, p_loss=p_loss)

    def _accumulate_semantic_loss(self, losses):
        """Sum individual feature losses into a single optimisation target."""
        total = losses["s_loss"] + losses["r_loss"]
        if losses["c_loss"] is not None:
            total = total + self.opt_params.lambda_cohesion * losses["c_loss"]
        if losses["p_loss"] is not None:
            total = total + 0.1 * losses["p_loss"]
        return total

    # ------------------------------------------------------------------
    # Gaussian density / adaptive control helpers
    # ------------------------------------------------------------------

    def _update_densification_stats(
        self, viewspace_pts_list, vis_filter_list, radii_list
    ):
        """Accumulate max-radii and gradient stats across all rendered viewpoints."""
        for viewspace_pts, vis_filter, radii in zip(
            viewspace_pts_list, vis_filter_list, radii_list
        ):
            self.gaussians.max_radii2D[vis_filter] = torch.max(
                self.gaussians.max_radii2D[vis_filter],
                radii[vis_filter],
            )
            self.gaussians.add_densification_stats(viewspace_pts, vis_filter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_track_data(self, packet):
        """Ingest a tracking packet: apply pose updates, register viewpoints, map."""
        # Lazily build projection matrix from the first packet
        if not hasattr(self, "projection_matrix"):
            H, W = packet["images"].shape[-2:]
            self.K = K = list(packet["intrinsics"][0]) + [W, H]
            self.projection_matrix = getProjectionMatrix2(
                znear=0.01, zfar=100.0,
                fx=K[0], fy=K[1], cx=K[2], cy=K[3], W=W, H=H,
            ).transpose(0, 1).cuda()

        # Apply incremental pose / scale corrections to existing Gaussians
        if packet["pose_updates"] is not None:
            with torch.no_grad():
                tstamps = packet["tstamp"]
                indices = (
                    tstamps.unsqueeze(1) == self.gaussians.unique_kfIDs.unsqueeze(0)
                ).nonzero()[:, 0]
                updates       = packet["pose_updates"].cuda()[indices]
                updates_scale = packet["scale_updates"].cuda()[indices]

                self.gaussians._xyz[:] = (
                    updates * self.gaussians.get_xyz
                ) / updates_scale

                scale = self.gaussians.get_scaling / updates_scale
                self.gaussians._scaling[:] = (
                    self.gaussians.scaling_inverse_activation(scale)
                )

                rot = SO3(updates.data[:, 3:]) * SO3(self.gaussians.get_rotation)
                self.gaussians._rotation[:] = rot.data

        # Register new viewpoints and initialise the map on the very first frame
        w2c = SE3(packet["poses"]).matrix().cuda()
        for i, _ in enumerate(packet["viz_idx"]):
            idx    = packet["tstamp"][i].item()
            tstamp = idx
            viewpoint = Camera.init_from_tracking(
                packet["images"][i] / 255.0,
                packet["depths"][i],
                packet["normals"][i],
                w2c[i],
                idx,
                self.projection_matrix,
                self.K,
                tstamp,
            )
            if idx not in self.current_window:
                self.current_window = (
                    [idx] + self.current_window[:-1]
                    if len(self.current_window) > 10
                    else [idx] + self.current_window
                )
                if not self.initialized:
                    self.reset()
                    self.viewpoints[idx] = viewpoint
                    self.add_next_kf(
                        0, viewpoint, depth_map=packet["depths"][0].numpy(), init=True
                    )
                    self.initialize_map(0, viewpoint)
                    self.initialized = True
                elif idx not in self.viewpoints:
                    self.viewpoints[idx] = viewpoint
                    self.add_next_kf(
                        idx, viewpoint, depth_map=packet["depths"][i].numpy()
                    )
                else:
                    self.viewpoints[idx] = viewpoint

        self.map(self.current_window, iters=self.opt_params.iteration_per_scene)

        if self.use_gui:
            self._push_gui_update(viewpoint)

    def finalize(self):
        """Run final colour refinement, save PLY, and return sorted camera poses."""
        self.gaussians.save_ply(f"{self.save_dir}/3dgs_before_opt.ply")
        self.color_refinement(iteration_total=self.gaussians.max_steps)
        self.gaussians.save_ply(f"{self.save_dir}/3dgs_final.ply")
        torch.save(self.predictor.state_dict(), f"{self.save_dir}/predictor.pth")

        poses_cw = []
        for view in self.viewpoints.values():
            T_w2c          = np.eye(4)
            T_w2c[:3, :3]  = view.R.cpu().numpy()
            T_w2c[:3,  3]  = view.T.cpu().numpy()
            poses_cw.append(np.hstack(([view.tstamp], to_se3_vec(T_w2c))))
        poses_cw.sort(key=lambda x: x[0])
        return np.stack(poses_cw)

    @torch.no_grad()
    def eval_rendering(self, gtimages, gtdepthdir, traj, kf_idx):
        eval_rendering(
            gtimages, gtdepthdir, traj,
            self.gaussians, self.save_dir, self.background, self.empty_ins_feats,
            self.projection_matrix, self.K, kf_idx, iteration="after_opt",
        )
        eval_rendering_kf(
            self.viewpoints, self.gaussians, self.save_dir,
            self.background, self.empty_ins_feats, iteration="after_opt",
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.current_window  = []
        self.initialized     = False
        # Remove all Gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def initialize_map(self, cur_frame_idx, viewpoint):
        """Optimise Gaussians on the first frame until convergence."""
        for mapping_iteration in (pbar := trange(self.init_itr_num)):
            self.iteration_count += 1

            pkg = self._render(viewpoint)
            image, viewspace_pts, vis_filter, radii, depth, alpha, ins_feat = (
                self._unpack_render_pkg(pkg)
            )

            loss = get_loss_mapping_rgbd(self.config, image, depth, viewpoint)

            # Add semantic losses after a warm-up period
            optimize_sem = (
                self.iteration_count > self.optimize_ins_feats_step
                and self.iteration_count % self.opt_params.ins_feat_optimization_per_step == 0
            )
            if optimize_sem:
                sam_masks, sem_masks, _sem_ids = self._load_semantic_masks(
                    viewpoint.tstamp
                )
                sam_feat_mean = mask_feature_mean(ins_feat, sam_masks)
                s_loss = separation_loss(sam_feat_mean)
                if sem_masks is not None:
                    sem_feat_mean = mask_feature_mean(ins_feat, sem_masks)
                    c_loss = cohesion_loss(sem_masks, ins_feat, sem_feat_mean)
                    loss   = loss + s_loss + c_loss * self.opt_params.lambda_cohesion
                else:
                    loss = loss + s_loss

            # Logging
            self._log_scalar("InitLoss/loss_init", loss.item(), self.iteration_count)
            if optimize_sem:
                self._log_scalar(
                    "InitLoss/separation_loss", s_loss.item(), self.iteration_count
                )
                if sem_masks is not None:
                    self._log_scalar(
                        "InitLoss/cohesion_loss", c_loss.item(), self.iteration_count
                    )
            if self.iteration_count % 50 == 0:
                self.log_instance_feats(
                    "InitLoss/PCA_RGB_Image", ins_feat, self.iteration_count
                )
                self.log_rgb_images(
                    "InitLoss/RGB_Image", image, self.iteration_count
                )

            loss.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[vis_filter] = torch.max(
                    self.gaussians.max_radii2D[vis_filter], radii[vis_filter]
                )
                self.gaussians.add_densification_stats(viewspace_pts, vis_filter)

                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            pbar.set_description(f"Init GS loss {loss.item():.3f}")

        Log("Initialized map")
        return pkg

    def map(self, current_window, iters, prune=False):
        """Run *iters* Gaussian mapping steps over the current sliding window."""
        if not current_window:
            return

        window_viewpoints = [self.viewpoints[i] for i in current_window]
        current_window_set = set(current_window)
        extra_viewpoints   = [
            vp for idx, vp in self.viewpoints.items() if idx not in current_window_set
        ]

        for _ in range(iters):
            self.iteration_count += 1
            optimize_sem = (
                self.iteration_count % self.opt_params.ins_feat_optimization_per_step == 0
            )

            loss_mapping = 0
            has_p_loss   = False
            viewspace_pts_list = []
            vis_filter_list    = []
            radii_list         = []

            # Window viewpoints + 2 random extra viewpoints for regularisation
            n_random   = min(2, len(extra_viewpoints))
            random_vps = [
                extra_viewpoints[i]
                for i in torch.randperm(len(extra_viewpoints))[:n_random].tolist()
            ]

            for viewpoint in window_viewpoints + random_vps:
                pkg = self._render(viewpoint)
                image, viewspace_pts, vis_filter, radii, depth, alpha, ins_feat = (
                    self._unpack_render_pkg(pkg)
                )

                # Geometry losses
                normal_loss  = get_loss_normal(depth, viewpoint)
                rgbd_loss    = get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
                loss_mapping = (
                    loss_mapping
                    + self.lambda_dnormal * normal_loss / 10.0
                    + rgbd_loss
                )

                # Semantic losses
                semantic_loss = 0
                if optimize_sem:
                    sam_masks, sem_masks, sem_mask_ids = self._load_semantic_masks(
                        viewpoint.tstamp
                    )
                    feat_losses   = self._compute_mask_feature_losses(
                        ins_feat, sam_masks, sem_masks, sem_mask_ids
                    )
                    semantic_loss = self._accumulate_semantic_loss(feat_losses)
                    loss_mapping  = loss_mapping + semantic_loss
                    if feat_losses["p_loss"] is not None:
                        has_p_loss = True

                # Logging
                self._log_scalar(
                    "Loss/normal_loss", normal_loss.item(), self.iteration_count
                )
                self._log_scalar(
                    "Loss/rgbd_loss", rgbd_loss.item(), self.iteration_count
                )
                if optimize_sem:
                    self._log_scalar(
                        "Loss/separation_loss",
                        feat_losses["s_loss"].item(), self.iteration_count,
                    )
                    self._log_scalar(
                        "Loss/kl_regularization_loss",
                        feat_losses["r_loss"].item(), self.iteration_count,
                    )
                    if feat_losses["c_loss"] is not None:
                        self._log_scalar(
                            "Loss/cohesion_loss",
                            feat_losses["c_loss"].item(), self.iteration_count,
                        )
                    if feat_losses["p_loss"] is not None:
                        self._log_scalar(
                            "Loss/prediction_loss",
                            feat_losses["p_loss"].item(), self.iteration_count,
                        )
                    self._log_scalar(
                        "Loss/semantic_loss",
                        semantic_loss.item(), self.iteration_count,
                    )
                if self.iteration_count % 100 == 0:
                    self.log_instance_feats(
                        "Loss/PCA_RGB_Image", ins_feat, self.iteration_count
                    )
                    self.log_rgb_images(
                        "Loss/RGB_Image", image, self.iteration_count
                    )

                viewspace_pts_list.append(viewspace_pts)
                vis_filter_list.append(vis_filter)
                radii_list.append(radii)

            # Isotropic scale regularisation
            scaling        = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1, keepdim=True))
            self._log_scalar(
                "Loss/isotropic_loss", isotropic_loss.mean().item(), self.iteration_count
            )
            loss_mapping = loss_mapping + 10 * isotropic_loss.mean()

            loss_mapping.backward()

            # Step predictor if prediction loss was computed
            if has_p_loss:
                self.predictor_optimizer.step()
                self.predictor_optimizer.zero_grad(set_to_none=True)

            # Gaussian adaptive control
            with torch.no_grad():
                self._update_densification_stats(
                    viewspace_pts_list, vis_filter_list, radii_list
                )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                elif self.iteration_count % self.gaussian_reset == 0:
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(vis_filter_list)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Colour refinement (post-hoc global optimisation)
    # ------------------------------------------------------------------

    def color_refinement(self, iteration_total):
        """Global colour + semantic refinement over all keyframes."""
        Log("Starting color refinement")
        self._setup_pose_optimizers()
        self._gs_refinement_loop(iteration_total)
        self._predictor_training_loop(iteration_total)
        self._finalize_segmentation()
        Log("Map refinement done")
        if self.writer:
            self.writer.close()

    def _setup_pose_optimizers(self):
        """Build per-keyframe pose (and optional exposure) optimizers."""
        opt_params = []
        for view in self.viewpoints.values():
            opt_params += [
                {
                    "params": [view.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": f"rot_{view.uid}",
                },
                {
                    "params": [view.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": f"trans_{view.uid}",
                },
            ]
            if self.config["Training"]["compensate_exposure"]:
                opt_params += [
                    {
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": f"exposure_a_{view.uid}",
                    },
                    {
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": f"exposure_b_{view.uid}",
                    },
                ]
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

    def _gs_refinement_loop(self, iteration_total):
        """Joint Gaussian + pose + semantic optimisation loop."""
        for iteration in (pbar := trange(1, iteration_total + 1)):
            # Sample a random keyframe
            viewpoint_cam = self.viewpoints[
                random.choice(list(self.viewpoints.keys()))
            ]

            pkg = self._render(viewpoint_cam)
            image, _, _, _, depth, _, ins_feat = self._unpack_render_pkg(pkg)

            # Exposure compensation
            image = torch.exp(viewpoint_cam.exposure_a) * image + viewpoint_cam.exposure_b

            # Load masks
            sam_masks, sem_masks, sem_mask_ids = self._load_semantic_masks(
                viewpoint_cam.tstamp
            )

            # Colour losses
            gt_image = viewpoint_cam.original_image.cuda()
            l1_color = l1_loss(image, gt_image)
            ssim_l   = 1.0 - ssim(image, gt_image)
            loss = (
                (1.0 - self.opt_params.lambda_dssim) * l1_color
                + self.opt_params.lambda_dssim * ssim_l
            )
            rgb_mapping_loss = get_loss_mapping_rgbd(
                self.config, image, depth, viewpoint_cam
            )
            loss = loss + rgb_mapping_loss

            # Normal loss — tapered after 7 000 iterations
            normal_weight = 1.0 if iteration < 7000 else 0.5
            loss = loss + self.lambda_dnormal * normal_weight * get_loss_normal(
                depth, viewpoint_cam
            )

            # Semantic losses
            optimize_sem = (
                iteration % self.opt_params.ins_feat_optimization_per_step == 0
            )
            p_loss = None
            if optimize_sem:
                feat_losses   = self._compute_mask_feature_losses(
                    ins_feat, sam_masks, sem_masks, sem_mask_ids
                )
                semantic_loss = self._accumulate_semantic_loss(feat_losses)
                loss          = loss + semantic_loss
                p_loss        = feat_losses["p_loss"]

            # Logging
            self._log_scalar("Refinement_Loss/color_refinement", loss.item(), iteration)
            self._log_scalar("Refinement_Loss/l1_loss",          l1_color.item(), iteration)
            self._log_scalar("Refinement_Loss/ssim_loss",        ssim_l.item(),   iteration)
            self._log_scalar("Refinement_Loss/rgb_mapping_loss", rgb_mapping_loss.item(), iteration)
            if optimize_sem:
                self._log_scalar(
                    "Refinement_Loss/separation_loss",
                    feat_losses["s_loss"].item(), iteration,
                )
                self._log_scalar(
                    "Refinement_Loss/kl_regularization_loss",
                    feat_losses["r_loss"].item(), iteration,
                )
                if feat_losses["c_loss"] is not None:
                    self._log_scalar(
                        "Refinement_Loss/cohesion_loss",
                        feat_losses["c_loss"].item(), iteration,
                    )
            if p_loss is not None:
                self._log_scalar(
                    "Refinement_Loss/prediction_loss", p_loss.item(), iteration
                )
            if iteration % 100 == 0:
                self.log_instance_feats(
                    "Refinement_Loss/PCA_RGB_Image", ins_feat, iteration
                )
                self.log_rgb_images("Refinement_Loss/RGB_Image", image, iteration)

            loss.backward()

            # Step predictor from prediction loss
            if p_loss is not None:
                self.predictor_optimizer.step()
                self.predictor_optimizer.zero_grad(set_to_none=True)
                if iteration % 100 == 0:
                    self.predictor_scheduler.step()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                update_gaussian = (
                    iteration % self.gaussian_update_every == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                update_pose(viewpoint_cam)

            if self.use_gui and iteration % 50 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians))
                )

            pbar.set_description(
                f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}"
            )

    def _predictor_training_loop(self, iteration_total):
        """Predictor-only fine-tuning loop with Gaussian parameters frozen."""
        pending_losses = []
        for iteration in (pbar := trange(1, iteration_total + 1)):
            viewpoint_cam = self.viewpoints[
                random.choice(list(self.viewpoints.keys()))
            ]
            _, sem_masks, sem_mask_ids = self._load_semantic_masks(viewpoint_cam.tstamp)

            if sem_mask_ids is None:
                continue

            ins_feat      = self._render(viewpoint_cam)["rendered_features"]
            sem_feat_mean = mask_feature_mean(ins_feat, sem_masks)

            # Detach so gradients only update the predictor head
            predictions = self.predictor(sem_feat_mean.clone().detach())
            p_loss      = prediction_loss(predictions, sem_mask_ids)
            pending_losses.append(p_loss)

            self._log_scalar("Prediction_head/prediction_loss", p_loss.item(), iteration)
            if iteration % 100 == 0:
                self.predictor_scheduler.step()

            if pending_losses:
                total_p_loss = sum(pending_losses)
                total_p_loss.backward()
                self.predictor_optimizer.step()
                self.predictor_optimizer.zero_grad(set_to_none=True)
                pending_losses = []
                pbar.set_description(
                    f"Prediction head lr {self.predictor_scheduler.get_last_lr()[0]:.3E}"
                    f" loss {total_p_loss.item():.3f}"
                )

    def _finalize_segmentation(self):
        """Cluster Gaussians by instance feature and assign class labels."""
        n_gaussians = self.gaussians.get_xyz.shape[0]
        if n_gaussians < 30:
            Log(f"Skipping segmentation: only {n_gaussians} Gaussians (need ≥ 30 for HDBSCAN)")
            return

        with torch.no_grad():
            cluster_labels = cluster_hdbscan(
                self.gaussians.get_ins_feat.cpu(),
                self.gaussians.get_xyz.cpu(),
                30,
            )
            self.gaussians.cluster_index      = cluster_labels.to(
                self.gaussians.get_ins_feat.device
            )
            self.gaussians.segmentation_label = torch.zeros_like(cluster_labels)

            mean_features = {}
            for label in np.unique(cluster_labels.cpu().numpy()):
                if label == -1:
                    continue  # skip HDBSCAN noise points
                mask          = cluster_labels == label
                cluster_feats = self.gaussians.get_ins_feat[mask]
                outs          = (
                    torch.softmax(self.predictor(cluster_feats.cuda()), dim=-1)
                    .mean(0)
                    .cpu()
                    .numpy()
                )
                self.gaussians.segmentation_label[label] = outs.argmax()
                mean_features[label] = {
                    "mean_feature": cluster_feats.mean(dim=0),
                    "label":        outs.argmax(),
                }
            self.gaussians.cluster_features = mean_features

        # Visualise segmentation for a sample of random viewpoints
        if not self.writer:
            return

        color_map = torch.stack(
            distinct_colors(self.config["masks"]["no_classes"]), dim=0
        ).to(self.gaussians.get_ins_feat.device)

        sample_keys = random.sample(
            list(self.viewpoints.keys()), min(10, len(self.viewpoints))
        )
        for vp_idx, vp_key in enumerate(sample_keys):
            viewpoint_cam = self.viewpoints[vp_key]
            render_feats  = self._render(viewpoint_cam)["rendered_features"]
            W, H          = render_feats.shape[1], render_feats.shape[2]
            flat_feats    = render_feats.permute(1, 2, 0).reshape(-1, 6)

            with torch.no_grad():
                seg_class = torch.argmax(self.predictor(flat_feats), dim=1)

            seg_img = color_map[seg_class].view(W, H, 3).permute(2, 0, 1)
            self.log_rgb_images("Final_seg", seg_img, vp_idx)

    # ------------------------------------------------------------------
    # GUI helper
    # ------------------------------------------------------------------

    def _push_gui_update(self, current_viewpoint):
        """Push the current Gaussian state and keyframe info to the GUI process."""
        keyframes = [self.viewpoints[i] for i in self.current_window]
        kf_window = {self.current_window[0]: self.current_window[1:]}
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                current_frame=current_viewpoint,
                keyframes=keyframes,
                kf_window=kf_window,
                gtcolor=current_viewpoint.original_image,
                gtdepth=current_viewpoint.depth.numpy(),
            )
        )
