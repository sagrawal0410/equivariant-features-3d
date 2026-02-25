from dataclasses import dataclass, asdict
import viser
import viser.transforms as tf
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
import kornia
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from model import BasicFCNConfig, BasicCNNConfig
import tyro
from typing import Union, Literal
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



@dataclass
class Config:
    encoder: BasicCNNConfig
    ae_decoder: BasicCNNConfig
    scn_model: BasicFCNConfig
    noise_input_std: float
    equivariant_features: Literal['image', 'random_equivariant', 'random_equivariant_downsample14', 'dino_features', 'dino_features_featup', 'dino_features_image_cat']
    noise_frozen_features_std: float = 0.0
    random_equivariant_feature_dim: int = 100
    datadir: str = '~/data/tartanair_sample/amusementpark/P008'
    outdir = 'outputs/test'
    log_metrics_every: int = 100
    log_images_every: int = 1000
    update_viser_every: int = 50
    save_checkpoint_every: int = 20000
    save_latest_checkpoint_every: int = 500
    seed: int = 43
    port: int = 8080
    loss_ace_weight: float = 1.0
    loss_ae_weight: float = 10.0
    extra_mask_penalty: float = 1.0
    downsample_factor: int = 1
    ace_loss_type: Literal['arccos', 'dotproduct'] = 'arccos'
    num_iterations: int = 50000
    num_frames: int = 10
    use_confidence_maps: bool = False # Flag that enables confidence maps
    video_stride: int = 10
    residual_feature_update: bool = False # If true, the encoder predicts a residual to add to the input features


def nn_randomizer_fn(pts, out_dim, h_dim=128, layers=2, nonlinearity='relu', scale_feats=1.0):
    nonlinearity_fn = None
    if nonlinearity == 'relu':
        nonlinearity_fn = torch.relu
    elif nonlinearity == 'sigmoid':
        nonlinearity_fn = torch.sigmoid
    elif nonlinearity == 'tanh':
        nonlinearity_fn = torch.tanh
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
    pts = pts[..., None]
    if layers == 1:
        return (torch.randn(out_dim, 3, device=pts.device) @ pts).squeeze(-1) * scale_feats
    else:
        x = torch.randn(h_dim, 3, device=pts.device) @ pts * scale_feats
        x = nonlinearity_fn(x)
        for _ in range(layers-2):
            x = torch.randn(h_dim, h_dim, device=pts.device) @ x * scale_feats
            x = nonlinearity_fn(x)
        x = torch.randn(out_dim, h_dim, device=pts.device) @ x * scale_feats
        return x.squeeze(-1)


def point_processing(pred_points, fs, Ts):
    # pred_points: (B, H, W, 3)
    # Ps: (B, 3, 4)

    B, H, W, _ = pred_points.shape
    pred_points = torch.cat([pred_points, torch.ones_like(pred_points[..., 0:1])], dim=-1) # (B, H, W, 4)

    xy = np.mgrid[:W,:H] + .5 # 2 x W x H
    xy = xy.transpose(2, 1, 0) # H x W x 2
    xy = torch.from_numpy(xy).float().to(pred_points.device)

    xy_homogeneous = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    y = xy_homogeneous - torch.tensor([W/2.0, H/2.0, 0.0], device=pred_points.device)

    xyz_camera_coords = torch.einsum('...ij,...HWj->...HWi', Ts, pred_points) # (B, H, W, 3)
    K_norm = torch.diag_embed(torch.stack([fs, fs, torch.ones_like(fs)], dim=-1)) # (B, 3, 3)
    y_hat = torch.einsum('...ij,...HWj->...HWi', K_norm, xyz_camera_coords) # (B, H, W, 3)

    # This normalizer can change in scale, though it does set the relative importance of getting height vs width correct by
    #   changing the aspect ratio of this normalized camera ray bundle, for which we maximize cosine similarity between rays...
    #   sorta changes the pixels from squares to rectangles, though the minimizer stays the same
    # TODO: We know this doesn't change the minimizer, but this might effect the level sets of the loss landscape;
    #   is there an optimal normalizer to make the gradients super well behaved and give nicer convergence dynamics?
    normalizer = torch.tensor([W, H, 1.0], device=pred_points.device)

    y = y / normalizer
    y_hat = y_hat / normalizer
    y = y / (torch.linalg.norm(y, dim=-1, keepdim=True) + EPS)
    y_hat = y_hat / (torch.linalg.norm(y_hat, dim=-1, keepdim=True) + EPS)

    return y, y_hat


def Ts_pred(poses):
    Bs = poses.shape[:-1]
    poses = poses.reshape(-1, 7)
    Rs = kornia.geometry.conversions.quaternion_to_rotation_matrix(poses[:, :4] / torch.linalg.norm(poses[:, :4], dim=-1, keepdim=True)) # (B, 3, 3)
    Ts = torch.cat([Rs, poses[:, 4:].unsqueeze(-1)], dim=-1) # (B, 3, 4)
    Ts = Ts.reshape(*Bs, 3, 4)
    return Ts


def process_fs(fs, widths):
    fs = (torch.nn.functional.softplus(fs, beta=7, threshold=10) + EPS) * widths
    return fs



if __name__ == '__main__':
    args = tyro.cli(Config)
    print("Loading Following Args:")
    print(args)
    print("-----\n")
    datadir = os.path.expanduser(args.datadir)
    outdir = args.outdir
    log_metrics_every = args.log_metrics_every
    log_images_every = args.log_images_every
    update_viser_every = args.update_viser_every
    save_checkpoint_every = args.save_checkpoint_every
    save_latest_checkpoint_every = args.save_latest_checkpoint_every
    pool_size = args.downsample_factor


    # If exists, add a number behind string to make the duplicate
    i = 0
    while os.path.exists(outdir + str(i)):
        i += 1
    outdir = outdir + str(i)
    os.makedirs(outdir, exist_ok=True)

    writer = SummaryWriter(outdir)
    hparams = asdict(args)

    def flatten_dict(d, prepend_str=''):
        flat_dict = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested_flat = flatten_dict(v, prepend_str=prepend_str + k + '.')
                flat_dict.update(nested_flat)
            else:
                flat_dict[prepend_str + k] = v
        return flat_dict


    writer.add_text('hparams', str(flatten_dict(hparams)))


    torch.manual_seed(args.seed)
    EPS=1e-6
    device=torch.device('cuda:0')
    num_frames = args.num_frames
    idxs = [args.video_stride * i for i in range(num_frames)]



    # Getting all the G.T.
    images = [np.array(Image.open(f'{datadir}/image_left/{idx:06d}_left.png').resize((640//pool_size, 480//pool_size))) for idx in idxs]
    depths = [np.load(f'{datadir}/depth_left/{idx:06d}_left_depth.npy') for idx in idxs] # [N] H x W
    depths = [np.array(Image.fromarray(depth).resize((640//pool_size, 480//pool_size), resample=Image.Resampling.NEAREST)) for depth in depths]
    poses = np.loadtxt(f'{datadir}/pose_left.txt')

    txs, tys, tzs, qxs, qys, qzs, qws = poses[idxs].T
    ts = np.stack([txs, tys, tzs], axis=1) # (2, 3)
    qs = np.stack([qws, qxs, qys, qzs], axis=1) # (2, 4)
    c2ws = []
    for t, q in zip(ts, qs):
        c2w = np.eye(4)
        c2w[:3, :3] = tf.SO3(q).as_matrix()
        c2w[:3, 3] = t
        T = np.array([[0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1]]) # NED to OpenCV
        T = np.linalg.inv(T)
        c2w = c2w @ T # Change from NED to OpenCV
        c2ws.append(c2w)
    c2w_visers = [tf.SE3.from_matrix(c2w) for c2w in c2ws]

    fx = 320.0 / pool_size  # focal length x
    fy = 320.0 / pool_size  # focal length y
    cx = 320.0 / pool_size  # optical center x
    cy = 240.0 / pool_size  # optical center y

    fov = 90  # field of view

    width = 640 / pool_size
    height = 480 / pool_size

    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy

    xy = np.mgrid[:width,:height].transpose(2, 1, 0) #  H x W x 2
    xy_h = np.concatenate([xy, np.ones_like(xy[..., 0:1])], axis=-1) # H x W x 3

    points_all = []
    for depth, c2w in zip(depths, c2ws):
        pts = (np.linalg.inv(K) @ xy_h[..., None]).squeeze(-1) * depth[..., None]
        pts = (c2w[:3, :3] @ pts[..., None] + c2w[:3, 3:4]).squeeze(-1)
        points_all.append(pts)
    assert points_all[0].shape == (height, width, 3)



    # Plotting
    server = viser.ViserServer(port=args.port)

    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    for idx, points, image, c2w_viser in zip(idxs, points_all, images, c2w_visers):
        server.scene.add_point_cloud(f'gt/points_{idx}', points=points.reshape(-1, 3), colors=image.reshape(-1, 3), point_size=.01)
        server.scene.add_camera_frustum(f'gt/cam_{idx}', fov=fov * np.pi / 180, aspect=width/height, wxyz=c2w_viser.wxyz_xyz[:4], position=c2w_viser.wxyz_xyz[4:])



    # Model inputs and ground truth
    images_tensor = torch.from_numpy(np.stack(images, axis=0)).float().to(device) / 255.0 # (B, H, W, 3)
    points_torch = [torch.from_numpy(pts).float() for pts in points_all]
    points_torch = torch.stack(points_torch, dim=0).to(device) # (B, H, W, 3)


    # Featurization
    equivariant_features = None
    if args.equivariant_features == 'image':
        equivariant_features = images_tensor
    elif args.equivariant_features in ['random_equivariant', 'random_equivariant_downsample14']:
        equivariant_features = nn_randomizer_fn(
                (points_torch - points_torch.reshape(-1, 3).mean(dim=0)) / (points_torch.reshape(-1, 3).std(dim=0) + EPS),
                out_dim=args.random_equivariant_feature_dim,
                h_dim=32,
                layers=3,
                nonlinearity='sigmoid',
                scale_feats=2
            ) # (B, H, W, C)
        noise = torch.randn_like(equivariant_features) * equivariant_features.std() * args.noise_frozen_features_std
        equivariant_features = equivariant_features + noise
        equivariant_features = equivariant_features - equivariant_features.reshape(-1, args.random_equivariant_feature_dim).mean(dim=0, keepdim=True)
        equivariant_features = equivariant_features / (equivariant_features.reshape(-1, args.random_equivariant_feature_dim).std(dim=0, keepdim=True) + EPS)
        if args.equivariant_features == 'random_equivariant_downsample14':
            equivariant_features = torchvision.transforms.functional.resize(
                equivariant_features.permute(0, 3, 1, 2), # BxCxHxW
                (equivariant_features.shape[1] // 14, equivariant_features.shape[2] // 14)
            )
            equivariant_features = [
                torch.nn.functional.interpolate(
                    feat.unsqueeze(0),
                    size=(images_tensor.shape[1], images_tensor.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for feat in equivariant_features
            ]
            equivariant_features = torch.stack(equivariant_features, dim=0)  # B x C x H x W
            equivariant_features = equivariant_features.permute(0, 2, 3, 1)

    elif args.equivariant_features in ['dino_features', 'dino_features_image_cat']:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        model = model.to(device)
        img_tensors = images_tensor.permute(0, 3, 1, 2)  # BxCxHxW
        assert (
            img_tensors.max() <= 1 and img_tensors.min() >= 0
        ), f"Images should be normalized between 0 and 1, got min {img_tensors.min()} and max {img_tensors.max()}"
        img_tensors = torchvision.transforms.functional.normalize(
            img_tensors, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
        img_tensors = torchvision.transforms.functional.resize(img_tensors, ((img_tensors.shape[2] // 14) * 14, (img_tensors.shape[3] // 14) * 14))
        assert (
            img_tensors[0].shape[0] == 3
        ), "Images should be RGB, BxCxHxW (can be a list instead of a batch tensor)"
        out_shape = (
            img_tensors.shape[0],
            img_tensors.shape[2] // 14,
            img_tensors.shape[3] // 14,
            -1,
        )
        with torch.no_grad():
            all_out = []
            for t in img_tensors:
                all_out.append(model.forward_features(t[None].to(device))["x_norm_patchtokens"])
            all_out = torch.cat(all_out, dim=0)
            all_out_pixel_aligned = torch.reshape(
                all_out, out_shape
            ) # B x H' x W' x C
            all_out_pixel_aligned = all_out_pixel_aligned.permute(0, 3, 1, 2)
            all_out_pixel_aligned = [
                torch.nn.functional.interpolate(
                    feat.unsqueeze(0),
                    size=(images_tensor.shape[1], images_tensor.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for feat in all_out_pixel_aligned
            ]
        all_out_pixel_aligned = torch.stack(all_out_pixel_aligned, dim=0)  # B x C x H x W
        equivariant_features = all_out_pixel_aligned.permute(0, 2, 3, 1)
        del model
        del all_out_pixel_aligned
        del all_out
        del img_tensors
        if args.equivariant_features == 'dino_features_image_cat':
            equivariant_features = torch.cat([equivariant_features, images_tensor], dim=-1)
    else:
        raise ValueError(f"Unknown equivariant_features type: {args.equivariant_features}")
    B, H, W, feat_dim = equivariant_features.shape
    print(f"Features of shape {equivariant_features.shape}")


    # Variables of optimization
    _fs = torch.rand(B, device=device) * 3 # (B,)
    print(f"Initial fs: {process_fs(_fs, W)}")
    _fs.requires_grad = True
    confidence_maps = torch.ones(B, H, W, 1, device=device) # (B, H, W, 1)
    confidence_maps.requires_grad = True
    poses = torch.randn(B, 7).float().to(device) # (B, 7)   (2, 7) 4 for quaternion, 3 for translation
    # # GT initialization
    # poses = c2w_visers[0].inverse()
    # poses = torch.cat([torch.tensor(poses.wxyz_xyz[:4]), torch.tensor(poses.wxyz_xyz[4:])]).reshape(1, 7).float()
    poses.requires_grad = True


    # ACE Training
    input_features = equivariant_features
    encoder = args.encoder.setup()
    decoder = args.ae_decoder.setup()
    scn_model = args.scn_model.setup()
    scn_model=scn_model.to(device)
    encoder=encoder.to(device)
    decoder=decoder.to(device)
    param_groups = [
        {'params': scn_model.parameters(), 'lr': 1e-2},
        {'params': encoder.parameters(), 'lr': 1e-4},
        {'params': decoder.parameters(), 'lr': 1e-4},
        {'params': [_fs], 'lr': 1e-2},
        {'params': [poses], 'lr': 1e-2},
        {'params': [confidence_maps], 'lr': 1e-2},
    ]
    assert args.num_iterations > 0
    optimizer = torch.optim.Adam(param_groups)
    pbar = tqdm(range(1,args.num_iterations+1))
    logs = {}


    for epoch in pbar:
        optimizer.zero_grad()
        Ts = Ts_pred(poses) # (B, 3, 4)
        fs = process_fs(_fs, W) # (B,)
        M = confidence_maps ** 2
        # sample_mask = torch.randn(B, H, W, 1, device=device) < .2

        #refined_features = equivariant_features
        noise = torch.randn_like(input_features) * input_features.std()
        refined_features = input_features + (noise * args.noise_input_std)
        if args.residual_feature_update:
            refined_features = refined_features + encoder(refined_features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            refined_features = encoder(refined_features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # (B, H, W, C)
        preds = scn_model(refined_features) # (B, H, W, 3)
        y, y_hat = point_processing(preds, fs, Ts)

        loss_ace = y * y_hat
        loss_ace = torch.clip(loss_ace.sum(-1, keepdim=True), min=EPS-1, max=1-EPS)
        if args.ace_loss_type == 'dotproduct':
            loss_ace = 1 - loss_ace
        elif args.ace_loss_type == 'arccos':
            loss_ace = torch.arccos(loss_ace) # Can taylor expand 1-cos(x) = x^2/2 + O(x^4), so arccos(cos(x)) is like an L1 loss and dot product is L2 loss
        else:
            raise ValueError(f"Unknown ace_loss_type: {args.ace_loss_type}")
        loss_ace_raw = loss_ace
        confidence_weighted_loss_ace = loss_ace / (M + EPS)
        if args.use_confidence_maps:
            loss_ace = torch.log((M * args.extra_mask_penalty) + confidence_weighted_loss_ace).mean()
        else:
            loss_ace = loss_ace.mean()

        decoded_im = decoder(refined_features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # (B, H, W, 3)
        loss_ae = ((decoded_im - images_tensor) ** 2).mean(dim=-1, keepdim=True)
        assert loss_ae.shape == (B, H, W, 1)
        loss_ae = loss_ae.mean()

        loss = 0
        loss = loss + (loss_ace * args.loss_ace_weight)
        loss = loss + (loss_ae * args.loss_ae_weight)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if epoch % update_viser_every == 0:
                shift_preds = preds.reshape(-1, 3).median(dim=0).values
                scale= (preds.reshape(-1, 3) - shift_preds).abs().median().item() * .3 + 1e-6
                shift_preds = shift_preds.detach().cpu().numpy()
                for i, idx in enumerate(idxs):
                    pose = poses[i].reshape(-1, 7)
                    R = kornia.geometry.conversions.quaternion_to_rotation_matrix(pose[:, :4] / torch.linalg.norm(pose[:, :4], dim=-1, keepdim=True))[0] # (B, 3, 3)
                    w2c_viser = np.eye(4)
                    w2c_viser[:3,:3] = R.cpu().detach().numpy()
                    w2c_viser[:3, 3] = pose[0, 4:].cpu().detach().numpy()
                    c2w_viser = tf.SE3.from_matrix(np.linalg.inv(w2c_viser))
                    im = M[i].detach().cpu().numpy().squeeze()
                    im = np.stack([im, im, im], axis=-1)
                    im = (im - im.min()) / (im.max() - im.min() + 1e-6)
                    server.scene.add_camera_frustum(f'pred/cam_{idx}_pred', fov= 90 * np.pi / 180, aspect=W/H, wxyz=c2w_viser.wxyz_xyz[:4], position=(c2w_viser.wxyz_xyz[4:] - shift_preds) / scale, color=(255, 0, 0), image=im)
                    points = (preds[i].detach().cpu().numpy().reshape(-1, 3) - shift_preds) / scale
                    colors = images[i].reshape(-1, 3)
                    assert points.shape == colors.shape, f"{points.shape}, {colors.shape}"
                    server.scene.add_point_cloud(f'pred/points_{idx}_pred', points=points, colors=colors, point_size=.01)

            if epoch % log_metrics_every == 0:
                # Log scalars
                mask_z_positive = y_hat[..., 2] > 0
                mask_z_positive = mask_z_positive[..., None]
                writer.add_scalar('Loss/total', loss.item(), epoch)
                writer.add_scalar('Loss/ace', loss_ace.item(), epoch)
                writer.add_scalar('Loss/ae', loss_ae.item(), epoch)
                writer.add_scalar('Metrics/percent_z_correct', mask_z_positive.float().mean().item(), epoch)
                writer.add_scalar('Metrics/mean_f', fs.mean().item(), epoch)
                writer.add_scalar('Metrics/min_f', fs.min().item(), epoch)
                writer.add_scalar('Metrics/max_f', fs.max().item(), epoch)
                writer.add_scalar('Metrics/Ms_mean', M.mean().item(), epoch)
                writer.add_scalar('Metrics/decoded_im_min', decoded_im.min().item(), epoch)
                writer.add_scalar('Metrics/decoded_im_max', decoded_im.max().item(), epoch)
                writer.add_scalar('Metrics/decoded_im_std', decoded_im.std().item(), epoch)
                writer.add_scalar('Metrics/decoded_im_median', decoded_im.median().item(), epoch)
                writer.add_scalar('Metrics/decoded_im_mean', decoded_im.mean().item(), epoch)
                writer.add_scalar('Metrics/confidence_weighted_loss_ace_mean', confidence_weighted_loss_ace.mean().item(), epoch)
                writer.add_scalar('Metrics/loss_ace_raw_mean', loss_ace_raw.mean().item(), epoch)
                loss_ace_raw_sorted = torch.sort(loss_ace_raw.detach().cpu().flatten()).values
                num_points = len(loss_ace_raw_sorted)
                writer.add_scalar('Quantiles/loss_ace_raw_q.9', loss_ace_raw_sorted[int(num_points * .9)].item(), epoch)
                writer.add_scalar('Quantiles/loss_ace_raw_q.95', loss_ace_raw_sorted[int(num_points * .95)].item(), epoch)
                writer.add_scalar('Quantiles/loss_ace_raw_q.99', loss_ace_raw_sorted[int(num_points * .99)].item(), epoch)
                writer.add_scalar('Quantiles/loss_ace_raw_q.999', loss_ace_raw_sorted[int(num_points * .999)].item(), epoch)
                writer.add_scalar('Quantiles/loss_ace_raw_q.9999', loss_ace_raw_sorted[int(num_points * .9999)].item(), epoch)
                writer.add_scalar('Quantiles/loss_ace_raw_max', loss_ace_raw_sorted[-1].item(), epoch)


            if epoch % log_images_every == 0:
                # Compute PCA of features
                refined_feat_dim = refined_features.shape[-1]
                feats = refined_features.detach().reshape(-1, refined_feat_dim)
                feats_mean = feats.mean(dim=0, keepdim=False)
                feats = feats - feats_mean
                cov = feats.T @ feats / (feats.shape[0] - 1)
                U, S, Vh = torch.linalg.svd(cov)
                all_feats2d = feats @ U[:, :3]
                cmin = all_feats2d.min(dim=0).values
                cmax = all_feats2d.max(dim=0).values

                for i, idx in enumerate(idxs):
                    # Feature PCA visualization
                    feats = refined_features[i] - feats_mean
                    feats_2d = feats @ U[:, :3]
                    feats_2d = (feats_2d - cmin[None, None, :]) / (cmax[None, None, :] - cmin[None, None, :] + 1e-6)
                    writer.add_image(f'Features_PCA/frame_{idx:06d}', feats_2d.permute(2, 0, 1), epoch)

                    # Confidence maps
                    confidence = M[i].detach().cpu().numpy().squeeze()
                    confidence_norm = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-6)
                    writer.add_image(f'Confidence/frame_{idx:06d}', torch.from_numpy(confidence_norm).unsqueeze(0), epoch)

                    # Decoded images
                    _decoded_im = torch.clamp(decoded_im[i], 0.0, 1.0)
                    writer.add_image(f'Decoded/frame_{idx:06d}', _decoded_im.permute(2, 0, 1), epoch)

                    # Ground truth images for comparison
                    writer.add_image(f'GT/frame_{idx:06d}', images_tensor[i].permute(2, 0, 1), epoch)


                loss_ace_raw_sorted = torch.sort(loss_ace_raw.detach().cpu().flatten()).values
                num_points = len(loss_ace_raw_sorted)
                writer.add_histogram('Histograms/loss_ace', loss_ace_raw_sorted.numpy(), epoch, bins='auto')
                writer.add_histogram('Histograms/loss_ace_q.9+', loss_ace_raw_sorted[int(num_points * .9):].numpy(), epoch, bins='auto')
                writer.add_histogram('Histograms/loss_ace_q.95+', loss_ace_raw_sorted[int(num_points * .95):].numpy(), epoch, bins='auto')
                writer.add_histogram('Histograms/loss_ace_q.99+', loss_ace_raw_sorted[int(num_points * .99):].numpy(), epoch, bins='auto')
                # Make matplotlib figures instead for each of these
                for q in [.9, .95, .99, .999, .9999]:
                    fig = plt.figure()
                    plt.hist(loss_ace_raw_sorted[int(num_points * q):].numpy(), bins=100)
                    plt.title(f'Histogram of loss_ace values above quantile {q}')
                    plt.xlabel('loss_ace value')
                    plt.ylabel('Frequency')
                    writer.add_figure(f'Histograms/loss_ace_q{q:4f}/', fig, epoch)
                    plt.close(fig)


            if epoch % save_latest_checkpoint_every == 0:
                # Save checkpoint
                checkpoint = {
                    'scn_model_state_dict': scn_model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    '_fs': _fs,
                    'poses': poses,
                    'confidence_maps': confidence_maps,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'input_features': input_features,
                    'epoch': epoch,
                    'args': args,
                }
                torch.save(checkpoint, os.path.join(outdir, f'latest_checkpoint.pt'))

            if epoch % save_checkpoint_every == 0 and epoch > 0 and save_checkpoint_every > 0:
                # Save checkpoint
                checkpoint = {
                    'scn_model_state_dict': scn_model.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    '_fs': _fs,
                    'poses': poses,
                    'confidence_maps': confidence_maps,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'input_features': input_features,
                    'epoch': epoch,
                    'args': args,
                }
                torch.save(checkpoint, os.path.join(outdir, f'checkpoint_epoch_{epoch:06d}.pt'))

            if epoch % 10 == 0:
                # Compute mask_z_positive for progress bar
                with torch.no_grad():
                    mask_z_positive = y_hat[..., 2] > 0
                pbar.set_description(
                    f'loss: {loss.item():.4f} | loss_ace: {loss_ace.item():.4f} | loss_ae: {loss_ae.item():.4f} | percent_z_correct: {mask_z_positive.float().mean().item():.4f}'
                )


