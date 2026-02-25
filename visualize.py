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
import time
from toy_test import Config

argparser = argparse.ArgumentParser()
argparser.add_argument('--datadir', type=str, default='~/data/tartanair_sample/amusementpark/P008', help='Path to Tartanair dataset directory.')
argparser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint.', required=True)
args = argparser.parse_args()
datadir = os.path.expanduser(args.datadir)
checkpoint_path = os.path.expanduser(args.checkpoint_path)



device=torch.device('cuda:0')
checkpoint = torch.load(checkpoint_path)
encoder = checkpoint['args'].encoder.setup().to(device)
decoder = checkpoint['args'].ae_decoder.setup().to(device)
scn_model = checkpoint['args'].scn_model.setup().to(device)
scn_model.load_state_dict(checkpoint['scn_model_state_dict'])
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
scn_model.eval()
encoder.eval()
decoder.eval()
_fs = checkpoint['_fs'].to(device)
poses = checkpoint['poses'].to(device)
confidence_maps = checkpoint['confidence_maps'].to(device)
input_features = checkpoint['input_features'].to(device)
B, H, W, _ = input_features.shape
print(f"Loaded checkpoint from {checkpoint_path} at epoch {checkpoint['epoch']}")



torch.manual_seed(43)
EPS=1e-6
num_frames = 10
idxs = [10 * i for i in range(num_frames)]


# Getting all the G.T., resize to new HW
images = [np.array(Image.open(f'{datadir}/image_left/{idx:06d}_left.png')) for idx in idxs]
images = [np.array(Image.fromarray(image).resize((W, H))) / 255.0 for image in images]
depths = [np.load(f'{datadir}/depth_left/{idx:06d}_left_depth.npy') for idx in idxs]
depths = [np.array(Image.fromarray(depth).resize((W, H), resample=Image.NEAREST)) for depth in depths]
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

downsample_x = W / 640
downsample_y = H / 480
fx = 320.0 * downsample_x # focal length x
fy = 320.0 * downsample_y  # focal length y
cx = W / 2  # optical center x
cy = H / 2  # optical center y

fov = 90  # field of view

width = W
height = H

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
server = viser.ViserServer(port=8080)

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


class FCN(nn.Module):
    def __init__(self, in_dim, h_dim=128, n_layers=2, use_bias=True, nonlinearity='relu'):
        super().__init__()
        assert n_layers >= 1
        self.nonlinearity = None
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'swish':
            self.nonlinearity = nn.SiLU()
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        if n_layers == 1:
            self.predict = nn.Linear(in_dim, 3, bias=use_bias)
        if n_layers > 1:
            self.predict = [nn.Linear(in_dim, h_dim, bias=use_bias), ]
            self.predict = self.predict + ([nn.Linear(h_dim, h_dim, bias=use_bias)]*(n_layers-2))
            self.predict.append(nn.Linear(h_dim, 3, bias=use_bias))
            self.predict = nn.ModuleList(self.predict)

    def forward(self, x):
        x = self.predict[0](x)
        x = self.nonlinearity(x)
        for layer in self.predict[1:-1]:
            x = self.nonlinearity(layer(x))
        x = self.predict[-1](x)
        return x


class CNN(nn.Module):
    def __init__(self, in_dim, h_dim=128, n_layers=3, kernel_size=3, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        layers = []
        layers.append(nn.Conv2d(in_dim, h_dim, kernel_size=kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Conv2d(h_dim, h_dim, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(h_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)





# ACE Training

# if epoch % flush_latest_checkpoint_every == 0:
#     # Save checkpoint
#     checkpoint = {
#         'scn_model_state_dict': scn_model.state_dict(),
#         'encoder_state_dict': encoder.state_dict(),
#         'decoder_state_dict': decoder.state_dict(),
#         '_fs': _fs,
#         'poses': poses,
#         'confidence_maps': confidence_maps,
#         'optimizer_state_dict': optimizer.state_dict(),
#         'input_features': input_features,
#         'epoch': epoch,
#     }
#     torch.save(checkpoint, os.path.join(outdir, f'latest_checkpoint.pt'))
_fs = checkpoint['_fs'].to(device)
poses = checkpoint['poses'].to(device)
confidence_maps = checkpoint['confidence_maps'].to(device)
input_features = checkpoint['input_features'].to(device)
B, H, W, _ = input_features.shape
print(f"Loaded checkpoint from {checkpoint_path} at epoch {checkpoint['epoch']}")



Ts = Ts_pred(poses) # (B, 3, 4)
fs = process_fs(_fs, W) # (B,)
M = confidence_maps ** 2
# sample_mask = torch.randn(B, H, W, 1, device=device) < .2

_input_features = input_features
refined_features = encoder(_input_features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # (B, H, W, C)
preds = scn_model(refined_features) # (B, H, W, 3)
y, y_hat = point_processing(preds, fs, Ts)

decoded_im = decoder(refined_features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # (B, H, W, 3)


with torch.no_grad():
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

while True:
    time.sleep(1)
