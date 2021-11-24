import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def load_dtu(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    print('Load data: Begin')
    device = torch.device('cuda')
    # conf = conf

    data_dir = basedir
    render_cameras_name = "cameras_sphere.npz"
    object_cameras_name = "cameras_sphere.npz"

    camera_dict = np.load(os.path.join(data_dir, render_cameras_name))
    camera_dict = camera_dict
    images_lis = sorted(glob(os.path.join(data_dir, 'images/*.png')))
    n_images = len(images_lis)
    images_np = np.stack([cv.imread(im_name) for im_name in images_lis]) / 256.0
    # masks_lis = sorted(glob(os.path.join(data_dir, 'mask/*.png')))
    # masks_np = np.stack([cv.imread(im_name) for im_name in masks_lis]) / 256.0

    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    scale_mats_np = []

    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []

    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    images = torch.from_numpy(images_np.astype(np.float32)).numpy()  # [n_images, H, W, 3]
    # masks  = torch.from_numpy(masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
    intrinsics_all = torch.stack(intrinsics_all)#.to(device)   # [n_images, 4, 4]
    # intrinsics_all_inv = torch.inverse(intrinsics_all)  # [n_images, 4, 4]
    focal = intrinsics_all[0][0, 0]
    pose_all = torch.stack(pose_all).numpy()#.to(device)  # [n_images, 4, 4]
    H, W = images.shape[1], images.shape[2]
    image_pixels = H * W
    hwf = [H, W, focal]

    # object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
    # object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
    # Object scale mat: region of interest to **extract mesh**
    # object_scale_mat = np.load(os.path.join(data_dir, object_cameras_name))['scale_mat_0']
    # object_bbox_min = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
    # object_bbox_max = np.linalg.inv(scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
    # object_bbox_min = object_bbox_min[:3, 0]
    # object_bbox_max = object_bbox_max[:3, 0]

    poses = recenter_poses(pose_all)
    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = 0.1, 5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    if path_zflat:
        zloc = -close_depth * .1
        c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
        rads[2] = 0.
        N_rots = 1
        N_views/=2

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    return images, poses, render_poses, i_test, hwf

# def gen_rays_at(img_idx, resolution_level=1):
#     """
#     Generate rays at world space from one camera.
#     """
#     l = resolution_level
#     tx = torch.linspace(0, W - 1, W // l)
#     ty = torch.linspace(0, H - 1, H // l)
#     pixels_x, pixels_y = torch.meshgrid(tx, ty)
#     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
#     p = torch.matmul(intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
#     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
#     rays_v = torch.matmul(pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
#     rays_o = pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
#     return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

# def gen_random_rays_at(img_idx, batch_size):
#     """
#     Generate random rays at world space from one camera.
#     """
#     pixels_x = torch.randint(low=0, high=W, size=[batch_size])
#     pixels_y = torch.randint(low=0, high=H, size=[batch_size])
#     color = images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
#     mask = masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
#     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
#     p = torch.matmul(intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
#     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
#     rays_v = torch.matmul(pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
#     rays_o = pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
#     return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

# def gen_rays_between(idx_0, idx_1, ratio, resolution_level=1):
#     """
#     Interpolate pose between two cameras.
#     """
#     l = resolution_level
#     tx = torch.linspace(0, W - 1, W // l)
#     ty = torch.linspace(0, H - 1, H // l)
#     pixels_x, pixels_y = torch.meshgrid(tx, ty)
#     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
#     p = torch.matmul(intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
#     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
#     trans = pose_all[idx_0, :3, 3] * (1.0 - ratio) + pose_all[idx_1, :3, 3] * ratio
#     pose_0 = pose_all[idx_0].detach().cpu().numpy()
#     pose_1 = pose_all[idx_1].detach().cpu().numpy()
#     pose_0 = np.linalg.inv(pose_0)
#     pose_1 = np.linalg.inv(pose_1)
#     rot_0 = pose_0[:3, :3]
#     rot_1 = pose_1[:3, :3]
#     rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
#     key_times = [0, 1]
#     slerp = Slerp(key_times, rots)
#     rot = slerp(ratio)
#     pose = np.diag([1.0, 1.0, 1.0, 1.0])
#     pose = pose.astype(np.float32)
#     pose[:3, :3] = rot.as_matrix()
#     pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
#     pose = np.linalg.inv(pose)
#     rot = torch.from_numpy(pose[:3, :3]).cuda()
#     trans = torch.from_numpy(pose[:3, 3]).cuda()
#     rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
#     rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
#     return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

# def near_far_from_sphere(rays_o, rays_d):
#     a = torch.sum(rays_d**2, dim=-1, keepdim=True)
#     b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
#     mid = 0.5 * (-b) / a
#     near = mid - 1.0
#     far = mid + 1.0
#     return near, far

# def image_at(idx, resolution_level):
#     img = cv.imread(images_lis[idx])
#     return (cv.resize(img, (W // resolution_level, H // resolution_level))).clip(0, 255)


## from load_llff
def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    
def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses