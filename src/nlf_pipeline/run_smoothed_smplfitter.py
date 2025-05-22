import argparse
import itertools
import os.path as osp
from typing import Optional

import cameravision
import more_itertools
import framepump
import numpy as np
import rlemasklib
import scipy.optimize
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import smplfitter.np
import smplfitter.pt
import torch
import torch.nn.functional as F
from bodycompress import BodyDecompressor
from nlf_pipeline.util import smpl_mask
from nlf_pipeline.util.paths import DATA_ROOT, INFERENCE_ROOT
from simplepyutils import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=str)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--fov", type=float, default=55)
    parser.add_argument('--fill-gaps', action=spu_argparse.BoolAction)
    parser.add_argument('--skip-existing', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--fps-factor', type=int, default=1)
    spu.initialize(parser)
    torch.set_num_threads(12)


def main():
    initialize()
    body_model_name = 'smpl'
    fov = FLAGS.fov

    pred_path = f'{INFERENCE_ROOT}/preds_np/{FLAGS.video_id}{FLAGS.suffix}.xz'
    mask_path = f'{INFERENCE_ROOT}/masks/{FLAGS.video_id}_masks.pkl'
    camera_path = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}.pkl'

    dfps_str = f'_fps{FLAGS.fps_factor}' if FLAGS.fps_factor != 1 else ''
    video_path = f'{INFERENCE_ROOT}/videos_in/{FLAGS.video_id}.mp4'

    fit_path = (
        f'{INFERENCE_ROOT}/smooth_fits/{FLAGS.video_id}{FLAGS.suffix}_smoothfits{dfps_str}.pkl'
    )
    camera_path_new = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}{dfps_str}.pkl'
    ground_path = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}{FLAGS.suffix}_ground.pkl'

    if FLAGS.skip_existing and osp.exists(fit_path):
        return

    body_model = smplfitter.np.get_cached_body_model(body_model_name)

    video = framepump.VideoFrames(video_path)
    masks = spu.load_pickle(mask_path)

    if osp.isfile(camera_path):
        cameras_display = spu.load_pickle(camera_path)
    else:
        cameras_display = [cameravision.Camera.from_fov(fov, video.imshape)] * len(masks)

    n_verts = body_model.num_vertices
    n_joints = body_model.num_joints
    tracks = build_tracks_via_masks(pred_path, cameras_display, video.imshape, masks, n_verts, n_joints)
    cam_points = np.stack([c.t for c in cameras_display], axis=0)

    fit_tracks = [
        smooth_person_track(
            body_model_name,
            track[:, :n_verts, :3] / 1000,
            track[:, n_verts : n_verts + n_joints, :3] / 1000,
            track[:, :n_verts, 3],
            track[:, n_verts : n_verts + n_joints, 3],
            cam_points / 1000,
            video.fps,
            n_verts,
            n_joints,
        )
        for track in spu.progressbar(tracks, desc='Fitting tracks')
    ]

    valid_fits_per_frame = collect_valid_fits_per_frame(fit_tracks)
    spu.dump_pickle(valid_fits_per_frame, fit_path)

    ground_height, new_up = fit_ground_plane(
        body_model, valid_fits_per_frame, cameras_display[0].world_up
    )
    spu.dump_pickle(dict(ground_height=ground_height, world_up=new_up), ground_path)

    if FLAGS.fps_factor != 1:

        def interp_fn(cams):
            mids = [interp_cam(cam1, cam2, 0.5) for cam1, cam2 in itertools.pairwise(cams)]
            mids.append(cams[-1])
            return list(more_itertools.interleave(cams, mids))

        cameras_display = interp_fn(cameras_display)
        spu.dump_pickle(cameras_display, camera_path_new)


def collect_valid_fits_per_frame(fit_tracks):
    fits_per_frame = []
    pose_rotvecs_shape = fit_tracks[0]['pose_rotvecs'].shape[1:]
    shape_betas_shape = fit_tracks[0]['shape_betas'].shape[1:]
    trans_shape = fit_tracks[0]['trans'].shape[1:]

    for i_frame in range(fit_tracks[0]['pose_rotvecs'].shape[0]):
        pose_rotvecs = []
        shape_betas = []
        trans = []
        for track in fit_tracks:
            if np.all(np.isfinite(track['pose_rotvecs'][i_frame])):
                pose_rotvecs.append(track['pose_rotvecs'][i_frame])
                shape_betas.append(track['shape_betas'][i_frame])
                trans.append(track['trans'][i_frame])
        fits_per_frame.append(
            dict(
                pose_rotvecs=stack(pose_rotvecs, pose_rotvecs_shape),
                shape_betas=stack(shape_betas, shape_betas_shape),
                trans=stack(trans, trans_shape),
            )
        )

    return fits_per_frame


def interp_cam(cam1, cam2, t):
    f1 = np.array([cam1.intrinsic_matrix[0, 0], cam1.intrinsic_matrix[1, 1]])
    f2 = np.array([cam2.intrinsic_matrix[0, 0], cam2.intrinsic_matrix[1, 1]])
    c1 = cam1.intrinsic_matrix[:2, 2]
    c2 = cam2.intrinsic_matrix[:2, 2]
    f = np.exp(np.log(f1) + t * (np.log(f2) - np.log(f1)))
    c = np.exp(np.log(c1) + t * (np.log(c2) - np.log(c1)))
    intr = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]])
    dist1 = cam1.get_distortion_coeffs()
    dist2 = cam2.get_distortion_coeffs()
    dist = dist1 + t * (dist2 - dist1)
    optical_center = cam1.t + t * (cam2.t - cam1.t)
    rot = project_to_SO3(cam1.R + t * (cam2.R - cam1.R))

    return cameravision.Camera(
        rot_world_to_cam=rot,
        intrinsic_matrix=intr,
        optical_center=optical_center,
        distortion_coeffs=dist,
        world_up=cam1.world_up,
    )


def project_to_SO3(A):
    U, _, Vh = np.linalg.svd(A)
    T = U @ Vh
    has_reflection = (np.linalg.det(T) < 0)[..., np.newaxis, np.newaxis]
    T_mirror = T - 2 * U[..., -1:] @ Vh[..., -1:, :]
    return np.where(has_reflection, T_mirror, T)


def stack(a, element_shape):
    if len(a) == 0:
        return np.zeros((0,) + element_shape, np.float32)
    return np.stack(a, axis=0)


def conv1d_indep(a: torch.Tensor, kernel: torch.Tensor):
    return F.conv1d(a, kernel[np.newaxis, np.newaxis], padding='same')


def moving_mean(a: torch.Tensor, weights: torch.Tensor, kernel: torch.Tensor):
    finite = torch.all(
        torch.logical_and(torch.isfinite(a), torch.isfinite(weights)), dim=1, keepdim=True
    )
    a = torch.where(finite, a, torch.zeros_like(a))
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    return torch.nan_to_num(conv1d_indep(weights * a, kernel) / conv1d_indep(weights, kernel))


def moving_mean_dim(x: torch.Tensor, weights: torch.Tensor, kernel: torch.Tensor, dim: int = -2):
    weights = torch.broadcast_to(weights, x.shape)
    x = x.movedim(dim, -1)
    weights = weights.movedim(dim, -1)
    mean = moving_mean(
        x.reshape(-1, 1, x.shape[-1]), weights.reshape(-1, 1, weights.shape[-1]), kernel
    ).reshape(x.shape)
    return mean.movedim(-1, dim)


@torch.jit.script
def robust_geometric_filter(
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    kernel: torch.Tensor,
    dim: int = -2,
    eps: float = 1e-1,
    n_iter: int = 10,
):
    w_ = torch.ones_like(x[..., :1]) if w is None else w.unsqueeze(-1)

    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    y = moving_mean_dim(x, w_, kernel, dim=dim)
    w_modified = w_
    for _ in range(n_iter):
        dist = torch.norm(x - y, dim=-1, keepdim=True)
        w_modified = w_ / (dist + eps)
        y = moving_mean_dim(x, w_modified, kernel, dim=dim)

    w2 = moving_mean_dim(w_, w_modified, kernel, dim=dim)
    return y, w2.squeeze(-1)


def robust_geometric_filter_twosided(
    x,
    w,
    kernel,
    dim_t=-3,
    dim_n=None,
    eps=1e-1,
    n_iter=10,
    split_threshold=1,
    split_indices=None,
):
    if dim_t < 0:
        dim_t = len(x.shape) + dim_t

    if split_indices is None:
        kernel_half_size = kernel.shape[0] // 2

        center_half = kernel[kernel_half_size : kernel_half_size + 1] / 2
        left_half_kernel = torch.cat(
            [kernel[:kernel_half_size], center_half, torch.zeros_like(kernel[kernel_half_size:])],
            dim=0,
        )
        right_half_kernel = torch.cat(
            [
                torch.zeros_like(kernel[:-kernel_half_size]),
                center_half,
                kernel[-kernel_half_size:],
            ],
            dim=0,
        )

        left_filtered = robust_geometric_filter(
            x, w, left_half_kernel, dim=dim_t, eps=eps, n_iter=n_iter
        )[0]
        right_filtered = robust_geometric_filter(
            x, w, right_half_kernel, dim=dim_t, eps=eps, n_iter=n_iter
        )[0]

        d = torch.norm(left_filtered - right_filtered, dim=-1)
        if dim_n is not None:
            d = d.min(dim=dim_n, keepdim=True)

        midslice = tuple(
            [slice(1, -1) if i == dim_t else slice(None) for i in range(len(d.shape))]
        )
        leftslice = tuple(
            [slice(0, -2) if i == dim_t else slice(None) for i in range(len(d.shape))]
        )
        rightslice = tuple(
            [slice(2, None) if i == dim_t else slice(None) for i in range(len(d.shape))]
        )

        tiebreaker_noise = torch.randn_like(d)
        noisy_diff = d + 1e-3 * split_threshold * tiebreaker_noise

        is_local_max = torch.logical_and(
            noisy_diff[midslice] > noisy_diff[leftslice],
            noisy_diff[midslice] > noisy_diff[rightslice],
        )
        pad = torch.zeros_like(
            torch.index_select(is_local_max, dim_t, torch.tensor([0], dtype=torch.int64))
        )
        is_local_max = torch.cat([pad, is_local_max, pad], dim=dim_t)
        is_split_point = torch.logical_and(is_local_max, d > split_threshold)

        split_indices = torch.where(is_split_point)[0] + 1
        # split_sizes = torch.cat(
        #     [
        #         split_indices[:1],
        #         split_indices[1:] - split_indices[:-1],
        #         is_split_point.shape[0] - split_indices[-1:],
        #     ]
        # )

    ragged_x = torch.tensor_split(x, split_indices, dim=dim_t)
    ragged_w = torch.tensor_split(w, split_indices, dim=dim_t)

    filtered = []
    filtered_weights = []
    for _x, _w in zip(ragged_x, ragged_w):
        filtered_x, filtered_w = robust_geometric_filter(
            _x, _w, kernel, dim=dim_t, eps=eps, n_iter=n_iter
        )
        filtered.append(filtered_x)
        filtered_weights.append(filtered_w)
    filtered = torch.cat(filtered, dim=dim_t)
    filtered_weights = torch.cat(filtered_weights, dim=dim_t)
    return filtered, filtered_weights, split_indices


def apply_nanmask(values, weights):
    if values.dim() > weights.dim():
        values, weights = apply_nanmask(values, weights.unsqueeze(-1))
        return values, weights.squeeze(-1)

    finite = torch.all(
        torch.logical_and(torch.isfinite(values), torch.isfinite(weights)), dim=-1, keepdim=True
    )

    values = torch.where(finite, values, torch.zeros_like(values))
    weights = torch.where(finite, weights, torch.zeros_like(weights))
    return values, weights


def reduce_nanmean(x, dim, keepdim=False):
    is_finite = torch.isfinite(x)
    x = torch.where(is_finite, x, torch.zeros_like(x))
    return torch.sum(x, dim=dim, keepdim=keepdim) / torch.sum(is_finite, dim=dim, keepdim=keepdim)


def smooth_person_track(
    body_model_name,
    vertices,
    joints,
    vertices_uncert,
    joints_uncert,
    cam_points,
    fps,
    n_verts,
    n_joints,
    n_subset=1024,
):
    vertices = torch.as_tensor(vertices, dtype=torch.float32)
    joints = torch.as_tensor(joints, dtype=torch.float32)
    vertices_uncert = torch.as_tensor(vertices_uncert, dtype=torch.float32)
    joints_uncert = torch.as_tensor(joints_uncert, dtype=torch.float32)
    cam_points = torch.as_tensor(cam_points, dtype=torch.float32)

    def get_weights(uncerts, exponent):
        w = uncerts**-exponent
        return w / reduce_nanmean(w, dim=(-2, -1), keepdim=True)

    joint_weights = get_weights(joints_uncert, 1.5)
    vertex_weights = get_weights(vertices_uncert, 1.5)

    vertices, vertex_weights = apply_nanmask(vertices, vertex_weights)
    joints, joint_weights = apply_nanmask(joints, joint_weights)

    if n_subset < n_verts:
        vertex_subset_np = np.load(f'{DATA_ROOT}/body_models/smpl/vertex_subset_{n_subset}.npz')[
            'i_verts'
        ]
    else:
        vertex_subset_np = np.arange(n_verts)

    vertex_subset_tup = tuple(vertex_subset_np)
    vertex_subset_pt = torch.from_numpy(vertex_subset_np)

    reg_fac = (n_subset + n_joints) / (n_verts + n_joints)

    # Obtain a scale correction estimation via shared beta estimation and beta prior
    fits_prelim = smplfitter.pt.get_cached_fit_fn(
        body_model_name=body_model_name,
        requested_keys=('vertices', 'joints', 'pose_rotvecs'),
        beta_regularizer=10 * reg_fac,
        beta_regularizer2=0.2 * reg_fac,
        scale_regularizer=1 * reg_fac,
        num_betas=10,
        vertex_subset=vertex_subset_tup,
        share_beta=True,
        final_adjust_rots=False,
        scale_target=True,
        device='cpu',
    )(
        torch.index_select(vertices, dim=1, index=vertex_subset_pt),
        joints,
        torch.index_select(vertex_weights, dim=1, index=vertex_subset_pt),
        joint_weights,
    )

    scale = fits_prelim['scale_corr']
    scale /= torch.nanmedian(scale, dim=0).values

    def scale_from_cam(scene_points, cam_points, scale):
        cam_points = cam_points[:, np.newaxis]
        return (scene_points - cam_points) * scale[:, np.newaxis] + cam_points

    vertices = scale_from_cam(vertices, cam_points, scale[:, np.newaxis])
    joints = scale_from_cam(joints, cam_points, scale[:, np.newaxis])

    # Now smooth the temporal sequence of root distances and scale accordingly
    if fps < 40:
        kernel_large = torch.tensor(
            [0.01, 0.05, 0.1, 1, 2, 6, 2, 1, 0.1, 0.05, 0.01], dtype=torch.float32
        )
    else:
        kernel_large = torch.tensor(
            [
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                1,
                1.3,
                2,
                3.3,
                6,
                3.3,
                2,
                1.3,
                1,
                0.2,
                0.1,
                0.05,
                0.02,
                0.01,
            ],
            dtype=torch.float32,
        )

    filtered_root, _, split_indices = robust_geometric_filter_twosided(
        joints[:, 0], joint_weights[:, 0], kernel_large, dim_t=-2, eps=5e-2, split_threshold=1
    )

    root_dist_sq = torch.sum(torch.square(joints[:, 0] - cam_points), dim=-1, keepdim=True)
    filtered_root_distance = torch.sum(
        (filtered_root - cam_points) * (joints[:, 0] - cam_points), dim=-1, keepdim=True
    )
    scales = filtered_root_distance / root_dist_sq
    vertices = scale_from_cam(vertices, cam_points, scales)
    joints = scale_from_cam(joints, cam_points, scales)

    # Now smooth the temporal sequence of vertices and joints
    if fps < 40:
        kernel_small = torch.tensor([1, 3, 12, 3, 1], dtype=torch.float32)
    else:
        kernel_small = torch.tensor([1, 1.5, 3, 6, 12, 6, 3, 1.5, 1], dtype=torch.float32)
    points = torch.cat([vertices, joints], dim=-2)
    weights = torch.cat([vertex_weights, joint_weights], dim=-1)
    points, new_weights, _ = robust_geometric_filter_twosided(
        points,
        weights,
        kernel_small,
        dim_t=-3,
        dim_n=-2,
        eps=5e-2,
        split_threshold=1,
        split_indices=split_indices,
    )
    vertices, joints = torch.split(points, [n_verts, n_joints], dim=-2)
    vertex_weights, joint_weights = torch.split(new_weights, [n_verts, n_joints], dim=-1)

    if FLAGS.fps_factor != 1:

        def interp_fun(vals):
            mids = 0.5 * (vals[:-1] + vals[1:])
            mids = torch.cat([mids, vals[-1:]], dim=0)
            return torch.reshape(torch.stack([vals, mids], dim=1), [-1, *vals.shape[1:]])

        # interpolate new inbetween frames
        vertices = interp_fun(vertices)
        joints = interp_fun(joints)
        vertex_weights = interp_fun(vertex_weights)
        joint_weights = interp_fun(joint_weights)
        split_indices = split_indices * FLAGS.fps_factor
        cam_points = interp_fun(cam_points)

    # Now fit again but now do not estimate scale correction
    fits = smplfitter.pt.get_cached_fit_fn(
        body_model_name=body_model_name,
        requested_keys=('vertices', 'joints', 'pose_rotvecs'),
        beta_regularizer=10 * reg_fac,
        beta_regularizer2=0 * reg_fac,
        num_betas=10,
        vertex_subset=vertex_subset_tup,
        share_beta=True,
        final_adjust_rots=True,
        device='cpu',
    )(
        torch.index_select(vertices, dim=1, index=vertex_subset_pt),
        joints,
        torch.index_select(vertex_weights, dim=1, index=vertex_subset_pt),
        joint_weights,
    )

    body_model = smplfitter.pt.get_cached_body_model(body_model_name)
    fit_res = body_model.forward(fits['pose_rotvecs'], fits['shape_betas'], fits['trans'])
    vertices = fit_res['vertices']
    joints = fit_res['joints']

    # Now smooth the root distance by translation
    if fps * FLAGS.fps_factor < 40:
        kernel = torch.tensor([1, 2, 3, 2, 1], dtype=torch.float32)
    else:
        kernel = torch.tensor([1, 1.5, 2, 2.5, 3, 2.5, 2, 1.5, 1], dtype=torch.float32)
    filtered_root = robust_geometric_filter_twosided(
        joints[:, 0],
        joint_weights[:, 0],
        kernel,
        dim_t=-2,
        eps=5e-2,
        split_indices=split_indices,
        split_threshold=1,
    )[0]
    root_dist_sq = torch.sum(torch.square(joints[:, 0] - cam_points), dim=-1, keepdim=True)
    filtered_root_distance = torch.sum(
        (filtered_root - cam_points) * (joints[:, 0] - cam_points), dim=-1, keepdim=True
    )
    scales = (filtered_root_distance / root_dist_sq)[..., np.newaxis]
    offset = (scales - 1) * joints[:, :1]

    vertices = vertices + offset
    joints = joints + offset
    fits['trans'] = fits['trans'] + torch.squeeze(offset, dim=-2)

    is_invalid = torch.all(joint_weights == 0, dim=1, keepdim=True)[..., np.newaxis]
    vertices = torch.where(is_invalid, torch.nan, vertices)
    joints = torch.where(is_invalid, torch.nan, joints)

    is_invalid = torch.squeeze(is_invalid, dim=-1)
    pose_rotvecs = torch.where(is_invalid, torch.nan, fits['pose_rotvecs'])
    shape_betas = torch.where(is_invalid, torch.nan, fits['shape_betas'])
    trans = torch.where(is_invalid, torch.nan, fits['trans'])

    return dict(
        vertices=vertices.numpy(),
        joints=joints.numpy(),
        pose_rotvecs=pose_rotvecs.numpy(),
        shape_betas=shape_betas.numpy(),
        trans=trans.numpy(),
    )


def fill_nan_with_prev_nonnan(a, axis):
    prev = None
    for item in iterdim(a, axis):
        isnan = np.isnan(item)
        if prev is not None:
            item[isnan] = prev[isnan]
        prev = item


def iterdim(a, axis=0):
    a = np.asarray(a)
    leading_indices = (slice(None),) * axis
    for i in range(a.shape[axis]):
        yield a[leading_indices + (i,)]


def build_tracks_via_masks(pred_path, cameras_display, video_imshape, masks, n_verts, n_joints):
    preds_per_frame = []

    pred_reader = BodyDecompressor(pred_path)
    for i_frame, (d, camera_display) in enumerate(
        spu.progressbar(
            zip(pred_reader, cameras_display),
            total=len(masks),
            desc='Matching meshes to masks',
        )
    ):
        points = np.concatenate([d['vertices'], d['joints']], axis=1)
        uncerts = np.concatenate([d['vertex_uncertainties'], d['joint_uncertainties']], axis=1)
        points_and_uncerts = np.concatenate([points, uncerts[..., np.newaxis]], axis=-1)
        ordered_preds = associate_predictions_to_masks_mesh(
            poses3d_pred=points_and_uncerts,
            frame_shape=video_imshape,
            masks=masks[i_frame],
            camera=camera_display,
            n_points=n_verts + n_joints,
            n_verts=n_verts,
            n_coords=4,
            iou_threshold=0.1,
        )
        preds_per_frame.append(ordered_preds)

    tracks = np.stack(preds_per_frame, axis=1)
    if FLAGS.fill_gaps:
        fill_nan_with_prev_nonnan(tracks, axis=1)
    return tracks


def associate_predictions_to_masks_mesh(
    poses3d_pred, frame_shape, masks, camera, n_points, n_verts, n_coords=4, iou_threshold=0
):
    n_true_poses = len(masks)
    result = np.full((n_true_poses, n_points, n_coords), np.nan, dtype=np.float32)
    if n_true_poses == 0:
        return result

    mask_rles = [rlemasklib.RLEMask.from_dict(m) for m in masks]
    mask_shape = mask_rles[0].shape
    camera_rescaled = camera.scale_output(
        [mask_shape[1] / frame_shape[1], mask_shape[0] / frame_shape[0]], inplace=False
    )
    # camera_rescaled = camera.scale_output(mask_shape[1] / frame_shape[1], inplace=False)
    pose_rles = smpl_mask.render_rle(
        poses3d_pred[:, :n_verts, :3], camera_rescaled, mask_shape, 1024
    )

    iou_matrix = rlemasklib.RLEMask.iou_matrix(mask_rles, pose_rles)
    true_indices, pred_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

    for ti, pi in zip(true_indices, pred_indices):
        if iou_matrix[ti, pi] >= iou_threshold:
            result[ti] = poses3d_pred[pi]
    return result


def find_up_vector(points, almost_up=(0, -1, 0), thresh_degrees=60):
    almost_up = np.asarray(almost_up, np.float32)

    import pyransac3d

    plane1 = pyransac3d.Plane()
    _, best_inliers = plane1.fit(points, thresh=25, maxIteration=5000)
    if len(best_inliers) < 3:
        raise ValueError('Could not fit a plane to the points, too few inliers')

    world_up = np.asarray(fit_plane(points[best_inliers]), np.float32)
    if np.dot(world_up, almost_up) < 0:
        world_up *= -1

    if np.rad2deg(np.arccos(np.dot(world_up, almost_up))) > thresh_degrees:
        world_up = almost_up

    world_up = np.array(world_up, np.float32)
    return world_up


def fit_ground_plane(bm, fits, prev_up):
    all_toes = []
    for fit in fits:
        smpl_joints = (
            bm(fit['pose_rotvecs'], fit['shape_betas'], fit['trans'], return_vertices=False)[
                'joints'
            ][:1]
            * 1000
        )
        smpl_toes = smpl_joints[:, 10:12].reshape([-1, 3])
        all_toes.append(smpl_toes)
    all_toes = np.concatenate(all_toes, axis=0)
    new_up = find_up_vector(all_toes, almost_up=prev_up)
    toe_heights = np.dot(all_toes, new_up)
    toe_height_median = np.median(toe_heights)
    freqs, bin_edges = np.histogram(
        toe_heights, bins=100, range=(toe_height_median - 200, toe_height_median + 200)
    )
    i_max_bin = np.argmax(freqs)
    ground_height = (bin_edges[i_max_bin] + bin_edges[i_max_bin + 1]) / 2
    return ground_height, new_up


def fit_plane(points):
    points = np.asarray(points, np.float32)
    x = points - np.mean(points, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return vt[-1]


if __name__ == '__main__':
    main()
