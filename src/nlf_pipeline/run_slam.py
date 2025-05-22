"""This script is based on the TRAM repo by Wang et al. https://github.com/yufu-wang/tram"""

import torch

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import argparse
import cameravision
import framepump
import numpy as np
import rlemasklib
from rlemasklib import RLEMask
import simplepyutils as spu
import torch
import torchmin
from PIL import Image
from masked_droid_slam.droid import Droid
from nlf_pipeline.util.paths import DATA_ROOT
from simplepyutils import FLAGS
import os.path as osp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid-model-path", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--fov", type=float)
    parser.add_argument("--semseg-mask-path", type=str, required=True)
    parser.add_argument("--vos-mask-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--smooth", action=spu.argparse.BoolAction)
    parser.add_argument("--skip-existing", action=spu.argparse.BoolAction, default=True)
    spu.argparse.initialize(parser)

    if FLAGS.skip_existing and osp.exists(FLAGS.output_path):
        print(f"Already done")
        return

    if FLAGS.droid_model_path is None:
        FLAGS.droid_model_path = f'{DATA_ROOT}/models/droid.pth'

    video_sliced = framepump.VideoFrames(FLAGS.video_path)[::5][:200]
    masks = load_mask_unions()

    if FLAGS.fov is not None:
        camera = cameravision.Camera.from_fov(FLAGS.fov, video_sliced.imshape)
        is_static = False
    else:
        print('Calibrating intrinsics...')
        camera, is_static = calibrate_intrinsics(video_sliced, masks[::5][:200])

    video_full = framepump.VideoFrames(FLAGS.video_path)
    if is_static:
        cams = [camera.copy() for _ in range(len(video_full))]
    else:
        cam_R, cam_t = run_metric_slam(video_full, masks, camera)
        cams = [
            cameravision.Camera(
                rot_world_to_cam=R.T,
                optical_center=t * 1000,
                intrinsic_matrix=camera.intrinsic_matrix,
                world_up=(0, -1, 0),
            )
            for R, t in zip(cam_R.numpy(), cam_t.numpy())
        ]

    if FLAGS.smooth:
        print('Smoothing camera trajectory...')
        from nlf_pipeline.run_camtraj import smooth_camera_trajectory

        cams = smooth_camera_trajectory(cams, filter_scale=0.1)
    spu.dump_pickle(cams, FLAGS.output_path)


def run_metric_slam(video, masks, camera):
    droid, traj = run_slam(video, masks, camera)
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    del droid
    torch.cuda.empty_cache()

    # Estimate metric depth
    print('Estimating metric depth...')
    model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, verbose=False)
    model_zoe_n = model_zoe_n.eval().to('cuda')

    h0, w0, h1, w1, scale_factor = scale_resolution(video.imshape)

    pred_depths = []
    for t, im in spu.progressbar(
        spu.filter_by_index(video.resized((h1, w1)), indices=tstamp, enumerate=True),
        desc='Estimating depth',
    ):
        pred_depth = model_zoe_n.infer_pil(Image.fromarray(im))
        pred_depths.append(pred_depth)

    # Estimate metric scale
    scales = []
    for t, disp, pred_depth in zip(
        spu.progressbar(tstamp, desc='Estimating scale'), disps, pred_depths
    ):
        slam_depth = 1 / disp
        rle_bg = rlemasklib.RLEMask.from_dict(masks[t]).complement(inplace=True)
        mask_bg = rle_bg.resize(pred_depth.shape[:2]).to_array().view(np.bool_)
        scale = estimate_scale_hybrid(slam_depth, pred_depth, mask_bg=mask_bg)
        scales.append(scale)

    scale = np.median(scales)

    # convert to metric-scale camera extrinsics
    pred_cam_t = torch.tensor(traj[:, :3]) * scale
    pred_cam_q = torch.tensor(traj[:, 3:])
    pred_cam_r = quaternion_to_matrix(pred_cam_q)
    return pred_cam_r, pred_cam_t


def run_slam(video, masks, camera, depth=None):
    h0, w0, h1, w1, scale_factor = scale_resolution(video.imshape)
    droid = Droid(
        argparse.Namespace(
            t0=0,
            stride=1,
            weights=FLAGS.droid_model_path,
            buffer=512,
            image_size=(h1, w1),
            beta=0.3,
            filter_thresh=2.4,
            warmup=8,
            keyframe_thresh=4.0,
            frontend_radius=2,
            frontend_nms=1,
            backend_thresh=22.0,
            backend_radius=2,
            backend_nms=3,
            stereo=False,
            upsample=True,
            disable_vis=True,
            frontend_window=25,
            frontend_thresh=16.0,
        )
    )

    camera = camera.scale_output([w1 / w0, h1 / h0], inplace=False)
    K = camera.intrinsic_matrix
    intrinsics = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)

    for t, (image, mask) in enumerate(
        zip(spu.progressbar(video.resized((h1, w1)), desc='Running SLAM'), masks)
    ):
        rle = rlemasklib.RLEMask.from_dict(mask)
        cameravision.reprojection.mask_image_by_rle(image, rle.resize((h1, w1)), value=0)
        small_mask_arr = rle.resize((h1 // 8, w1 // 8)).to_array(value=255)
        droid.track(
            t,
            to_pt(image).float(),
            intrinsics=intrinsics,
            depth=depth,
            mask=torch.from_numpy(small_mask_arr),
        )

    def _image_stream():
        for t, image in enumerate(
            spu.progressbar(video.resized((h1, w1)), desc='Terminating SLAM')
        ):
            yield t, to_pt(image), intrinsics

    traj = droid.terminate(_image_stream())
    return droid, traj


def to_pt(im):
    return torch.as_tensor(im).to(device='cuda', non_blocking=True).permute(2, 0, 1).unsqueeze(0)


def calibrate_intrinsics(sliced_video, masks, low=50, high=75, step=5):
    err_dict = {}

    fov = 55
    camera = cameravision.Camera.from_fov(fov, sliced_video.imshape)
    print(f"Testing FOV={fov}°...")
    err, is_static = test_slam(sliced_video, masks, camera=camera)
    print(f"FOV={fov}°, Error={err:.3f}")
    err_dict[fov] = err

    if is_static:
        return camera, True

    def try_fovs(fovs):
        for fov in fovs:
            if fov in err_dict:
                continue
            print(f"Testing FOV={fov}°...")
            camera.intrinsic_matrix = cameravision.intrinsics_from_fov(fov, sliced_video.imshape)
            err, is_static = test_slam(sliced_video, masks, camera=camera)
            print(f"FOV={fov}°, Error={err:.3f}")
            err_dict[fov] = err

    try_fovs(range(low, high, step))
    best_fov = min(err_dict, key=err_dict.get)
    try_fovs(range(best_fov - step // 2, best_fov + step // 2 + 1))
    best_fov = min(err_dict, key=err_dict.get)
    print(f"Best FOV={best_fov}°, Error={err_dict[best_fov]:.3f}")
    camera.intrinsic_matrix = cameravision.intrinsics_from_fov(best_fov, sliced_video.imshape)
    return camera, is_static


def test_slam(video, masks, camera):
    h0, w0, h1, w1, scale_factor = scale_resolution(video.imshape)
    droid = Droid(
        argparse.Namespace(
            t0=0,
            stride=1,
            weights=FLAGS.droid_model_path,
            buffer=512,
            image_size=(h1, w1),
            beta=0.3,
            filter_thresh=2.4,
            warmup=8,
            keyframe_thresh=4.0,
            frontend_radius=2,
            frontend_nms=1,
            backend_thresh=22.0,
            backend_radius=2,
            backend_nms=3,
            stereo=False,
            upsample=True,
            disable_vis=True,
            frontend_window=10,
            frontend_thresh=10,
        )
    )

    camera = camera.scale_output([w1 / w0, h1 / h0], inplace=False)
    K = camera.intrinsic_matrix
    intrinsics = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)

    for t, (image, mask) in enumerate(zip(spu.progressbar(video.resized((h1, w1))), masks)):
        rle = rlemasklib.RLEMask.from_dict(mask)
        cameravision.reprojection.mask_image_by_rle(image, rle.resize((h1, w1)), value=0)
        small_mask_arr = rle.resize((h1 // 8, w1 // 8)).to_array(value=255)
        droid.track(t, to_pt(image), intrinsics=intrinsics, mask=torch.from_numpy(small_mask_arr))

    if droid.video.counter.value <= 1:
        return None, True
    else:
        return droid.compute_error(), False


def quaternion_to_matrix(quaternions):
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (torch.square(quaternions)).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def scale_resolution(imshape):
    h0, w0 = imshape[:2]
    scale_factor = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale_factor) // 8 * 8
    w1 = int(w0 * scale_factor) // 8 * 8
    return h0, w0, h1, w1, scale_factor


def load_mask_unions():
    sem_masks = [RLEMask.from_dict(x) for x in spu.load_pickle(FLAGS.semseg_mask_path)]
    vos_masks = [[RLEMask.from_dict(x) for x in xs] for xs in spu.load_pickle(FLAGS.vos_mask_path)]
    vos_mask_unions = [RLEMask.union(ms) for ms in vos_masks]
    return [(m | v.resize(m.shape)).to_dict() for m, v in zip(sem_masks, vos_mask_unions)]


def estimate_scale_hybrid(slam_depth, pred_depth, sigma=0.5, mask_bg=None, far_thresh=10):
    """Depth-align by iterative + robust least-square"""

    # Stage 1: Iterative steps
    s = pred_depth / slam_depth

    robust = mask_bg & (0 < pred_depth) & (pred_depth < 10)
    s_est = s[robust]
    scale = np.median(s_est)

    for _ in range(10):
        slam_depth_0 = slam_depth * scale
        robust = (
            mask_bg
            & (0 < slam_depth_0)
            & (slam_depth_0 < far_thresh)
            & (0 < pred_depth)
            & (pred_depth < far_thresh)
        )
        s_est = s[robust]
        scale = np.median(s_est)

    # Stage 2: Robust optimization
    robust = (
        mask_bg
        & (0 < slam_depth_0)
        & (slam_depth_0 < far_thresh)
        & (0 < pred_depth)
        & (pred_depth < far_thresh)
    )
    pm = torch.from_numpy(pred_depth[robust])
    sm = torch.from_numpy(slam_depth[robust])

    def f(x):
        return gmof(sm * x - pm, sigma=sigma).mean()

    x0 = torch.tensor([scale])
    result = torchmin.minimize(f, x0, method='bfgs')
    scale = result.x.detach().cpu().item()
    return scale


def gmof(x, sigma=100):
    """Geman-McClure error function"""
    x_squared = torch.square(x)
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


if __name__ == '__main__':
    main()
