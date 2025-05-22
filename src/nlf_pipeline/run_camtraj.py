import argparse
import os.path as osp

import boxlib
import cameravision
import numpy as np
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import smplfitter.np
import smplfitter.pt
import torch
import framepump
from nlf_pipeline.run_smoothed_smplfitter import robust_geometric_filter
from nlf_pipeline.util.paths import INFERENCE_ROOT
from simplepyutils import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=str)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--camera-move-start-sec", type=int, default=4)
    parser.add_argument("--fov", type=float, default=55)
    parser.add_argument("--resolution", default=(1920, 1080), nargs=2, type=int)
    parser.add_argument('--skip-existing', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--fps-factor', type=int, default=1)
    spu.initialize(parser)
    torch.set_num_threads(12)


def main():
    initialize()
    body_model_name = 'smpl'
    fov = FLAGS.fov
    cam_padding_factor = 0.2

    dfps_str = f'_fps{FLAGS.fps_factor}' if FLAGS.fps_factor != 1 else ''
    video_path = f'{INFERENCE_ROOT}/videos_in/{FLAGS.video_id}.mp4'

    fit_path = (
        f'{INFERENCE_ROOT}/smooth_fits/{FLAGS.video_id}{FLAGS.suffix}_smoothfits{dfps_str}.pkl'
    )
    cam_traj_path = (
        f'{INFERENCE_ROOT}/camera_traj/' f'{FLAGS.video_id}{FLAGS.suffix}_camtraj{dfps_str}.pkl'
    )
    camera_path = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}{dfps_str}.pkl'
    ground_plane_path = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}{FLAGS.suffix}_ground.pkl'

    if FLAGS.skip_existing and osp.exists(cam_traj_path):
        return

    video = framepump.VideoFrames(video_path)
    video_imsize = video.imshape[::-1]
    fps = video.fps * FLAGS.fps_factor
    FLAGS.resolution = spu.rounded_int_tuple(
        video_imsize
        * (np.min(np.array(FLAGS.resolution, np.float32) / video_imsize.astype(np.float32))),
        2,
    )

    if osp.isfile(camera_path):
        cameras_display = spu.load_pickle(camera_path)
    else:
        cameras_display = [cameravision.Camera.from_fov(fov, video.imshape)] * len(video)

    ground_data = spu.load_pickle(ground_plane_path)
    world_up = ground_data['world_up']
    for cam in cameras_display:
        cam.world_up = world_up

    fits_per_frame = spu.load_pickle(fit_path)

    camera_move_start_frame = FLAGS.camera_move_start_sec * fps
    view_cameras = follow_people_with_camera(
        body_model_name, cam_padding_factor, fits_per_frame, cameras_display, video.imshape, fps
    )
    filter_scale = 1 if fps < 40 else 2
    view_cameras = smooth_camera_trajectory(view_cameras, filter_scale=filter_scale)
    smooth_display_cameras = smooth_camera_trajectory(cameras_display, filter_scale=filter_scale)

    view_cameras_base = [
        interpolate_cameras(
            cam_base,
            cam_smooth,
            t=smootherstep(i_frame, camera_move_start_frame, length=8 * fps),
        ).scale_output(FLAGS.resolution[1] / video.imshape[0])
        for i_frame, (cam_base, cam_smooth) in enumerate(
            zip(cameras_display, smooth_display_cameras)
        )
    ]

    view_cameras = [
        interpolate_cameras(
            cam_base,
            cam_view,
            t=smootherstep(i_frame, camera_move_start_frame, length=8 * fps),
        )
        for i_frame, (cam_base, cam_view) in enumerate(zip(view_cameras_base, view_cameras))
    ]

    # view_cameras = smooth_zoom_to_keep_person_in_frame(people_per_frame, view_cameras)
    spu.dump_pickle(view_cameras, cam_traj_path)


def smooth_camera_trajectory(view_cameras, filter_scale=1):
    s = filter_scale
    cam_translations = torch.as_tensor(np.stack([cam.t for cam in view_cameras], axis=0))
    cam_targets = torch.as_tensor(
        np.stack([cam.t + cam.R[2] * 100 for cam in view_cameras], axis=0)
    )
    ramp_up = torch.linspace(0, 1, int(round(15 * s)))
    ramp_down = torch.linspace(1, 0, int(round(15 * s)))
    kernel = torch.cat([ramp_up, torch.ones([1 + 2 * int(round(20 * s))]), ramp_down], dim=0)
    smoothed_translations, _ = robust_geometric_filter(
        cam_translations, None, kernel, dim=-2, eps=10, n_iter=0
    )
    smoothed_targets, _ = robust_geometric_filter(
        cam_targets, None, kernel, dim=-2, eps=10, n_iter=0
    )
    cam_focals = torch.as_tensor(
        [[cam.intrinsic_matrix[0, 0], cam.intrinsic_matrix[1, 1]] for cam in view_cameras]
    )
    smoothed_focals, _ = robust_geometric_filter(cam_focals, None, kernel, dim=0, eps=10, n_iter=0)

    cam_rolls = torch.as_tensor(
        [cam.get_pitch_roll()[1] for cam in view_cameras], dtype=torch.float32
    )
    smoothed_rolls, _ = robust_geometric_filter(cam_rolls, None, kernel, dim=0, eps=10, n_iter=0)

    out_cameras = []
    for i, (cam, t, f, targ, roll) in enumerate(
        zip(view_cameras, smoothed_translations, smoothed_focals, smoothed_targets, smoothed_rolls)
    ):
        cam: cameravision.Camera = cam.copy()
        cam.t = t.numpy()
        cam.turn_towards(target_world_point=targ.numpy())
        cam.rotate(roll=-roll)
        f = f.numpy()
        cam.intrinsic_matrix[0, 0] = f[0]
        cam.intrinsic_matrix[1, 1] = f[1]
        out_cameras.append(cam)

    return out_cameras


def follow_people_with_camera(
    body_model_name, cam_padding_factor, fits_per_frame, cameras, video_imshape, fps
):
    body_model = smplfitter.pt.BodyModel(body_model_name, num_betas=10, vertex_subset_size=128)

    view_cameras = []
    for i_frame, (fits, cameravision_view_camera) in enumerate(
        zip(spu.progressbar(fits_per_frame, desc='Following people with camera'), cameras)
    ):
        smpl_vertices = body_model(
            torch.from_numpy(fits['pose_rotvecs']),
            torch.from_numpy(fits['shape_betas']),
            torch.from_numpy(fits['trans']),
        )['vertices'].numpy()

        n_people = len(smpl_vertices)
        half_height = 0.5 * video_imshape[0]
        offset = np.abs(cameravision_view_camera.intrinsic_matrix[1, 2] - half_height)
        tan = (
            (1 + cam_padding_factor)
            * (half_height + offset)
            / cameravision_view_camera.intrinsic_matrix[1, 1]
        )
        view_angle = np.rad2deg(2 * np.arctan(tan))
        cameravision_view_camera = cameravision_view_camera.copy()
        cameravision_view_camera.intrinsic_matrix = cameravision.intrinsics_from_fov(
            view_angle, video_imshape[:2], side='height'
        )
        cameravision_view_camera.undistort()
        cameravision_view_camera.scale_output(FLAGS.resolution[1] / video_imshape[0])
        cameravision_view_camera.center_principal_point(imshape=FLAGS.resolution[::-1])

        if n_people == 0:
            if view_cameras:
                view_cameras.append(view_cameras[-1])
            else:
                view_cameras.append(cameravision_view_camera)
            continue

        smpl_vertices = np.reshape(smpl_vertices, (-1, 3))

        bird_cam = cameravision_view_camera.copy()
        bird_cam.t = bird_cam.camera_to_world(np.array([-2000, -800, 1800], np.float32))
        mesh_center = np.mean(smpl_vertices, axis=0)
        cam_mesh_avg = (cameravision_view_camera.t + mesh_center * 1000) * 0.5

        bird_cam.turn_towards(target_world_point=cam_mesh_avg)

        for _ in range(5):
            image_points = bird_cam.world_to_image(smpl_vertices * 1000)
            center = boxlib.center(boxlib.bb_of_points(image_points))
            bird_cam.turn_towards(target_image_point=center)

        image_points = bird_cam.world_to_image(smpl_vertices * 1000)
        dist_from_center = np.max(np.abs(image_points - np.array(FLAGS.resolution) / 2), axis=0)
        zoom_factor = np.min((np.array(FLAGS.resolution) / 2) / dist_from_center) * 0.6

        mean_z = np.mean(bird_cam.world_to_camera(smpl_vertices * 1000)[..., 2])
        new_mean_z = np.clip(mean_z / zoom_factor, 2000, 15000)
        bird_cam.t -= bird_cam.R[2] * new_mean_z

        for _ in range(5):
            image_points = bird_cam.world_to_image(smpl_vertices * 1000)
            zoom = (
                np.min(np.array(FLAGS.resolution) / boxlib.bb_of_points(image_points)[2:4]) * 0.6
            )
            bird_cam.zoom(zoom)

        t = np.clip((i_frame - FLAGS.camera_move_start_sec * fps) / (4 * fps), 0, 1)
        interp_cam = interpolate_cameras(cameravision_view_camera, bird_cam, t)

        interp_cam2 = interp_cam.copy()
        interp_cam2.turn_towards(target_world_point=cam_mesh_avg)

        for _ in range(5):
            image_points = interp_cam2.world_to_image(smpl_vertices * 1000)
            center = boxlib.center(boxlib.bb_of_points(image_points))
            interp_cam2.turn_towards(target_image_point=center)

            image_points = interp_cam2.world_to_image(smpl_vertices * 1000)
            zoom = (
                np.min(np.array(FLAGS.resolution) / boxlib.bb_of_points(image_points)[2:4]) * 0.6
            )
            interp_cam2.zoom(zoom)

        interp_cam3 = interpolate_cameras(interp_cam, interp_cam2, np.minimum(1, t * 10))
        view_cameras.append(interp_cam3)
    return view_cameras


def interpolate_cameras(cam1, cam2, t):
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
    cam = cameravision.Camera(
        intrinsic_matrix=intr,
        optical_center=optical_center,
        distortion_coeffs=dist,
        world_up=cam1.world_up,
    )

    target1 = cam1.t + cam1.R[2] * 100
    target2 = cam2.t + cam2.R[2] * 100
    target = target1 + t * (target2 - target1)
    cam.turn_towards(target_world_point=target)
    return cam


def smootherstep(x, x0=0, x1=None, length=None):
    if length is None:
        length = x1 - x0
    y = np.clip((x - x0) / length, 0, 1)
    return y * y * y * (y * (y * 6.0 - 15.0) + 10.0)



if __name__ == '__main__':
    main()
