import argparse
import os.path as osp

import bodycompress
import cameravision
import numpy as np
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import smplfitter.np
from nlf_pipeline.util.paths import INFERENCE_ROOT
from simplepyutils import FLAGS
import framepump


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=str)
    parser.add_argument("--video-id", type=str)
    parser.add_argument("--suffix", type=str, default='')
    parser.add_argument("--fov", type=float, default=55)
    parser.add_argument("--ground-offset", type=float, default=1.02)
    parser.add_argument("--fps-factor", type=int, default=1)
    parser.add_argument("--resolution", default=(1920, 1080), nargs=2, type=int)
    parser.add_argument('--camera-view', action=spu_argparse.BoolAction)
    parser.add_argument('--nonparametric', action=spu_argparse.BoolAction)
    parser.add_argument('--nonparametric-also', action=spu_argparse.BoolAction)
    parser.add_argument('--display-frame', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--viz', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--skip-existing', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--renderer', type=str, default='blendipose')
    parser.add_argument('--motion-blur', action=spu_argparse.BoolAction)
    parser.add_argument('--depth-of-field', action=spu_argparse.BoolAction)
    spu.initialize(parser)



def main():
    initialize()
    dfps_str = f'_fps{FLAGS.fps_factor}' if FLAGS.fps_factor != 1 else ''
    video_path_base = f'{INFERENCE_ROOT}/videos_in/{FLAGS.video_id}.mp4'
    video_path_fpsfac = f'{INFERENCE_ROOT}/videos_in/{FLAGS.video_id}{dfps_str}.mp4'
    frame_dtype = np.uint16 if FLAGS.renderer == 'blendipose' else np.uint8
    if osp.exists(video_path_fpsfac):
        video = framepump.VideoFrames(video_path_fpsfac, dtype=frame_dtype)
    else:
        video = framepump.VideoFrames(video_path_base, dtype=frame_dtype).repeat_each_frame(2)

    camera_path = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}{dfps_str}.pkl'
    fit_path = (
        f'{INFERENCE_ROOT}/smooth_fits/{FLAGS.video_id}{FLAGS.suffix}_smoothfits{dfps_str}.pkl'
    )
    np_path = f'{INFERENCE_ROOT}/preds_np/{FLAGS.video_id}{FLAGS.suffix}.xz'
    cam_traj_path = (
        f'{INFERENCE_ROOT}/camera_traj/{FLAGS.video_id}{FLAGS.suffix}_camtraj{dfps_str}.pkl'
    )
    ground_plane_path = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}{FLAGS.suffix}_ground.pkl'

    np_suf = '_np' if FLAGS.nonparametric else ''
    np_suf = '_np_both' if FLAGS.nonparametric_also else np_suf
    noframe_suffix = f'_noframe' if not FLAGS.display_frame else ''
    out_video_camview_path = (
        f'{INFERENCE_ROOT}/videos_out/{FLAGS.video_id}{FLAGS.suffix}_{FLAGS.renderer}_camview'
        f'{dfps_str}{np_suf}{noframe_suffix}.mp4'
    )
    out_video_bird_path = (
        f'{INFERENCE_ROOT}/videos_out/{FLAGS.video_id}'
        f'{FLAGS.suffix}_{FLAGS.renderer}_birdview{dfps_str}{np_suf}.mp4'
    )
    out_video_path = out_video_camview_path if FLAGS.camera_view else out_video_bird_path
    if FLAGS.skip_existing and osp.exists(out_video_path):
        print(f'Skipping existing video {out_video_path}')
        return

    video_imshape = video.imshape[:2]
    video_imsize = np.array(video_imshape[::-1])

    resolution = rounded_int_tuple(
        video_imsize
        * (np.min(np.array(FLAGS.resolution, np.float32) / np.array(video_imsize, np.float32))),
        2,
    )
    preview_resolution = rounded_int_tuple(
        video_imsize
        * (np.min(np.array([1920, 1080], np.float32) / np.array(video_imsize, np.float32))),
        2,
    )

    body_model = smplfitter.np.get_cached_body_model('smpl', 'neutral')

    n_frames = len(video)

    if osp.exists(camera_path):
        camera_displays = spu.load_pickle(camera_path)
    else:
        camera_displays = [
            cameravision.Camera.from_fov(FLAGS.fov, video_imshape) for _ in range(n_frames)
        ]

    if FLAGS.nonparametric:
        fits = [None] * n_frames
    else:
        fits = spu.load_pickle(fit_path)

    if FLAGS.camera_view:
        view_cameras = [c.shift_image(0.5, inplace=False).scale_output(resolution[1] / video_imsize[1]).shift_image(-0.5) for c in camera_displays]
    else:
        view_cameras = spu.load_pickle(cam_traj_path)
        view_cameras = (c.scale_output(resolution[1] / 1080, inplace=False).center_principal_point(resolution[::-1]) for c in view_cameras)

    if osp.exists(ground_plane_path):
        ground_data = spu.load_pickle(ground_plane_path)
        ground_height = ground_data['ground_height']
        world_up = ground_data['world_up']
    else:
        ground_height = FLAGS.ground_offset
        world_up = np.array([0, -1, 0], np.float32)

    if FLAGS.renderer == 'blendipose':
        import blendipose

        print(FLAGS.viz, 'viz')
        renderer = blendipose.Renderer(
            resolution=resolution,
            preview_resolution=preview_resolution if FLAGS.viz else None,
            body_model_faces=body_model.faces,
            show_image=FLAGS.display_frame,
            frame_background=FLAGS.camera_view and FLAGS.display_frame,
            ground_plane_height=ground_height,
            world_up=world_up,
            show_ground_plane=not FLAGS.camera_view,
            show_camera=not FLAGS.camera_view,
            body_alpha=0.75 if FLAGS.camera_view and FLAGS.display_frame else 1.0,
            motion_blur=FLAGS.motion_blur,
            depth_of_field=FLAGS.depth_of_field,
        )
    elif FLAGS.renderer == 'poseviz':
        import poseviz

        renderer = poseviz.PoseViz(
            resolution=resolution,
            body_model_faces=body_model.faces,
            show_image=FLAGS.display_frame,
            ground_plane_height=ground_height,
            world_up=world_up,
            show_ground_plane=not FLAGS.camera_view,
            show_camera_wireframe=not FLAGS.camera_view,
        )

    else:
        raise ValueError(f'Unknown renderer {FLAGS.renderer}')

    renderer.new_sequence_output(out_video_path, fps=video.fps, audio_source_path=video.path)

    np_preds = bodycompress.BodyDecompressor(np_path)
    with renderer:
        if FLAGS.fps_factor != 1:
            np_preds = spu.repeat_n(np_preds, FLAGS.fps_factor)

        for i_frame, (cameravision_view_camera, frame, fit, np_pred, camera_display) in enumerate(
            zip(
                spu.progressbar(view_cameras, total=n_frames),
                video,
                fits,
                np_preds,
                camera_displays,
            )
        ):
            if FLAGS.nonparametric_also:
                smpl_vertices_fit = (
                    body_model(fit['pose_rotvecs'], fit['shape_betas'], fit['trans'])['vertices']
                    * 1000
                )
                renderer.update(
                    frame=frame,
                    vertices=smpl_vertices_fit,
                    vertices_alt=np_pred['vertices'],
                    camera=camera_display,
                    viz_camera=cameravision_view_camera,
                )
            elif FLAGS.nonparametric:
                renderer.update(
                    frame=frame,
                    vertices=np_pred['vertices'],
                    camera=camera_display,
                    viz_camera=cameravision_view_camera,
                )
            else:
                smpl_vertices = (
                    body_model(fit['pose_rotvecs'], fit['shape_betas'], fit['trans'])['vertices']
                    * 1000
                )
                renderer.update(
                    frame=frame,
                    vertices=smpl_vertices,
                    camera=camera_display,
                    viz_camera=cameravision_view_camera,
                )

            #if i_frame > 10:
            #    break


def rounded_int_tuple(p, divisor=1):
    return tuple([round(x / divisor) * divisor for x in p])


if __name__ == '__main__':
    main()
