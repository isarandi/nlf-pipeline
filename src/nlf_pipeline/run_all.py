import argparse
import os.path as osp
import subprocess
import sys
from datetime import datetime

import simplepyutils as spu
import framepump
import simplepyutils.argparse as spu_argparse
from nlf_pipeline.util.paths import DATA_ROOT, INFERENCE_ROOT
from simplepyutils import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=str)
    parser.add_argument(
        "--time-range", type=str, default=None, help='Format: [HH:]MM:SS[.ff]-[HH:]MM:SS[.ff]'
    )
    parser.add_argument(
        '--static-camera',
        action=spu_argparse.BoolAction,
        help='If True, skip camera motion estimation',
    )
    parser.add_argument(
        '--gui-selection',
        action=spu_argparse.BoolAction,
        help='If True, display a GUI for selecting the target subjects, otherwise the persons with '
        'highest segmentation scores are selected',
    )
    parser.add_argument(
        '--seg-init-time',
        type=str,
        help='Which time in the video should be used for selecting the target subjects. Format:'
        ' [HH:]MM:SS[.ff], default: middle of the video (after applying --time-range)',
    )
    parser.add_argument(
        '--max-persons',
        type=int,
        default=-1,
        help='Maximum number of persons to segment, -1 means unlimited',
    )
    parser.add_argument(
        '--fps-factor',
        type=int,
        default=None,
        help='Factor by which to increase the FPS of the smoothed fitted predictions, and the video '
        'itself for rendering. By default, the FPS is doubled if the original FPS is less than '
        '40, otherwise it is kept the same',
    )
    parser.add_argument(
        '--fov',
        type=float,
        help='Field of view in degrees, if not specified and --static-camera is False, fov will be estimated',
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='/work9/sarandi/data/nlf_models/nlf_l_0.3.2b.torchscript',
    )
    parser.add_argument('--viz', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--renderer', type=str)
    spu.initialize(parser)


def main():
    initialize()
    vid = FLAGS.video_id
    # check if the video exists. if not, run the download script
    video_path = f'{INFERENCE_ROOT}/videos_in/{vid}.mp4'
    if not osp.exists(video_path):
        downloader_path = osp.join(osp.dirname(__file__), 'download_yt.sh')
        subprocess.run(['bash', downloader_path, vid])

    if FLAGS.fps_factor is None:
        fps = framepump.get_fps(video_path)
        FLAGS.fps_factor = 1 if fps > 40 else 2

    maybe_viz = ['--viz' if FLAGS.viz else '--no-viz']

    if FLAGS.time_range is not None:
        vid_new = FLAGS.video_id + '_cut' + FLAGS.time_range.replace(':', '').replace('-', '_')
        video_path_new = f'{INFERENCE_ROOT}/videos_in/{vid_new}.mp4'
        if not osp.exists(video_path_new):
            start_time, end_time = FLAGS.time_range.split('-')
            framepump.trim_video(video_path, video_path_new, start_time, end_time)
        vid = vid_new
        video_path = video_path_new

    banner('Running video object segmentation')

    if FLAGS.seg_init_time is None:
        n_frames = framepump.num_frames(video_path)
        init_frame_num = n_frames // 2
    else:
        start_time_sec = (
            timestamp_to_seconds(FLAGS.time_range.split('-')[0])
            if FLAGS.time_range is not None
            else 0
        )
        init_frame_num = int(
            (timestamp_to_seconds(FLAGS.seg_init_time) - start_time_sec)
            * framepump.get_fps(video_path)
        )

    subprocess.run(
        [
            sys.executable,
            '-m',
            'stcnbuf.segment_video',
            f'--video-path={video_path}',
            f'--output={INFERENCE_ROOT}/masks/{vid}_masks.pkl',
            f'--init-frame={init_frame_num}',
            f'--max-persons={FLAGS.max_persons}',
            '--permute-frames',
            '--passes=2',
            '--morph-cleanup',
        ]
        + maybe_viz
        + (['--gui-selection'] if FLAGS.gui_selection else [])
    )

    if not FLAGS.static_camera:
        output_cam_path = f'{INFERENCE_ROOT}/cameras/{vid}.pkl'
        if not osp.exists(output_cam_path):
            banner('Running semantic segmentation for camera motion estimation')
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'nlf_pipeline.run_semseg',
                    f'--video-dir={INFERENCE_ROOT}/videos_in',
                    f'--file-pattern={vid}.mp4',
                    f'--output-dir={INFERENCE_ROOT}/masks_semseg',
                    '--mask-threshold=0.2',
                    '--batch-size=16',
                ]
                + maybe_viz
            )

            banner('Running camera motion estimation')
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'nlf_pipeline.run_slam',
                    f'--droid-model-path={DATA_ROOT}/models/droid.pth',
                    f'--video-path={video_path}',
                    f'--output-path={output_cam_path}',
                    f'--semseg-mask-path={INFERENCE_ROOT}/masks_semseg/{vid}_masks.pkl',
                    f'--vos-mask-path={INFERENCE_ROOT}/masks/{vid}_masks.pkl',
                    '--smooth',
                ]
                + ([f'--fov={FLAGS.fov}'] if FLAGS.fov is not None else [])
            )

            if FLAGS.viz:
                subprocess.run(
                    [
                        sys.executable,
                        '-m',
                        'nlf_pipeline.viz_estimated_camera_motion',
                        f'--video-id={vid}',
                        '--camera-view',
                    ]
                )

    banner('Running nonparametric per-frame 3D human estimation (NLF)')
    subprocess.run(
        [
            sys.executable,
            '-m',
            'nlf_pipeline.run_nlf',
            f'--model-path={FLAGS.model_path}',
            #'--model-path=/work9/sarandi/data/nlf_models/nlf_vitg_multi.torchscript',
            f'--video-dir={INFERENCE_ROOT}/videos_in',
            f'--file-pattern={vid}.mp4',
            f'--output-dir={INFERENCE_ROOT}/preds_np',
            f'--mask-dir={INFERENCE_ROOT}/masks',
            f'--output-video-dir={INFERENCE_ROOT}/videos_out',
            '--viz-downscale=1',
            '--batch-size=8',
            '--num-aug=5',
            '--internal-batch-size=8',
            '--antialias-factor=2',
            '--suppress-implausible-poses',
            '--max-detections=-1',
            '--detector-threshold=0.15',
            '--detector-nms-iou-threshold=0.8',
            '--detector-flip-aug',
            '--write-video',
            #'--use-detector-even-with-masks',
            f'--suffix={FLAGS.suffix}',
            #'--clahe',
        ]
        + maybe_viz
        + ([] if FLAGS.static_camera else ['--camera-dir=' + f'{INFERENCE_ROOT}/cameras'])
    )

    banner('Running SMPL fitting and smoothing')
    subprocess.run(
        [
            sys.executable,
            '-m',
            'nlf_pipeline.run_smoothed_smplfitter',
            f'--video-id={vid}',
            f'--fps-factor={FLAGS.fps_factor}',
            f'--suffix={FLAGS.suffix}',
        ]
    )

    banner('Generating visualizer camera trajectory')
    subprocess.run(
        [
            sys.executable,
            '-m',
            'nlf_pipeline.run_camtraj',
            f'--video-id={vid}',
            '--resolution',
            '1920',
            '1080',
            f'--fps-factor={FLAGS.fps_factor}',
            f'--suffix={FLAGS.suffix}',
        ]
        + ([f'--fov={FLAGS.fov}'] if FLAGS.fov is not None else [])
    )

    if FLAGS.fps_factor > 1 and not osp.exists(f'{INFERENCE_ROOT}/videos_in/{vid}_fps2.mp4'):
        banner('Running frame interpolation')
        if FLAGS.fps_factor != 2:
            raise ValueError('fps_factor must be 1 or 2 for now')
        subprocess.run(
            [
                sys.executable,
                '-m',
                'nlf_pipeline.run_frame_interp',
                f'{INFERENCE_ROOT}/videos_in/{vid}.mp4',
                f'{INFERENCE_ROOT}/videos_in/{vid}_fps2.mp4',
            ]
            # + maybe_viz
        )

    if FLAGS.renderer is None:
        return

    banner('Rendering')
    render_args = [
        sys.executable,
        '-m',
        'nlf_pipeline.run_render',
        f'--video-id={vid}',
        '--resolution',
        '480',
        '270',
        f'--renderer={FLAGS.renderer}',
        f'--fps-factor={FLAGS.fps_factor}',
        f'--suffix={FLAGS.suffix}',
        '--motion-blur',
    ] + maybe_viz

    for camview_arg in ['--camera-view', '--no-camera-view']:
        fov_arg = (
            [f'--fov={FLAGS.fov}']
            if FLAGS.fov is not None and camview_arg == '--camera-view'
            else []
        )
        subprocess.run(render_args + [camview_arg] + fov_arg)

    # banner('Rendering (no frame)')
    # subprocess.run(render_args + ['--no-display-frame', '--camera-view'] + maybe_viz)
    #
    # banner('Rendering (nonparametric)')
    # for camview_arg in ['--no-camera-view', '--camera-view']:
    #     subprocess.run(
    #         render_args
    #         + [camview_arg, '--nonparametric', '--fps-factor=1', '--no-motion-blur']
    #         + maybe_viz
    #     )


def banner(title):
    print('\n' + '=' * 80)
    print('\n' * 2)
    print(' ' * 10 + title)
    print('\n' * 2)
    print('=' * 80 + '\n')
    print('\n')


def timestamp_to_seconds(timestamp):
    if timestamp.count(":") == 1:
        time_format = "%M:%S"
    elif timestamp.count(":") == 2:
        time_format = "%H:%M:%S"
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

    if "." in timestamp:
        time_format += ".%f"
    dt = datetime.strptime(timestamp, time_format)
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


if __name__ == '__main__':
    main()
