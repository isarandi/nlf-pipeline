import argparse

import numpy as np
import poseviz
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import framepump
from nlf_pipeline.util.paths import INFERENCE_ROOT
from simplepyutils import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=str)
    parser.add_argument("--cam-suffix", type=str, default='')
    parser.add_argument(
        "--resolution",
        default=(640, 360),
        nargs=2,
        type=int,
        help="Rendering resolution, (default: (640, 360))",
    )
    parser.add_argument('--camera-view', action=spu_argparse.BoolAction)
    parser.add_argument('--display-frame', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--viz', action=spu_argparse.BoolAction)
    spu.initialize(parser)


def main():
    initialize()
    video_path = f'{INFERENCE_ROOT}/videos_in/{FLAGS.video_id}.mp4'
    camera_path = f'{INFERENCE_ROOT}/cameras/{FLAGS.video_id}{FLAGS.cam_suffix}.pkl'

    video = framepump.VideoFrames(video_path)
    video_imsize = video.imshape[::-1]
    FLAGS.resolution = spu.rounded_int_tuple(
        video_imsize
        * np.min(np.array(FLAGS.resolution, np.float32) / video_imsize.astype(np.float32)),
        2,
    )
    camera_displays = spu.load_pickle(camera_path)

    camera_type = 'original' if FLAGS.camera_view else 'free'
    with (
        poseviz.PoseViz(
            camera_type=camera_type, resolution=FLAGS.resolution, show_image=FLAGS.display_frame
        ) as viz,
    ):
        for camera_display, frame in zip(camera_displays, video):
            viz.update(frame=frame, camera=camera_display)


if __name__ == '__main__':
    main()
