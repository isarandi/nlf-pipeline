from vtkmodules.vtkIOImage import *

"""separator"""
import argparse
import contextlib
import datetime
import functools
import itertools
import os
import os.path as osp

import kornia.enhance.equalization
import bodycompress
import cameravision
import cv2
import more_itertools
import numpy as np
import poseviz
import rlemasklib
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import smplfitter.np
import smplfitter.pt
import torch

# noinspection PyUnresolvedReferences
import torchvision
import framepump
from nlf_pipeline.util.paths import DATA_ROOT
from simplepyutils import FLAGS


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='https://bit.ly/nlf_l_pt')
    parser.add_argument('--video-dir', type=str)
    parser.add_argument('--mask-dir', type=str)
    parser.add_argument('--file-pattern', type=str, default='**/*.mp4')
    parser.add_argument('--paths-file', type=str)
    parser.add_argument('--ignore-paths-file', type=str)
    parser.add_argument('--videos-per-task', type=int, default=1)
    parser.add_argument('--viz', action=spu_argparse.BoolAction)
    parser.add_argument('--viz-downscale', type=int, default=1)
    parser.add_argument('--high-quality-viz', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--write-video', action=spu_argparse.BoolAction)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--output-video-dir', type=str)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--camera-intrinsics-file', type=str)
    parser.add_argument('--camera-file', type=str)
    parser.add_argument('--camera-dir', type=str)
    parser.add_argument('--fov', type=float, default=55)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--internal-batch-size', type=int, default=128)
    parser.add_argument('--max-detections', type=int, default=-1)
    parser.add_argument('--detector-threshold', type=float, default=0.2)
    parser.add_argument('--detector-nms-iou-threshold', type=float, default=0.7)
    parser.add_argument('--detector-flip-aug', action=spu_argparse.BoolAction)
    parser.add_argument('--detector-both-flip-aug', action=spu_argparse.BoolAction)
    parser.add_argument('--antialias-factor', type=int, default=1)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--suppress-implausible-poses', action=spu_argparse.BoolAction)
    parser.add_argument('--skip-existing', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--constant-framerate', action=spu_argparse.BoolAction, default=True)
    parser.add_argument('--clahe', action=spu_argparse.BoolAction)
    parser.add_argument('--use-detector-even-with-masks', action=spu_argparse.BoolAction)
    parser.add_argument('--body-model', type=str, default='smpl')
    parser.add_argument('--mask-suffix', type=str, default='')
    parser.add_argument('--downscale-factor', type=int, default=1)
    spu.initialize(parser)
    cv2.setNumThreads(1)
    if FLAGS.output_video_dir is None:
        FLAGS.output_video_dir = FLAGS.output_dir


def main():
    initialize()
    video_relpaths = get_paths()
    grouped = spu.groupby(video_relpaths, osp.dirname)
    dirnames = list(grouped.keys())
    video_relpaths = sum((grouped[d] for d in dirnames), [])
    camera_tracks = get_camera_calibrations(video_relpaths)

    output_paths = [
        f'{FLAGS.output_dir}/{osp.splitext(relpath)[0]}{FLAGS.suffix}.xz'
        for relpath in video_relpaths
    ]

    if FLAGS.skip_existing and all(osp.exists(p) for p in output_paths):
        print('All outputs already exist')
        return

    output_video_paths = [
        f'{FLAGS.output_video_dir}/{osp.splitext(relpath)[0]}{FLAGS.suffix}.mp4'
        for relpath in video_relpaths
    ]
    mask_paths = [
        f'{FLAGS.mask_dir}/{osp.splitext(relpath)[0]}_masks{FLAGS.mask_suffix}.pkl'
        for relpath in video_relpaths
    ]

    vertex_subset = None
    cano_verts = np.load(f'{DATA_ROOT}/nlf/canonical_verts/{FLAGS.body_model}.npy')
    if vertex_subset is not None:
        cano_verts = cano_verts[vertex_subset]
    cano_joints = np.load(f'{DATA_ROOT}/nlf/canonical_joints/{FLAGS.body_model}.npy')
    cano_all = torch.cat([torch.as_tensor(cano_verts), torch.as_tensor(cano_joints)], dim=0).to(
        dtype=torch.float32, device='cuda'
    )
    bm = smplfitter.pt.get_cached_body_model(FLAGS.body_model, 'neutral')
    model = get_model().cuda()

    if FLAGS.mask_dir and not FLAGS.use_detector_even_with_masks:
        predict_fn = functools.partial(
            model.estimate_poses_batched,
            default_fov_degrees=FLAGS.fov,
            num_aug=FLAGS.num_aug,
            internal_batch_size=FLAGS.internal_batch_size,
            antialias_factor=FLAGS.antialias_factor,
            weights=model.get_weights_for_canonical_points(cano_all),
        )
    else:
        predict_fn = functools.partial(
            model.detect_poses_batched,
            default_fov_degrees=FLAGS.fov,
            detector_threshold=FLAGS.detector_threshold,
            num_aug=FLAGS.num_aug,
            detector_nms_iou_threshold=FLAGS.detector_nms_iou_threshold,
            max_detections=FLAGS.max_detections,
            internal_batch_size=FLAGS.internal_batch_size,
            antialias_factor=FLAGS.antialias_factor,
            suppress_implausible_poses=FLAGS.suppress_implausible_poses,
            detector_flip_aug=FLAGS.detector_flip_aug,
            detector_both_flip_aug=FLAGS.detector_both_flip_aug,
            weights=model.get_weights_for_canonical_points(cano_all),
        )

    first_camera, camera_tracks = spu.nested_spy_first(camera_tracks, levels=2)
    up = first_camera.world_up if first_camera is not None else np.array((0, -1, 0), np.float32)
    viz = (
        poseviz.PoseViz(
            camera_type='original',
            world_up=up,
            high_quality=FLAGS.high_quality_viz,
            downscale=FLAGS.viz_downscale,
            show_ground_plane=True,
            show_field_of_view=True,
            resolution=(1920, 1080) if FLAGS.high_quality_viz else (1280, 720),
            camera_view_padding=0.2,
            show_camera_wireframe=True,
            draw_detections=True,
            body_model_faces=bm.faces,
            queue_size=8,
        )
        if FLAGS.viz
        else contextlib.nullcontext()
    )

    with viz:
        for video_relpath, output_path, output_video_path, mask_path, camera_track in zip(
            video_relpaths, output_paths, output_video_paths, mask_paths, camera_tracks
        ):
            process_video(
                video_relpath,
                output_path,
                output_video_path,
                mask_path,
                camera_track,
                up,
                predict_fn,
                cano_verts,
                cano_joints,
                viz,
            )


def process_video(
    video_relpath,
    output_path,
    output_video_path,
    mask_path,
    camera_track,
    up,
    predict_fn,
    cano_verts,
    cano_joints,
    viz,
):
    if (
        FLAGS.skip_existing
        and os.path.isfile(output_path)
        and (not FLAGS.write_video or osp.exists(output_video_path))
    ):
        print(f'Already done {output_path}')
        return
    video = framepump.VideoFrames(
        osp.join(FLAGS.video_dir, video_relpath),
        dtype=np.uint16,
        constant_framerate=FLAGS.constant_framerate,
    )
    if FLAGS.downscale_factor:
        new_imshape = np.array(video.imshape) // FLAGS.downscale_factor
        video = video.resized(new_imshape)

    frame_bs = torch.utils.data.DataLoader(
        VideoDataset(video),  # , transform=improc.Clahe() if FLAGS.clahe else None),
        num_workers=1,
        batch_size=FLAGS.batch_size,
        pin_memory=True,
        prefetch_factor=3,
    )

    camera_bs = more_itertools.chunked(camera_track, FLAGS.batch_size)
    mask_bs = (
        more_itertools.chunked(spu.load_pickle(mask_path), FLAGS.batch_size)
        if FLAGS.mask_dir
        else itertools.repeat(None)
    )

    if FLAGS.viz:
        viz.reinit_camera_view()
        if FLAGS.write_video:
            viz.new_sequence_output(output_video_path, fps=video.fps, audio_source_path=video.path)

    spu.ensure_parent_dir_exists(output_path)
    n_frames = len(video)
    metadata = dict(
        video_path=video_relpath,
        world_up=up,
        config=vars(FLAGS),
        created_time=datetime.datetime.now().isoformat(),
        n_frames=n_frames,
    )

    frame_bs = spu.progressbar(frame_bs, total=n_frames, unit='frame', step=FLAGS.batch_size)
    i_frame = 0
    with bodycompress.BodyCompressor(output_path, metadata) as f:
        for (frame_b_cpu, frame_b_gpu), camera_b, mask_b in zip(
            precuda(frame_bs), camera_bs, mask_bs
        ):
            default_cam = cameravision.Camera.from_fov(FLAGS.fov, frame_b_cpu.shape[1:3])
            default_cam.world_up = up

            batch_size = frame_b_cpu.shape[0]
            camera_b = [
                (
                    c.scale_output(1 / FLAGS.downscale_factor, inplace=False)
                    if c is not None
                    else default_cam
                )
                for c in camera_b
            ][:batch_size]

            intrinsics = torch.as_tensor(np.stack([c.intrinsic_matrix for c in camera_b]))
            extrinsics = torch.as_tensor(np.stack([c.get_extrinsic_matrix() for c in camera_b]))
            distortion_coeffs = torch.as_tensor(
                np.stack([c.get_distortion_coeffs() for c in camera_b])
            )

            if FLAGS.mask_dir:
                mask_based_boxes_b = mask_to_bbox_batched(
                    mask_b,
                    frame_b_gpu.shape[2:],
                    score=0.85 if FLAGS.use_detector_even_with_masks else None,
                )
            else:
                mask_based_boxes_b = None

            if not FLAGS.mask_dir or FLAGS.use_detector_even_with_masks:
                pred = predict_fn(
                    frame_b_gpu,
                    extra_boxes=mask_based_boxes_b,
                    intrinsic_matrix=intrinsics,
                    extrinsic_matrix=extrinsics,
                    distortion_coeffs=distortion_coeffs,
                    world_up_vector=torch.tensor(up, dtype=torch.float32),
                )
                boxes_b = pred['boxes']
            else:
                pred = predict_fn(
                    frame_b_gpu,
                    mask_based_boxes_b,
                    intrinsic_matrix=intrinsics,
                    extrinsic_matrix=extrinsics,
                    distortion_coeffs=distortion_coeffs,
                    world_up_vector=torch.tensor(up, dtype=torch.float32),
                )
                boxes_b = mask_based_boxes_b

            verts_b, joints_b = ragged_split(
                pred['poses3d'], [cano_verts.shape[0], cano_joints.shape[0]], dim=-2
            )

            uncerts_verts_b, uncerts_joints_b = ragged_split(
                pred['uncertainties'], [cano_verts.shape[0], cano_joints.shape[0]], dim=-1
            )

            for frame_cpu, boxes, joints, vertices, uncerts_verts, uncerts_joints, camera in zip(
                frame_b_cpu,
                boxes_b,
                joints_b,
                verts_b,
                uncerts_verts_b,
                uncerts_joints_b,
                camera_b,
            ):

                vertices_np = to_np(vertices)
                joints_np = to_np(joints)
                boxes_np = to_np(boxes)
                f.append(
                    boxes=boxes_np * FLAGS.downscale_factor,
                    joints=joints_np,
                    vertices=vertices_np,
                    joint_uncertainties=to_np(uncerts_joints),
                    vertex_uncertainties=to_np(uncerts_verts),
                    camera=camera.scale_output(FLAGS.downscale_factor, inplace=False),
                )

                if FLAGS.viz:
                    frame_viz = frame_cpu.numpy()
                    viz.update(frame_viz, boxes=boxes_np, vertices=vertices_np, camera=camera)
                i_frame += 1


def ragged_split(xs, sizes, dim):
    return zip(*[torch.split(x, sizes, dim=dim) for x in xs])


def precuda(cpu_batches):
    for cur_cpu_batch in cpu_batches:
        cur_gpu_batch = cur_cpu_batch.cuda(non_blocking=True).permute(0, 3, 1, 2)
        if cur_gpu_batch.dtype == torch.uint16:
            cur_gpu_batch = cur_gpu_batch.half().mul_(1.0 / 65536.0).nan_to_num_(posinf=1.0)
            cur_cpu_batch = cur_cpu_batch.view(torch.uint8)[..., 1::2]
            if FLAGS.viz:
                cur_cpu_batch = cur_cpu_batch.contiguous()
        elif cur_gpu_batch.dtype == torch.uint8:
            cur_gpu_batch = cur_gpu_batch.half().mul_(1.0 / 255.0)

        if FLAGS.clahe:
            yuv = kornia.color.rgb_to_yuv(cur_gpu_batch)
            y_clahe = kornia.enhance.equalization.equalize_clahe(
                yuv[:, 0], clip_limit=2.5, grid_size=(12, 12)
            )
            yuv[:, 0] = y_clahe
            cur_gpu_batch = kornia.color.yuv_to_rgb(yuv).clamp_(0, 1)

        yield cur_cpu_batch, cur_gpu_batch


def to_np(x):
    if isinstance(x, (list, tuple)):
        return [to_np(y) for y in x]
    return x.detach().cpu().numpy()


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, video, transform=None):
        self.video = video
        self.transform = transform if transform is not None else lambda x: x

    def __iter__(self):
        for frame in self.video:
            yield torch.from_numpy(self.transform(frame))


def mask_to_bbox_batched(masks_batch, imshape, score=0.95):
    boxes = [mask_to_bbox(m, imshape) for m in masks_batch]

    if score is None:
        return [torch.as_tensor(b) for b in boxes]

    return [
        torch.as_tensor(np.concatenate([b, np.full([len(b), 1], score, np.float32)], axis=1))
        for b in boxes
    ]


def mask_to_bbox(masks, imshape):
    if len(masks) == 0:
        return np.zeros([0, 4], np.float32)

    mask_size = np.array(masks[0]['size'][::-1], np.float32)
    imsize = np.array(imshape[:2][::-1], np.float32)
    boxes = rlemasklib.to_bbox(masks)
    boxes = (boxes.reshape(-1, 2) * (imsize / mask_size)).reshape(boxes.shape)
    is_nonempty = boxes[:, 2] * boxes[:, 3] > 0
    boxes = boxes[is_nonempty]
    return boxes


def get_paths():
    relpaths = get_relpaths(
        FLAGS.video_dir,
        FLAGS.paths_file,
        FLAGS.ignore_paths_file,
        FLAGS.file_pattern,
        FLAGS.videos_per_task,
        int(os.getenv('SLURM_ARRAY_TASK_ID', -1)),
    )

    if FLAGS.camera_file or FLAGS.camera_intrinsics_file:
        camera_dict = spu.load_pickle(
            FLAGS.camera_file if FLAGS.camera_file else FLAGS.camera_intrinsics_file
        )

        if isinstance(camera_dict, dict):
            relpaths = [p for p in relpaths if p in camera_dict]

    return relpaths


def get_relpaths(dirpath, paths_file, ignore_paths_file, file_pattern, files_per_task, i_task):
    if paths_file:
        relpaths = sorted([osp.normpath(p) for p in spu.read_lines(paths_file)])
    else:
        globs = [
            spu.sorted_recursive_glob(f'{dirpath}/{p}') for p in FLAGS.file_pattern.split(',')
        ]
        video_paths = sorted([x for l in globs for x in l])
        relpaths = [osp.relpath(video_path, dirpath) for video_path in video_paths]

    if ignore_paths_file:
        ignore_relpaths = set([osp.normpath(x) for x in spu.read_lines(ignore_paths_file)])
        relpaths = [p for p in relpaths if p not in ignore_relpaths]

    if i_task != -1:
        relpaths = relpaths[i_task * files_per_task : (i_task + 1) * files_per_task]

    return relpaths


def get_camera_calibrations(relpaths):
    if FLAGS.camera_dir:
        camera_files = [f'{FLAGS.camera_dir}/{osp.splitext(p)[0]}.pkl' for p in relpaths]
        return [get_camera_track(spu.load_pickle(f)) for f in camera_files]

    if FLAGS.camera_intrinsics_file:
        intrinsics = spu.load_pickle(FLAGS.camera_intrinsics_file)
        if isinstance(intrinsics, dict):
            intrinsics = [intrinsics[relpath] for relpath in relpaths]
        else:
            intrinsics = itertools.repeat(intrinsics)
        return (
            itertools.repeat(cameravision.Camera(intrinsic_matrix=intr, world_up=(0, -1, 0)))
            for intr in intrinsics
        )

    if FLAGS.camera_file:
        camera = spu.load_pickle(FLAGS.camera_file)
        if isinstance(camera, dict):
            return [get_camera_track(camera[relpath]) for relpath in relpaths]
        else:
            return itertools.repeat(get_camera_track(camera))

    return itertools.repeat(itertools.repeat(None))


def get_camera_track(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, cameravision.Camera):
        return itertools.repeat(x)
    else:
        raise Exception('Invalid camera file format')


def get_model():
    if FLAGS.model_path.startswith('https://') or FLAGS.model_path.startswith('http://'):
        cache_dir = torch.hub.get_dir()
        filename = osp.basename(FLAGS.model_path) + '.torchscript'
        path = osp.join(cache_dir, filename)
        if not osp.exists(path):
            torch.hub.download_url_to_file(FLAGS.model_path, path)
    else:
        path = FLAGS.model_path

    model = torch.jit.load(path).eval()

    def nop(*args, **kwargs):
        pass

    model.forward = nop
    return torch.jit.optimize_for_inference(
        model,
        other_methods=[
            'detect_poses_batched',
            'estimate_poses_batched',
            'get_weights_for_canonical_points',
        ],
    )


if __name__ == '__main__':
    with torch.inference_mode(), torch.device('cuda'):
        main()
