import argparse
import os
import os.path as osp

import cv2
import numpy as np
import rlemasklib
import simplepyutils as spu
import simplepyutils.argparse as spu_argparse
import torch
import framepump
from simplepyutils import FLAGS
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--file-pattern', type=str, default='**/*.mp4')
    parser.add_argument('--videos-per-task', type=int, default=1)
    parser.add_argument('--viz', action=spu_argparse.BoolAction)
    parser.add_argument('--write-video', action=spu_argparse.BoolAction)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--mask-threshold', type=float, default=0.2)
    parser.add_argument('--skip-existing', action=spu_argparse.BoolAction, default=True)
    spu.initialize(parser)

    relpaths, video_paths, output_paths = get_video_paths()
    if FLAGS.skip_existing and all(spu.is_pickle_readable(p) for p in output_paths):
        print('All done')
        return

    weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = torch.jit.script(deeplabv3_resnet101(weights=weights).eval().cuda())
    person_class_id = 15

    for video_path, output_path in zip(video_paths, output_paths):
        video = framepump.VideoFrames(video_path, dtype=np.float32)
        frame_batches = torch.utils.data.DataLoader(
            VideoDataset(video, transform=weights.transforms()),
            num_workers=1,
            batch_size=FLAGS.batch_size,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )

        if FLAGS.skip_existing and spu.is_pickle_readable(output_path):
            continue

        preds = []
        for frame_batch in spu.progressbar(
            frame_batches, total=len(video), step=FLAGS.batch_size, desc='Segmenting'
        ):
            probs = model(frame_batch.cuda(non_blocking=True))['out'].softmax(dim=1)
            mask_batch_np = (probs[:, person_class_id] >= FLAGS.mask_threshold).cpu().numpy()
            preds.extend(rlemasklib.encode(mask_batch_np, batch_first=True, zlevel=-1))

            if FLAGS.viz:
                for predframe in mask_batch_np:
                    cv2.imshow('frame', predframe.view(np.uint8) * 255)
                    cv2.waitKey(1)

        spu.dump_pickle(preds, output_path)


def get_video_paths():
    globs = [
        spu.sorted_recursive_glob(f'{FLAGS.video_dir}/{p}') for p in FLAGS.file_pattern.split(',')
    ]
    video_paths = [x for l in globs for x in l]
    try:
        i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
        i_start = i_task * FLAGS.videos_per_task
        video_paths = video_paths[i_start: i_start + FLAGS.videos_per_task]
    except KeyError:
        pass
    relpaths = [osp.relpath(video_path, FLAGS.video_dir) for video_path in video_paths]
    output_paths = [
        f'{FLAGS.output_dir}/{osp.splitext(relpath)[0]}_masks.pkl' for relpath in relpaths
    ]
    return relpaths, video_paths, output_paths


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, video, transform):
        self.video = video
        self.transform = transform

    def __iter__(self):
        for frame in self.video:
            yield self.transform(torch.from_numpy(frame).permute(2, 0, 1))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
