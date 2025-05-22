# This script uses the following model for frame interpolation:
# (those repos are not required to run this code, this is self-contained)
# https://github.com/hzwer/ECCV2022-RIFE
# https://github.com/hzwer/Practical-RIFE

import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import tqdm
import framepump
from nlf_pipeline.rife.IFNet_HDv3 import IFNet


def main():
    parser = argparse.ArgumentParser(description='Double the frame rate of a video')
    parser.add_argument('input_video', type=str, help='input video file')
    parser.add_argument('output_video', type=str, help='output video file')
    parser.add_argument('--viz', action='store_true')
    args = parser.parse_args()

    video = framepump.VideoFrames(args.input_video, dtype=np.float32)
    scale = 1.0 if video.imshape[0] < 1600 else 0.5

    with torch.inference_mode(), torch.device('cuda'), framepump.VideoWriter(
        args.output_video, fps=video.fps * 2, audio_source_path=args.input_video
    ) as video_writer:
        flownet = IFNet()
        flownet.load_from_file()
        flownet.eval()
        frame_loader = torch.utils.data.DataLoader(
            VideoDataset(video),
            num_workers=1,
            batch_size=1,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )

        frame_loader = iter(frame_loader)
        frame_prev = next(frame_loader)
        frame_prev_np = to_np(frame_prev, video.imshape)
        video_writer.append_data(frame_prev_np)
        frame_prev_cuda = frame_prev.cuda(non_blocking=True)

        for i_frame, frame in enumerate(tqdm.tqdm(frame_loader, total=len(video) - 1)):
            frame: torch.Tensor
            frame_cuda = frame.cuda(non_blocking=True)
            frame_mid_cuda = flownet(frame_prev_cuda, frame_cuda, timestep=0.5, scale=scale)

            frame_mid_np = to_np(frame_mid_cuda, video.imshape)
            frame_np = to_np(frame, video.imshape)

            video_writer.append_data(frame_mid_np)
            video_writer.append_data(frame_np)
            if args.viz:
                fr1 = cv2.cvtColor(frame_mid_np, cv2.COLOR_RGB2BGR)
                fr2 = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', fr1)
                cv2.waitKey(1)
                cv2.imshow('frame', fr2)
                cv2.waitKey(1)

            frame_prev_cuda = frame_cuda
        video_writer.append_data(frame_np)


def to_np(tensor, imshape):
    h, w = imshape
    return (tensor[:, :, :h, :w].permute(0, 2, 3, 1).cpu().numpy() * 65535.0).astype(np.uint16)[0]


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, video):
        self.video = video
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad((0, 0, -video.imshape[1] % 32, -video.imshape[0] % 32), fill=0),
            ]
        )

    def __iter__(self):
        yield from map(self.transform, self.video)


if __name__ == '__main__':
    main()
