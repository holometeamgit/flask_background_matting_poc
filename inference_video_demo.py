import argparse
import cv2
import torch
import os
import shutil

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

from model import MattingBase, MattingRefine
from dataset import VideoDataset, ZipDataset
from dataset import augmentation as A
from inference_utils import HomographicAlignment

# parser = argparse.ArgumentParser(description='Inference video')
# args = parser.parse_args()

class VideoWriter:
    def __init__(self, path, frame_rate, width, height):
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    def add_batch(self, frames):
        frames = frames.mul(255).byte()
        frames = frames.cpu().permute(0, 2, 3, 1).numpy()
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def process(args):
    args.device = "cpu"
    device = torch.device(args.device)

    # Load model
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            0.7,
            3)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    # Load video and background
    vid = VideoDataset(args.video_src)
    bgr = [Image.open(args.video_bgr).convert('RGB')]
    dataset = ZipDataset([vid, bgr], transforms=A.PairCompose([
        A.PairApply(T.Resize(args.video_resize[::-1]) if args.video_resize else nn.Identity()),
        HomographicAlignment() if args.preprocess_alignment else A.PairApply(nn.Identity()),
        A.PairApply(T.ToTensor())
    ]))

    # Create output directory
    if os.path.exists(args.output_dir):
        if input(f'Directory {args.output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
            shutil.rmtree(args.output_dir)
        else:
            exit()
    os.makedirs(args.output_dir)

    # Prepare writers
    h = args.video_resize[1] if args.video_resize is not None else vid.height
    w = args.video_resize[0] if args.video_resize is not None else vid.width
    if 'com' in args.output_types:
        com_writer = VideoWriter(os.path.join(args.output_dir, 'com.mp4'), vid.frame_rate, w, h)

    # Conversion loop
    with torch.no_grad():
        for input_batch in tqdm(DataLoader(dataset, batch_size=1, pin_memory=True)):
            src, bgr = input_batch
            tgt_bgr = torch.tensor([120 / 255, 255 / 255, 155 / 255], device=device).view(1, 3, 1, 1)
            src = src.to(device, non_blocking=True)
            bgr = bgr.to(device, non_blocking=True)

            if args.model_type == 'mattingbase':
                pha, fgr, err, _ = model(src, bgr)
            elif args.model_type == 'mattingrefine':
                pha, fgr, _, _, err, ref = model(src, bgr)
            elif args.model_type == 'mattingbm':
                pha, fgr = model(src, bgr)

            if 'com' in args.output_types:
                # Output composite with green background
                com = fgr * pha + tgt_bgr * (1 - pha)
                com_writer.add_batch(com)


def process_from_server(video_src_filename, video_bgr_filename, output_dir):
    parser = argparse.ArgumentParser(description='Inference video')

    parser.add_argument('run', type=str)
    parser.add_argument('--model-type', type=str, choices=['mattingbase', 'mattingrefine'])
    parser.add_argument('--model-backbone', type=str, choices=['resnet101', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    parser.add_argument('--model-checkpoint', type=str)
    parser.add_argument('--model-refine-mode', type=str, default='sampling',
                        choices=['full', 'sampling', 'thresholding'])
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)

    parser.add_argument('--video-src', type=str)
    parser.add_argument('--video-bgr', type=str)
    parser.add_argument('--video-resize', type=int, default=None, nargs=2)

    parser.add_argument('--preprocess-alignment', action='store_true')

    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--output-types', type=str, nargs='+', choices=['com'])

    args = parser.parse_args()

    args.device = "cpu"
    args.model_type = "mattingrefine"
    args.model_backbone = "resnet50"
    args.model_backbone_scale = 0.25
    args.model_refine_mode = "full"
    args.model_checkpoint = "content/pytorch_resnet50.pth"
    args.video_src = "content/" + video_src_filename
    args.video_bgr = "content/" + video_bgr_filename
    args.output_dir = "content/" + output_dir
    args.output_types = ['com']

    process(args)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Inference video')
#
#     parser.add_argument('--model-type', type=str, choices=['mattingbase', 'mattingrefine'])
#     parser.add_argument('--model-backbone', type=str, choices=['resnet101', 'resnet50', 'mobilenetv2'])
#     parser.add_argument('--model-backbone-scale', type=float, default=0.25)
#     parser.add_argument('--model-checkpoint', type=str)
#     parser.add_argument('--model-refine-mode', type=str, default='sampling',
#                         choices=['full', 'sampling', 'thresholding'])
#     parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
#
#     parser.add_argument('--video-src', type=str)
#     parser.add_argument('--video-bgr', type=str)
#     parser.add_argument('--video-resize', type=int, default=None, nargs=2)
#
#     parser.add_argument('--preprocess-alignment', action='store_true')
#
#     parser.add_argument('--output-dir', type=str)
#     parser.add_argument('--output-types', type=str, nargs='+', choices=['com'])
#
#     args = parser.parse_args()
#
#     process()
