import cv2
import os
import shutil

import cv2
import torch
from PIL import Image
from flask import url_for
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from dataset import VideoDataset, ZipDataset
from dataset import augmentation as A
from inference_utils import HomographicAlignment
from model import MattingBase, MattingRefine

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

def matching_vid_bgr_size(vid, bgr):
    video_cap = cv2.VideoCapture(vid)
    vid_w = 0
    vid_h = 0
    if video_cap.isOpened():
        vid_w = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_h = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("Video width {} height {}".format(vid_w, vid_h))

    video_cap.release()

    bgr_w, bgr_h = bgr.size

    result = bgr.resize((int(vid_w), int(vid_h)), Image.ANTIALIAS)
    print("Image width {} height {}".format(bgr_w, bgr_h))
    return result

def process(device="cpu",
            model_type='mattingrefine',
            model_backbone="resnet50",
            model_backbone_scale=0.25,
            model_refine_mode="full",
            model_refine_sample_pixels=80_000,
            model_checkpoint="content/pytorch_resnet50.pth",
            video_src="content/out_1.mp4",
            video_bgr="content/bg_1.jpg",
            video_resize=None,
            preprocess_alignment=None,
            output_dir="content/output_1",
            output_types=['com'],
            server_uri="localhost"
            ):
    """Main method for video bg removal process.

    Args:
      device - the device where the computation takes place (cpu/gpu),
      model_type - computation model type,
      model_backbone - feature extraction,
      model_backbone_scale - none,
      model_refine_mode - none,
      model_refine_sample_pixels - none,
      model_checkpoin - model file path,
      video_src - video file path,
      video_bgr - background image file path,
      video_resize - none,
      preprocess_alignment - none,
      output_dir - folder where the result will be saved ,
      output_types - the types of results to be calculated ,
      server_uri - uri of the server for the link to download the result

    Returns:
      before calculation - the number of rendered frames
      after - a link to the file for download 
    """

    device = torch.device(device)

    # Load model
    if model_type == 'mattingbase':
        model = MattingBase(model_backbone)
    if model_type == 'mattingrefine':
        model = MattingRefine(
            model_backbone,
            model_backbone_scale,
            model_refine_mode,
            model_refine_sample_pixels,
            0.7,
            3)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device), strict=False)

    # Load video and background
    vid = VideoDataset(video_src)
    bgr = [Image.open(video_bgr).convert('RGB')]

    bgr_resized = [matching_vid_bgr_size(video_src, bgr[0])]
    bgr = bgr_resized

    dataset = ZipDataset([vid, bgr], transforms=A.PairCompose([
        A.PairApply(T.Resize(video_resize[::-1]) if video_resize else nn.Identity()),
        HomographicAlignment() if preprocess_alignment else A.PairApply(nn.Identity()),
        A.PairApply(T.ToTensor())
    ]))

    # Create output directory
    if os.path.exists(output_dir):
        if True: # #OLD CODE WAS# input(f'Directory {output_dir} already exists. Override? [Y/N]: ').lower() == 'y':
            shutil.rmtree(output_dir)
        else:
            exit()
    os.makedirs(output_dir)

    # Prepare writers
    h = video_resize[1] if video_resize is not None else vid.height
    w = video_resize[0] if video_resize is not None else vid.width
    if 'com' in output_types:
        com_writer = VideoWriter(os.path.join(output_dir, 'com.mp4'), vid.frame_rate, w, h)

    #yield url_for('static', filename=os.path.join(output_dir, 'com.mp4'))

    # Conversion loop
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)
        total = len(dataloader)
        i = 0
        for input_batch in tqdm(dataloader):
            src, bgr = input_batch
            tgt_bgr = torch.tensor([120 / 255, 255 / 255, 155 / 255], device=device).view(1, 3, 1, 1)
            src = src.to(device, non_blocking=True)
            bgr = bgr.to(device, non_blocking=True)

            if model_type == 'mattingbase':
                pha, fgr, err, _ = model(src, bgr)
            elif model_type == 'mattingrefine':
                pha, fgr, _, _, err, ref = model(src, bgr)
            elif model_type == 'mattingbm':
                pha, fgr = model(src, bgr)

            if 'com' in output_types:
                # Output composite with green background
                com = fgr * pha + tgt_bgr * (1 - pha)
                com_writer.add_batch(com)

            i = i + 1
            # return frame process count back to server
            yield str(i) + "/" + str(total)

    # return result file url that can be download
    yield "file url:  " + server_uri + os.path.join(output_dir, 'com.mp4')