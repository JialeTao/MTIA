import os
from tqdm import tqdm
import argparse
import numpy as np
import imageio
from imageio import imread
import random
## input path video of concatenated pngs format
## output all video frames of png format
parser = argparse.ArgumentParser()
parser.add_argument('--frame_path',help='path to the mp4 video folder', type=str)
parser.add_argument('--output_path', help='path to the output png frames', type=str)
parser.add_argument('--sample_portion', help='portion of sampling frames', type=float, default=1.0)
args = parser.parse_args()

frame_path = args.frame_path
output_path = args.output_path
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

frame_list = sorted(os.listdir(frame_path))
video_array = []
for frame in tqdm(frame_list):
    frame_file = os.path.join(frame_path, frame)
    img = imread(frame_file)
    video_array.append(img)

imageio.mimsave(output_path, video_array)
