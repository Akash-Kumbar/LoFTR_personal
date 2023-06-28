import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import argparse
import matplotlib.pyplot as plt
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg

parser = argparse.ArgumentParser(description='LoFTR feature matching infer')
parser.add_argument('--source_image', type=str, default='', help='source image with which keypoints need to be detected')
parser.add_argument('--other_image', type=str, default='', help='the other image with which keypoints are matched')
parser.add_argument('--save_image_path', type=str, default='out_img.png')
parser.add_argument('--save_feature_txt', type=str, default='out_fts.txt')
args = parser.parse_args()

## Check the type of setting in which image is captured
image_type = 'outdoor' #indoor

image_pair = [args.source_image, args.other_image]

matcher = LoFTR(config=default_cfg)
if image_type == 'indoor':
  matcher.load_state_dict(torch.load("weights/indoor_ds.ckpt")['state_dict'])
elif image_type == 'outdoor':
  matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
else:
  raise ValueError("Wrong image_type is given.")
matcher = matcher.eval().cuda()

img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

# Draw 
color = cm.jet(mconf, alpha=0.7)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
print(mkpts0.shape)
print(mkpts1.shape)
# exit()
print(text[1])
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text)
make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text, path=args.save_image_path)

file = open(args.save_feature_text, 'w')
for i in range(mkpts0.shape[0]):
    file.write(f'Keypoint 1 idx: {mkpts0[i]}, Keypoint 2 idx : {mkpts1[i]}')