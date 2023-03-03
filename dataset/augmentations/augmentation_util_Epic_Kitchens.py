from PIL import Image
import numpy as np
import torch
import cv2
import torchvision.transforms.functional as F
import random
import io
import math

def pil_from_raw_rgb(raw):
    return Image.open(io.BytesIO(raw)).convert('RGB')

def pil_from_raw_rgba(raw):
    return Image.open(io.BytesIO(raw)).convert('L')

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def calc_over_lab(z1, z2):
    z_min = max(z1[0], z2[0])
    z_max = min(z1[1], z2[1])
    if z_min >= z_max:
        return 0
    else:
        return (z_max - z_min) / (z1[1] - z1[0])



def get_cor(clip_mag, height, width, base_ratio=0.8, t=0.1):
    if isinstance(clip_mag, list):
        clip_mag = sum(clip_mag) / len(clip_mag)
    if len(clip_mag.shape) == 3:
        clip_mag = clip_mag[:,:,0]

    h, w = clip_mag.shape
    flat_mag = clip_mag.reshape(-1)
    idx = np.argsort(flat_mag)[::-1][:math.floor(h*w*t)]
    row = (idx // w) + 1
    col = (idx % w) + 1

    row_s = [row.min() - 1, row.max()]
    col_s = [col.min() - 1, col.max()]

    # c represent the area of the region with high motion
    c = (row_s[1] - row_s[0]) * (col_s[1] - col_s[0])

    # for different c, we would make minor change on base_ratio
    if   c <=  4: crop_ratio = base_ratio + 0.1
    elif c <=  8: crop_ratio = base_ratio 
    elif c <= 12: crop_ratio = base_ratio - 0.1
    else:         crop_ratio = 0

    row_s = list(map(lambda x: x * height / 7, row_s))
    col_s = list(map(lambda x: x * width / 7, col_s))

    return row_s, col_s, crop_ratio


def crop_from_corners(row_s,col_s, height, width):

    if random.randint(0, 1) == 0:
        i_h = random.randint(0, int(row_s[0]) + 1)
        i_w = random.randint(0, int(col_s[0]) + 1)
        w = col_s[1] - i_w
        h = row_s[1] - i_h
        return i_h, i_w, h, w
    else:
        i_h = random.randint(int(row_s[1] - 1), height)
        i_w = random.randint(int(col_s[1] - 1), width)
        w = i_w - col_s[0]
        h = i_h - row_s[0]
        return row_s[0], col_s[0], h, w


#######################
from glob import glob
import os
import decord


# root_path_org = 'MCL-Motion-Focused-Contrastive-Learning/data'
root_path_org = './'

#fuction for visualizing image with bounding box
def visual_bbx (images, bboxes, fps):
    """
    images (torch.Tensor or np.array): list of images in torch or numpy type.
    bboxes (List[List]): list of list having bounding boxes in [x1, y1, x2, y2]
    """
    videos = glob(os.path.join(root_path_org,'*.mp4'))[0]
    vr = decord.VideoReader(videos)
    images = vr.get_batch(range(len(vr))).asnumpy()

    
    color_list = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0, 255,255)]
    # if not os.path.exists('VideoMAE/scripts/data/Epic-kitchen/visual_bbx'):
    #     os.makedirs('VideoMAE/scripts/data/Epic-kitchen/visual_bbx')
    for i, (img, bbx) in enumerate(zip(images, bboxes)):
        if isinstance(img, Image.Image) or isinstance(img, np.ndarray):
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frame = frame / frame.max() * 255
            # (x1,y1,x2,y2) = (bbx[0], bbx[1], bbx[2], bbx[3])
        elif isinstance(img, torch.Tensor):
            frame = img.numpy().astype(np.uint8).transpose(1, 2, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ###change the range of the pixel value to 0-255 by devide max value of the pixel value
            frame = frame / frame.max() * 255
            # if len(bbx) != 0:
            #     (x1,y1,x2,y2) = (bbx[0][0], bbx[0][1], bbx[0][2], bbx[0][3])

        if len(bbx) != 0:
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_list[0], 4)
            ##:
                # cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color_list[0], 4)
            cv2.rectangle(frame, (int(bbx[0]), int(bbx[1])), (int(bbx[2]), int(bbx[3])), color_list[0], 4)
        cv2.imwrite(f'./{i}.png', frame)
        w , h = frame.shape[1], frame.shape[0]
            
    out = cv2.VideoWriter(filename=f'./video_{i}_visual_BB.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=int(fps.round()), frameSize=(w, h), isColor=True)
    for frame in images:
        out.write(np.array(frame)[:,:,::-1])
    out.release()




root_path = "./"
def get_cor_Epic(root_path,t=0.01):
    videos = glob(os.path.join(root_path,'*.mp4'))[0]
    print('begin')

    vid_name = '/'.join(videos.split('/')[-1:])

    dst_file = os.path.join(root_path, vid_name.split('.')[0]+'.mp4')
    os.makedirs(root_path, exist_ok=True)

    # read video with decord
    vr = decord.VideoReader(videos)
    frames_num = len(vr)
    frames = vr.get_batch(range(frames_num)).asnumpy()
    # if len(frames.shape) == 4:
    #     frames = frames[0,:,:,0]
    video_bboxs= []
    max_motion = []
    for i in range (frames.shape[0]):
        if len(frames[i].shape) == 3:
            frame = frames[i]
            frame = frame[:,:,0]
            max_motion.append(frame.max())
            h, w = frame.shape
            flat_mag = frame.reshape(-1)
            idx = np.argsort(flat_mag)[::-1][:math.floor(h*w*t)]
            row = (idx // w) + 1
            col = (idx % w) + 1

            row_s = [row.min() - 1, row.max()]
            col_s = [col.min() - 1, col.max()]

            # c represent the area of the region with high motion
            c = (row_s[1] - row_s[0]) * (col_s[1] - col_s[0])
            x_1 = col_s[0]
            y_1 = row_s[0]
            x_2 = col_s[1]
            y_2 = row_s[1]
            bbx = [x_1, y_1, x_2, y_2]
            video_bboxs.append(bbx)
    # find the index of the frame with the highest motion
    max_motion_index = max_motion.index(max(max_motion))
    # get the bounding box of the frame with the highest motion
    max_motion_bbx = video_bboxs[max_motion_index]
    video_bboxs_union = [max_motion_bbx]*len(video_bboxs)

    visual_bbx(root_path_org, video_bboxs_union, 50)

        
    return c

get_cor_Epic(root_path)

