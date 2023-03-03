import lmdb
import os
import cv2
import argparse
import pickle
import decord
from tqdm import tqdm
import numpy as np
from glob import glob
from joblib import delayed, Parallel
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.process_data.process_video import get_video_frames_cv, compute_TVL1
from sts.motion_sts import compute_motion_boudary, motion_mag_downsample, zero_boundary
import pandas as pd


video_root_add = "../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS"


data_train = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_train.csv")
data_val = pd.read_csv("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_validation.csv")
fps_videos = pd.read_csv ("VideoMAE/dataset/Epic_kitchen/annotation/EPIC_100_video_info.csv")



IDs_train = []
fps_train = []
for i, item in data_train.iterrows():
    IDs_train.append(i)
    video_id = item["video_id"]
    fps_train.append(fps_videos[fps_videos["video_id"] == video_id]["fps"].values)

IDs_val = []
fps_val = []
for i, item in data_val.iterrows():
    IDs_val.append(i)
    video_id = item["video_id"]
    fps_val.append(fps_videos[fps_videos["video_id"] == video_id]["fps"].values)






save_root_add_motion_map ="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_motion_map_videos"
if not os.path.exists(save_root_add_motion_map):
    os.makedirs(save_root_add_motion_map+ "/" + "train")
    os.makedirs(save_root_add_motion_map+ "/" + "validation")

save_root_add_optical_flow ="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_optical_flow_videos"
if not os.path.exists(save_root_add_optical_flow):
    os.makedirs(save_root_add_optical_flow+ "/" + "train")
    os.makedirs(save_root_add_optical_flow+ "/" + "validation")


def create_lmdb_video_dataset_optical_flow(dataset, root_path, dst_path, workers=-1, quality=100, fps=[]):

    # videos = glob(os.path.join(root_path,'*/*'))
    videos = glob(os.path.join(root_path,'*.mp4'))
    print('begin')
    
    def make_video_optical_flow(video_path, dst_path, vid_fps):
        # vid_names = '/'.join(video_path.split('/')[-2:])
        vid_names = '/'.join(video_path.split('/')[-1:])
        dst_file = os.path.join(dst_path, vid_names.split('.')[0]+'.mp4')
        # os.makedirs(dst_file, exist_ok=True)
        
        # ###load rgb frames from lmdb. You can change the code to load it in another way
        # frames = []
        # env = lmdb.open(video_path, readonly=True)
        # txn = env.begin(write=False)
        # for k,v in txn.cursor():
        #     frame_decode = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) 
        #     frames.append(frame_decode)
        # env.close()

        # read video with decord
        vr = decord.VideoReader(video_path)
        frames_num = len(vr)
        frames = vr.get_batch(range(frames_num)).asnumpy()

        height, width, _ = frames[0].shape
        empty_img = 128 * np.ones((int(height),int(width),3)).astype(np.uint8)
        # extract flows
        flows = []
        for idx, frame in enumerate(frames):
            if idx == 0: 
                pre_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = compute_TVL1(pre_frame, frame_gray)
            # create flow frame with 3 channel
            flow_img = empty_img.copy()
            flow_img[:,:,0:2] = flow
            flows.append(flow_img)
            pre_frame = frame_gray

        # save flows
        # _, frame_byte = cv2.imencode('.jpg', flows[0],  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        # env = lmdb.open(dst_file, frame_byte.nbytes * len(flows) * 50)
        # frames_num = len(flows)

        # for i in range(frames_num):
        #     txn = env.begin(write=True)
        #     key = 'image_{:05d}.jpg'.format(i+1)
        #     flow_img = flows[i]
        #     _, frame_byte = cv2.imencode('.jpg', flow_img,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        #     txn.put(key.encode(), frame_byte)
        #     txn.commit()
        # with open(os.path.join(dst_file, 'split.txt'),'w') as f:
        #     f.write(str(frames_num))

        out = cv2.VideoWriter(filename=dst_file, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), frameSize=(width, height), fps=int(vid_fps.round()))
        for frame in flows:
            out.write(frame)
        out.release()
        
        # for i, flow in enumerate(flows):
        #     cv2.imwrite(f'./video{vid_names}_{i}_flow.png', flow)
    Parallel(n_jobs=workers)(delayed(make_video_optical_flow)(vp, dst_path, vid_fps) for (vp, vid_fps) in tqdm(zip(videos, fps), total=len(videos)))

    # Parallel(n_jobs=workers)(delayed(make_video_optical_flow)(vp, dst_path) for vp in tqdm(videos, total=len(videos)))


def create_lmdb_video_dataset_flow_mag(dataset, root_path, dst_path, workers=-1, ws=8, quality=100, fps=[]):
        # videos = glob(os.path.join(root_path,'*/*'))
    videos = glob(os.path.join(root_path,'*.mp4'))

    # # fileter videos with _opt in their name
    # videos = [v for v in videos if 'optical' in v]
    # videos = [v for v in videos if 'mag' not in v]

    print('begin')

    def make_video_flow_mag(video_path, dst_path, vid_fps):
        vid_names = '/'.join(video_path.split('/')[-1:])
        dst_file = os.path.join(dst_path, vid_names.split('.')[0]+'_mag'+'.mp4')
        # os.makedirs(dst_file, exist_ok=True)

        if dataset == 'kinetics': ws = 4
        else: ws = 8

        # load flows from lmdb. You can change the code to load it in another way
        # if not os.path.exists(os.path.join(video_path, 'data.mdb')): return
        # flows = []
        # env = lmdb.open(video_path, readonly=True)
        # txn = env.begin(write=False)
        # for k,v in txn.cursor():
        #     flow_decode = cv2.imdecode(np.frombuffer(v, np.uint8), cv2.IMREAD_COLOR) 
        #     flows.append(flow_decode)
        # env.close()
        # duration = len(flows)

        # read video with decord
        vr = decord.VideoReader(video_path)
        duration = len(vr)
        flows = vr.get_batch(range(duration)).asnumpy()

        # compute frame mag offline with a sliding window
        frame_mags = []
        # env_flow_mag = lmdb.open(dst_file, readonly=False, map_size=int(2e12))
        for idx in range(1, duration+1):
            # txn_flow_mag = env_flow_mag.begin(write=True)
            if ws == 1:
                flow_clip = [flows[idx-1]]
            else:
                if idx - ws//2 >= 0 and idx + ws//2 <= duration:
                    flow_clip = flows[idx - ws//2 : idx + ws//2]
                elif idx - ws//2 >= 0 and idx + ws//2 > duration:
                    flow_clip = flows[-ws:]
                elif idx + ws//2 <= duration and idx - ws//2 < 0:
                    flow_clip = flows[:ws]
                else:
                    flow_clip = flows[:]

            height, width, _ = flow_clip[0].shape

            # flows_u = list([cv2.resize(flow[:,:,0], (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32) for flow in flow_clip])
            # flows_v = list([cv2.resize(flow[:,:,1], (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32) for flow in flow_clip])

            flows_u = list([flow[:,:,0].astype(np.float32) for flow in flow_clip])
            flows_v = list([flow[:,:,1].astype(np.float32) for flow in flow_clip])

            _, _, mb_x_u, mb_y_u = compute_motion_boudary(flows_u)
            _, _, mb_x_v, mb_y_v = compute_motion_boudary(flows_v)

            frame_mag_u, _ = cv2.cartToPolar(mb_x_u, mb_y_u, angleInDegrees=True)
            frame_mag_v, _ = cv2.cartToPolar(mb_x_v, mb_y_v, angleInDegrees=True)
            frame_mag = (frame_mag_u + frame_mag_v) / 2
            
            # zero boundary
            frame_mag = zero_boundary(frame_mag)

            # cast to RGB by repeating the same channel
            frame_mag = np.repeat(frame_mag[:,:,np.newaxis], 3, axis=2)

            # # downsample to match the fearture size of backbone output
            # if ws == 1:
            #     frame_mag_down = (motion_mag_downsample(frame_mag, 7, 224) * 5).astype(np.uint8)
            # else:
            #     frame_mag_down = motion_mag_downsample(frame_mag, 7, 224).astype(np.uint8)

            # save frame mag
            # key = 'image_{:05d}.jpg'.format(idx)
            # _, frame_byte = cv2.imencode('.jpg', frame_mag_down,  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            # txn_flow_mag.put(key.encode(), frame_byte)
            # txn_flow_mag.commit()

            frame_mags.append(frame_mag)

        height, width = frame_mags[0].shape[:2]

        # # plot a histogram of frame mag with a bounded range
        # import matplotlib.pyplot as plt
        # plt.hist(frame_mags[0].flatten(), bins=20, range=(0, 1.1))
        # plt.savefig('./hist.png')
        # plt.close()

        # remove values below 1 and replace them with 0
        # frame_mags_cliped = []
        # for frame in frame_mags:
        #     new_frame = frame.copy()
        #     new_frame[new_frame<20] = 0
        #     frame_mags_cliped.append(new_frame)
        frame_mags_cliped = frame_mags

        # normalize each frame based on the max value in the each frame
        frame_mags_normalized = [frame.astype(np.uint8) for frame in frame_mags_cliped]

        # #save frame mag
        # for i, frame in enumerate(frame_mags_normalized):
        #     cv2.imwrite(f'./video_{vid_names}_{i}.png', frame)


        # save video
        out = cv2.VideoWriter(filename=dst_file, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), frameSize=(width, height), fps=int(vid_fps.round()))
        for frame in frame_mags_normalized:
            out.write(frame)
        out.release()

        # for i, flow in enumerate(flows):
        #     cv2.imwrite(f'./{i}.png', flow)
        
        # with open(os.path.join(dst_file, 'split.txt'),'w') as f:
        #     f.write(str(duration))
    
    Parallel(n_jobs=workers)(delayed(make_video_flow_mag)(vp, dst_path, vid_fps) for (vp, vid_fps) in tqdm(zip(videos, fps), total=len(videos)))

    # Parallel(n_jobs=workers)(delayed(make_video_flow_mag)(vp, dst_path) for vp in tqdm(videos, total=len(videos)))



def parse_option():
    parser = argparse.ArgumentParser('training')

    # dataset
    parser.add_argument('--root-path-train', type=str, default="./", help='path of original data')
    parser.add_argument('--root-path-val', type=str, default="../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_videos/validation")
    parser.add_argument('--dst-path-train', type=str, default='./', help='path to store generated data')
    parser.add_argument('--dst-path-val', type=str, default='../../mnt/welles/scratch/datasets/Epic-kitchen/EPIC-KITCHENS/EPIC_100_action_recognition/mp4_optical_flow_videos/validation', help='path to store generated data')
    parser.add_argument('--dataset', type=str, default='Epic_Kitchens', choices=['Epic_Kitchens','kinetics','ucf101'], help='dataset to training')
    parser.add_argument('--data-type', type=str, default='mag', choices=['rgb','flow','mag','clip-mag'], help='which data')
    parser.add_argument('--video-type', type=str, default='mp4', choices=['mp4', 'avi'], help='which data')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--clip-length', type=int, default=16, help='num of clip length')
    parser.add_argument('--num-steps', type=int, default=2, help='num of sampling steps')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args =  parse_option()

    if args.data_type == 'flow':
        create_lmdb_video_dataset_optical_flow(args.dataset, args.root_path_train, args.dst_path_train, workers=args.num_workers, fps = fps_train)
        # create_lmdb_video_dataset_optical_flow(args.dataset, args.root_path_val, args.dst_path_val, workers=args.num_workers, fps = fps_val)
    elif args.data_type == 'mag':
        create_lmdb_video_dataset_flow_mag(args.dataset, args.root_path_train, args.dst_path_train, workers=args.num_workers, fps = fps_train)
        # for (dataset,u_pth, v_path, start_frame, stop_frame, fps, j, flag, root_path, dst_path, workers, ws, quality) in zip([args.dataset]*len(IDs_train), frames_u_path_train, frames_v_path_train, start_frames_train, stop_frames_train, fps_train, IDs_train, ["train"]*len(IDs_train), [args.root_path]*len(IDs_train), [args.dst_path_train]*len(IDs_train), [-1]*len(IDs_train), [8]*len(IDs_train), [100]*len(IDs_train),):
        #     create_lmdb_video_dataset_flow_mag(dataset,u_pth, v_path, start_frame, stop_frame, fps, j, flag, root_path, dst_path, workers, ws, quality)
