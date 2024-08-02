import pdb

import cv2
import numpy as np
import os
import re
def get_number_of_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total

def write_frame_nums(dict_path, video_dir, filenames):
    file_num = dict()
    for filename in filenames:
        video_names = np.loadtxt(filename, dtype=str)
        video_names.sort()

        for video_idx, video_name in enumerate(video_names):
            print(video_idx, video_name)
            scene_name = video_name.split("_")[2]

            video_path = os.path.join(video_dir, f'Scene{scene_name}', video_name + ".mp4")
            num_frames = get_number_of_frames(video_path)
            type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)', video_name)[0]
            clip_id = type + clip_id
            file_num[scene_id+'_'+clip_id] = num_frames
    np.save(dict_path,file_num,allow_pickle=True)

def write_frame_level_ground_truth(output_folder_base, video_dir, filenames, is_abnormal=True):
    video_names = np.loadtxt(filenames, dtype=str)
    video_names.sort()

    for video_idx, video_name in enumerate(video_names):
        print(video_idx, video_name)
        scene_name = video_name.split("_")[2]

        video_path = os.path.join(video_dir, f'Scene{scene_name}', video_name + ".mp4")
        num_frames = get_number_of_frames(video_path)
        ground_truth = np.zeros(num_frames)

        # gt_path = os.path.join(output_folder_base, video_name, "ground_truth_frame_level.txt")
        gt_path = os.path.join(output_folder_base, video_name+'.npy')
        if is_abnormal:
            tracks_path = os.path.join(video_dir, f'Scene{scene_name}', video_name + "_annotations", video_name + "_tracks.txt")
            tracks_video = np.loadtxt(tracks_path, delimiter=",")
            if tracks_video.ndim == 1:
                tracks_video = [tracks_video]

            for track in tracks_video:
                ground_truth[int(track[1]): int(track[2] + 1)] = 1

            np.save(gt_path, ground_truth)

        else:
            np.save(gt_path, ground_truth)

# get all the video length of UB
def calculate_video_length(video_dir):
    # replace the path to your own path, downloaded from https://github.com/lilygeorgescu/UBnormal
    filename1 = "/mnt/hdd8T/hh/code/UBnormal-main/scripts/abnormal_training_video_names.txt"
    filename2 = "/mnt/hdd8T/hh/code/UBnormal-main/scripts/normal_training_video_names.txt"

    filenames = [filename1,filename2]
    # path to your own data
    dict_path = '/mnt/hdd8T/hh/code/Joint-VAD/data/weakly_UBnormal/ub-num.npy'
    write_frame_nums(dict_path, video_dir, filenames)

if __name__ == '__main__':
    # path to the output folder
    output_folder_base = '/mnt/hdd8T/hh/code/Joint_VAD/data/UBnormal/gt/test_gt'
    # path to the data set
    video_dir = '/mnt/hdd8T/hh/dataset/UBnormal'
    # path to the txt file with the video names, downloaded from https://github.com/lilygeorgescu/UBnormal
    filename_abnormal = "/mnt/hdd8T/hh/code/UBnormal-main/scripts/abnormal_test_video_names.txt"
    filename_normal = "/mnt/hdd8T/hh/code/UBnormal-main/scripts/normal_test_video_names.txt"
    # if the list with the video names are for the normal videos set is_abnormal to False, otherwise set it to True.
    write_frame_level_ground_truth(output_folder_base, video_dir, filename_abnormal, is_abnormal=True)
    write_frame_level_ground_truth(output_folder_base, video_dir, filename_normal, is_abnormal=False)

    # get video length
    calculate_video_length(video_dir)
