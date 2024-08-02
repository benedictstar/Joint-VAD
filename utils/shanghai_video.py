import os
import numpy as np
def video_label_length():
    label_path = "./data/weakly_ShanghaiTech/gt/test_frame_mask"
    video_length = {}
    files = sorted(os.listdir(label_path))
    length = 0
    for f in files:
        label = np.load("{}/{}".format(label_path, f))
        video_length[f.split(".")[0]] = label.shape[0]
        length += label.shape[0]
    return video_length

def read_selected_numbers(file_list,directory):
    file_num = {}
    for root, dirs, files in os.walk(directory):
        count = 0
        if os.path.basename(root).split('.')[0] in file_list:
            count += len(files)
            file_num[os.path.basename(root).split('.')[0]]=count
    return file_num

def read_each_numbers(directory):
    file_num = {}
    for root, dirs, files in os.walk(directory):
        count = 0
        count += len(files)
        file_num[os.path.basename(root).split('.')[0]]=count
    return file_num

def read_txt(path):
    file_list=[]
    with open(path,'rb') as f:
        for line in f:
            line = line.decode()
            file = line.split('\n')
            file = file[0].split('\r')
            file_list.append(file[0])
    return file_list

if __name__ == '__main__':
    # path to your own frames
    directory = '/mnt/hdd8T/hh/dataset/Weakly_ShanghaiTech/Train_frames'
    txt_path = '/mnt/hdd8T/hh/code/Joint-VAD/data/weakly_ShanghaiTech/SH_Train.txt'
    file_list = read_txt(txt_path)
    training_num = read_selected_numbers(file_list,directory)
    video_length = {}
    video_length.update(training_num)
    # path to target data
    dict_path = '/mnt/hdd8T/hh/code/Joint-VAD/data/weakly_ShanghaiTech/shanghai-num.npy'
    np.save(dict_path,video_length,allow_pickle=True)