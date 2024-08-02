import numpy as np
import os
import glob

# Modified based on https://github.com/tianyu0207/RTFM/

root_path = "/mnt/hdd8T/xwz/dataset/ShanghaiTech/SH_Test_ten_crop_i3d"
dirs = os.listdir(root_path)

def get_check_abnormal_list(root_path):
    abnormal_path = 'data/weakly_ShanghaiTech/list/test_frame_mask/'
    abnormal_file = os.listdir(abnormal_path)
    print(len(abnormal_file))
    check_anomaly_files = []

    for file in abnormal_file:
        file = os.path.join(abnormal_path, file)
        new_file = file.split('.')[0].split('/')[-1]
        new_file = os.path.join(root_path, new_file + '_i3d.npy')
        check_anomaly_files.append(new_file)
        gt = np.load(file.strip('\n'))


    print(check_anomaly_files)
    return check_anomaly_files

with open('./data/weakly_ShanghaiTech/list/shanghai-i3d-test-10crop1.list', 'w+') as f:
    normal = []
    files = sorted(glob.glob(os.path.join(root_path, "*.npy")))
    check_anomaly_files = get_check_abnormal_list(root_path)
    count = 0
    for file in files:
        if not file in check_anomaly_files:  # Normal video
            if file.endswith('_i3d.npy'):
                normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
            count += 1
    print(count)
    for file in normal:  # 175 normal videos, 63 abnormal video
        newline = file+'\n'
        f.write(newline)
