import numpy as np
import os
import glob

# Modified based on https://github.com/tianyu0207/RTFM/
root_path = "/mnt/hdd8T/hh/dataset/UBnormal_new/UB_Test_ten_crop_i3d"
dirs = os.listdir(root_path)

def get_check_abnormal_list(root_path):
    check_anomaly_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_type = file.split('_')[0]
            if file_type == 'abnormal':
                new_file = os.path.join(root_path,file)
                check_anomaly_files.append(new_file)

    print(check_anomaly_files)
    return check_anomaly_files

with open('./data/weakly_UBnormal/list/ubnormal-i3d-test-10crop1.list', 'w+') as f:
    normal = []
    files = sorted(glob.glob(os.path.join(root_path, "*.npy")))
    check_anomaly_files = get_check_abnormal_list(root_path)
    count = 0
    for file in files:
        if not file in check_anomaly_files:  # Normal video
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
            count += 1
    print(count)
    for file in normal:  
        newline = file+'\n'
        f.write(newline)
