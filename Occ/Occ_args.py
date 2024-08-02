import os
import pickle
import time
import argparse

def init_sub_args(args):
    dataset = "UBnormal" if args.dataset == "UBnormal" else "ShanghaiTech"
    if args.vid_path_train and args.vid_path_test and args.pose_path_train and args.pose_path_test:
        args.vid_path = {'train': args.vid_path_train,
                         'test': args.vid_path_test}

        args.pose_path = {'train': args.pose_path_train,
                          'test': args.pose_path_test}
    else:
        args.vid_path = {'train': os.path.join(args.data_dir, dataset, 'train/images/'),
                         'train_test':  os.path.join(args.data_dir, dataset, 'train/frames/'),
                         'test': os.path.join(args.data_dir, dataset, 'test/images/'),}
        if args.mode == 'unsupervised':
            args.pose_path = {'train': os.path.join(args.data_dir, 'weakly_'+dataset, 'pose', 'train/'),
                        'train_test':  os.path.join(args.data_dir, 'weakly_'+dataset, 'pose', 'train/'),
                        'test': os.path.join(args.data_dir, 'weakly_'+dataset, 'pose', 'test/')}
            if dataset == 'UBnormal':
                args.occ_gt = os.path.join(args.data_dir, 'weakly_'+dataset, 'gt', 'test_gt/')
            else:
                args.occ_gt = os.path.join(args.data_dir, 'weakly_'+dataset, 'gt', 'test_frame_mask/')
    args.pose_path["train_abnormal"] = args.pose_path_train_abnormal
    args.ckpt_dir = None
    model_args = args_rm_prefix(args, 'model_')
    return args, model_args

def args_rm_prefix(args, prefix):
    wp_args = argparse.Namespace(**vars(args))
    args_dict = vars(args)
    wp_args_dict = vars(wp_args)
    for key, value in args_dict.items():
        if key.startswith(prefix):
            model_key = key[len(prefix):]
            wp_args_dict[model_key] = value

    return wp_args


def create_exp_dirs(experiment_dir, dirmap=''):
    time_str = time.strftime("%b%d_%H%M")

    experiment_dir = os.path.join(experiment_dir, dirmap, time_str)
    dirs = [experiment_dir]

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        print("Experiment directories created")
        return experiment_dir
    except Exception as err:
        print("Experiment directories creation Failed, error {}".format(err))
        exit(-1)


def save_dataset(dataset, fname):
    with open(fname, 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(fname):
    with open(fname, 'rb') as file:
        return pickle.load(file)

