import argparse
import os
def init_parser(default_data_dir='data/', default_exp_dir='data/exp_dir'):
    parser = argparse.ArgumentParser(prog="Joint-vad")

    # main config
    parser.add_argument('--dataset', type=str, default='ShanghaiTech',
                        choices=['ShanghaiTech','UBnormal'], help='Dataset for Eval')
    parser.add_argument('--device', type=str, default='cuda:1', metavar='DEV', help='Device for feature calculation (default: \'cuda:0\')')
    parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
    parser.add_argument('--rgb-list', help='list of rgb features ')
    parser.add_argument('--test-rgb-list', help='list of test rgb features')
    parser.add_argument('--mode', default='unsupervised',help='unsupervised')
    # parser.add_argument('--rgb-list', default='list/ubnormal-i3d-train-10crop.list', help='list of rgb features ')
    # parser.add_argument('--test-rgb-list', default='list/ubnormal-i3d-test-10crop.list', help='list of test rgb features ')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, metavar='DATA_DIR', help="Path to directory holding .npy and .pkl files (default: {})".format(default_data_dir))

    # Joint Args
    parser.add_argument('--joint-epoch', type=int, default=6, help='circle rounds')
    parser.add_argument('--error', type=float, default=0.1, help='Path to training vids')
    parser.add_argument('--ab_ratio', type=float, default=0.15, help='ratio of abnormal samples,the rest of dataset are thought to be normal')
    parser.add_argument('--stopping-criterion', type=float, default=0.1, help='ratio of abnormal samples,the rest of dataset are thought to be normal')
    parser.add_argument('--seed', type=int, metavar='S', default=2023, help='Random seed, use 999 for random (default: 2023)')
    parser.add_argument('--frame_path', type=str, help='frames path')
    parser.add_argument('--txt_path', type=str, default='SH_Train.txt', help='video seqeunces')
    
    # OCC Config
    parser.add_argument('--vid_path_train', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train_abnormal', type=str, default=None, help='Path to training vids')
    parser.add_argument('--pose_path_train', type=str, default=None, help='Path to training pose')
    parser.add_argument('--vid_path_test', type=str, default=None, help='Path to test vids')
    parser.add_argument('--pose_path_test', type=str, default=None, help='Path to test pose')
    parser.add_argument('--vid_res', type=str, default=None, help='Video Res')
    parser.add_argument('--verbose', type=int, default=1, metavar='V', choices=[0, 1], help='Verbosity [1/0] (default: 1)')
    parser.add_argument('--exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR', help="Path to the directory where models will be saved (default: {})".format(default_exp_dir))
    parser.add_argument('--Occ_num_workers', type=int, default=8, metavar='W', help='number of dataloader workers (0=current thread) (default: 32)')
    parser.add_argument('--plot_vid', type=int, default=0, help='Plot test videos')
    parser.add_argument('--only_test', action='store_true', help='Visualize train/test data')
    # Data Params
    parser.add_argument('--num_transform', type=int, default=2, metavar='T', help='number of transformations to use for augmentation (default: 2)')
    parser.add_argument('--headless', action='store_true', help='Remove head keypoints (14-17) and use 14 kps only. (default: False)')
    parser.add_argument('--norm_scale', '-ns', type=int, default=0, metavar='NS', choices=[0, 1], help='Scale without keeping proportions [1/0] (default: 0)')
    parser.add_argument('--prop_norm_scale', '-pns', type=int, default=1, metavar='PNS', choices=[0, 1], help='Scale keeping proportions [1/0] (default: 1)')
    parser.add_argument('--train_seg_conf_th', '-th', type=float, default=0.0, metavar='CONF_TH', help='Training set threshold Parameter (default: 0.0)')
    parser.add_argument('--seg_len', type=int, default=24, metavar='SGLEN', help='Number of frames for training segment sliding window, a multiply of 6 (default: 12)')
    parser.add_argument('--seg_stride', type=int, default=6, metavar='SGST', help='Stride for training segment sliding window')
    parser.add_argument('--specific_clip', type=int, default=None, help='Train and Eval on Specific Clip')
    parser.add_argument('--global_pose_segs', action='store_false', help='Use unormalized pose segs')
    # Model Params
    parser.add_argument('--Occ_checkpoint',type=str, default=None,metavar='model', help="Path to a pretrained model")
    parser.add_argument('--Occ_batch_size', type=int, default=256,  metavar='B', help='Batch size for train')
    parser.add_argument('--Occ_epochs', '-model_e', type=int, default=1, metavar='E', help = 'Number of epochs per cycle')
    parser.add_argument('--Occ_model_optimizer', '-model_o', type=str, default='adamx', metavar='model_OPT', help = "Optimizer")
    parser.add_argument('--model_sched', '-model_s', type=str, default='exp_decay', metavar='model_SCH', help = "Optimization LR scheduler")
    parser.add_argument('--Occ_model_lr', type=float, default=5e-4, metavar='LR', help='Optimizer Learning Rate Parameter')
    parser.add_argument('--model_weight_decay', '-model_wd', type=float, default=5e-5, metavar='WD', help='Optimizer Weight Decay Parameter')
    parser.add_argument('--model_lr_decay', '-model_ld', type=float, default=0.99, metavar='LD', help='Optimizer Learning Rate Decay Parameter')
    parser.add_argument('--model_hidden_dim', type=int, default=0, help='Features dim dimension')
    parser.add_argument('--model_confidence', action='store_true', help='Create Figs')
    parser.add_argument('--K', type=int, default=8, help='Features dim dimension')
    parser.add_argument('--L', type=int, default=1, help='Features dim dimension')
    parser.add_argument('--R', type=float, default=3., help='Features dim dimension')
    parser.add_argument('--temporal_kernel', type=int, default=None, help='Odd integer, temporal conv size')
    parser.add_argument('--edge_importance', action='store_true', help='Adjacency matrix edge weights')
    parser.add_argument('--flow_permutation', type=str, default='permute', help='Permutation layer type')
    parser.add_argument('--adj_strategy', type=str, default='uniform', help='Adjacency matrix strategy')
    parser.add_argument('--max_hops', type=int, default=8, help='Adjacency matrix neighbours')

    # Weakly Config
    parser.add_argument('--model-name', default='rtfm', help='name to save model')
    parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
    parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
    parser.add_argument('--gt', default='list/gt-sh.npy', help='file of ground truth ')
    parser.add_argument('--Weakly_lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
    parser.add_argument('--Weakly_batch_size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
    parser.add_argument('--Weakly_workers', default=4, help='number of workers in dataloader')
    parser.add_argument('--Weakly_pretrained_ckpt', default=None, help='ckpt for pretrained model')
    parser.add_argument('--num-classes', type=int, default=1, help='number of class')
    parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
    parser.add_argument('--Weakly-max-epoch', type=int, default=500, help='maximum iteration to train')
    parser.add_argument('--bag_size', type=int, default=16, help='size of every bag')
    
    return parser

def init_sub_args(args):
    dataset = "UBnormal" if args.dataset == "UBnormal" else "ShanghaiTech"
    if args.mode == 'unsupervised':
        if dataset == "ShanghaiTech":
            args.frame_path = os.path.join(args.data_dir,'weakly_'+dataset,'shanghai-num.npy')
            args.gt = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/gt-sh.npy')
            args.txt_path = os.path.join(args.data_dir, 'weakly_'+dataset, 'SH_Train.txt')
            if args.feat_extractor == 'i3d':
                args.rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/shanghai-i3d-train-10crop.list')
                args.test_rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/shanghai-i3d-test-10crop.list')
            elif args.feat_extractor == 'c3d':
                args.rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/shanghai-c3d-train-10crop.list')
                args.test_rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/shanghai-c3d-test-10crop.list')
                args.gt = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/gt-sh-c3d.npy')
        elif dataset == "UBnormal":
            args.txt_path = os.path.join(args.data_dir,'weakly_'+dataset,'UB_Train.txt')
            args.frame_path = os.path.join(args.data_dir,'weakly_'+dataset,'ub-num.npy')
            args.gt = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/gt-ub.npy')
            if args.feat_extractor == 'i3d':
                args.rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/ubnormal-i3d-train-10crop.list')
                args.test_rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/ubnormal-i3d-test-10crop.list')
            if args.feat_extractor == 'c3d':
                args.rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/ubnormal-c3d-train-10crop.list')
                args.test_rgb_list = os.path.join(args.data_dir, 'weakly_'+dataset, 'list/ubnormal-c3d-test-10crop.list')
    return args

def get_parameters(args,name):
    args = args
