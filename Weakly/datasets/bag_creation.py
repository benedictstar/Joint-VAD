import numpy as np
import torch
import os
import re
def create_bag(feat_pool=None,bag_size=16,shuffle=True,pseudo_scores=None):
    if pseudo_scores is not None:
        scores = torch.Tensor(pseudo_scores)
        scores,idx = torch.sort(scores)
        scores = scores.cpu().numpy()
        feat_pool = torch.Tensor(feat_pool)
        feat_pool = feat_pool[idx].cpu().numpy()
    bag_list=[]
    # score_list = []
    features = add_to_bag_size(feat_pool,bag_size)
    bag_num = features.shape[0] // bag_size
    if pseudo_scores is None:
        length = features.shape[0]
        index = torch.randperm(length).cpu().numpy()
        start_idx = 0
        while(start_idx<length):
            if shuffle:
                excerpt = index[start_idx:start_idx+bag_size]
                bag = features[excerpt,:,:]
            else:
                bag = features[start_idx:start_idx+bag_size,:,:]
            bag_list.append(bag)
            start_idx += bag_size
    # if not shuffle,then it doesn't need the following steps
    else:
        features = features[::-1]

        length = features.shape[0] - 3 * bag_num
        index = torch.randperm(length).cpu().numpy()
        sub_features = features[3*bag_num:,:,:]

        sum_bag = 0
        start_idx = 0
        while(sum_bag<bag_num):
            if shuffle:
                sub_bag = []
                sub_bag.append(features[sum_bag,:,:])
                sub_bag.append(features[sum_bag+bag_num,:,:])
                sub_bag.append(features[sum_bag+bag_num*2,:,:])
                sub_bag = np.stack(sub_bag,axis=0)
                excerpt = index[start_idx:start_idx+bag_size-3]
                bag = np.concatenate((sub_bag,sub_features[excerpt,:,:]),axis=0)
            else:
                bag = features[start_idx:start_idx+bag_size,:,:].copy()

            bag_list.append(bag)
            sum_bag += 1
            start_idx += bag_size-3
    return bag_list

def add_to_bag_size(features=None,bag_size=16):
    number_snippt = features.shape[0]
    assert number_snippt != 0
    
    if number_snippt >= bag_size:
        copy_number = number_snippt % bag_size
        if copy_number != 0:
            # copy last number_snippt features
            copy_feature = features[number_snippt-bag_size:number_snippt-copy_number,:,:]
            features = np.concatenate((features,copy_feature),axis=0)
    else:
        copy_number = bag_size - number_snippt
        copy_feature = features[number_snippt-copy_number:,:,:]
        features = np.concatenate((features,copy_feature),axis=0)

    assert features.shape[0] % bag_size == 0 

    return features

def get_pseudo_features(root_path='list/shanghai-i3d-train-10crop.list',pseudo_idx=None,is_normal=False,test_mode=False,dataset='ShanghaiTech',extractor='i3d'):
    video_list = list(open(root_path))
    total_features = []
    total_scores = []
    for video in video_list:
        features = np.load(video.strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        if not test_mode:
            if pseudo_idx is None:
                print('no such file')
                exit(1)
            
            if extractor == 'i3d':
                video = video.split('_i3d.npy')[0]
            elif extractor == 'c3d':
                video = video.split('_c3d.npy')[0]
            video = video.split('/')[-1]

            if dataset == 'UBnormal':
                type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)', video)[0]
                clip_id = type + clip_id
                video = scene_id+'_'+clip_id

            assert pseudo_idx[video]['p_labels'].shape[0] >= features.shape[0]

            num = pseudo_idx[video]['p_labels'].shape[0] - features.shape[0]

            flag = 0 if is_normal else 1
            pseudo_labels = pseudo_idx[video]['p_labels']
            pseudo_scores = pseudo_idx[video]['scores']
            if features.shape[0] < pseudo_idx[video]['p_labels'].shape[0]:
                pseudo_labels = pseudo_labels[:-num]
                pseudo_scores = pseudo_scores[:-num]
                pseudo_idx[video]['p_labels'] = pseudo_labels
                pseudo_idx[video]['scores'] = pseudo_scores

            assert  pseudo_idx[video]['p_labels'].shape[0] == features.shape[0]
            mask = np.where(pseudo_labels==flag,True,False)
            features = features[mask]
            scores = pseudo_scores[mask]

        total_features.append(features)
        total_scores.append(scores)
    total_features = np.concatenate(total_features,axis=0)
    total_scores = np.concatenate(total_scores,axis=0)
    return total_features,total_scores

def get_pool(pseudo_path='',is_normal=True):
    dic = np.load(pseudo_path,allow_pickle=True).item()
    if is_normal:
        return dic['nor_pool']
    else:
        return dic['ab_pool']
    