import math
import numpy as np
import os
import torch
import time
import re

def average_every_k(arr, k):
    n = len(arr)
    num_chunks = math.ceil(n/k)
    if num_chunks*k > n:
        last_frame_label = arr[-1]
        for i in range(n,num_chunks*k):
            arr = np.append(arr,last_frame_label)
    chunks = arr.reshape(-1, k)
    # Calculate the average value within each chunk
    averages = np.mean(chunks, axis=1)
    return averages

def read_txt(path):
    file_list=[]
    with open(path,'rb') as f:
        for line in f:
            line = line.decode()
            file = line.split('\n')
            file = file[0].split('\r')
            file_list.append(file[0])
    return file_list

def produce_labels(arr,num,a_num):
    _,frag_max_idx = torch.topk(arr,a_num)
    _,frag_min_idx = torch.topk(arr,num,largest=False)

    p_labels = torch.full(arr.shape,-1)
    p_labels[frag_max_idx] = 1
    p_labels[frag_min_idx] = 0

    for i in range(2,len(p_labels)-2):
        sta = i - 2
        end = i + 2
        if p_labels[i-1] == p_labels[i+1] and p_labels[i-1] != p_labels[i]:
            p_labels[i] = p_labels[i-1]
            arr[i] = (arr[i-1] + arr[i+1]) / 2
        else:
            flag = 0 
            for idx in range(sta,end):
                if p_labels[idx] == p_labels[i] and idx != i:
                    flag = 1
                    break
            if flag == 0:
                p_labels[i] = p_labels[i-1]
                arr[i] = arr[i-1]
    return p_labels,arr

class Balance():
    def __init__(self,args=None,last_Occ_weights=None,last_Weakly_weights=None,cur_Occ_weights=None,cur_Weakly_weights=None) -> None:
        self.args = args
        self.last_Occ_weights = last_Occ_weights
        self.last_Weakly_weights = last_Weakly_weights
        self.cur_Occ_weights = cur_Occ_weights
        self.cur_Weakly_weights = cur_Weakly_weights
        self.error = self.args.error
        self.ab_ratio = self.args.ab_ratio
        self.nor_ratio = 1 - self.args.ab_ratio
        self.fixed_ab_idx = None
        self.fixed_nor_idx = None
        self.fixed_ab_weights = dict()
        
        # self.epsilon = 0.02
        # self.start = 1.1

        self.cur_idx = dict()
        self.fixed_selected_idx = None

        self.data_path = self.args.frame_path
        self.feature_path = self.args.rgb_list
        self.best_occ_auc = -1
        self.best_ws_auc = -1
    

    def Update_weights(self,cur_Occ_weights=None,cur_Weakly_weights=None,cur=True):
        Occ_array = []
        Weakly_array = []
        for key in cur_Occ_weights.keys():
            Occ_array.append(cur_Occ_weights[key])

        Occ_array = np.concatenate(Occ_array,axis=0)
        Weakly_array = np.squeeze(cur_Weakly_weights)
        assert len(Occ_array) == len(Weakly_array)
        
        if cur:
            self.cur_Occ_weights = Occ_array
            self.cur_Weakly_weights = Weakly_array


            frame_num = len(Occ_array)
            a_num = math.ceil(self.ab_ratio * frame_num)

            Occ_ab_idx = np.argpartition(-self.cur_Occ_weights,a_num)[:a_num]
            ab_consistent_idx = Occ_ab_idx
            sub_scores = self.cur_Occ_weights[ab_consistent_idx]

            if len(self.cur_idx) != 0:
                mid_idx = np.intersect1d(ab_consistent_idx,np.array(list(self.cur_idx.keys())))
                mid_dict = {}
                for key in mid_idx:
                    if key in self.cur_idx:
                        mid_dict[key] = self.cur_idx[key]
                    else:
                        mid_dict[key] = 0
                self.cur_idx = mid_dict

                for i in range(len(ab_consistent_idx)):
                    if ab_consistent_idx[i] in self.cur_idx.keys(): 
                        self.cur_idx[ab_consistent_idx[i]] += sub_scores[i]
            else:
                for i in range(len(ab_consistent_idx)):
                    self.cur_idx[ab_consistent_idx[i]] = sub_scores[i]

        else:
            self.last_Occ_weights = Occ_array
            self.last_Weakly_weights = Weakly_array
            
    def Find_consistent(self):
        # select fixed num for next round
        keys = np.array(list(self.cur_idx.keys()))
        cur_dropping_num = 0

        if self.fixed_selected_idx is None:
            self.fixed_selected_idx = keys
        else:
            pre_fixed_num = len(self.fixed_selected_idx)
            self.fixed_selected_idx = np.intersect1d(keys,self.fixed_selected_idx)
            # pre_fixed_num >= self.fixed_selected_idx since intersection
            cur_dropping_num = pre_fixed_num - len(self.fixed_selected_idx)
        return cur_dropping_num
        
    def Judge(self):
        if self.last_Occ_weights is None and self.last_Weakly_weights is None:
            return False
        else:
            dif_Occ_array = abs(self.cur_Occ_weights-self.last_Occ_weights)
            dif_Occ = np.mean(dif_Occ_array,axis=0)
            dif_Weakly_array = abs(self.cur_Weakly_weights-self.last_Weakly_weights)
            dif_Weakly = np.mean(dif_Weakly_array,axis=0)
            balance = False
            if dif_Occ <= self.error or dif_Weakly <= self.error:
                balance = True
            else:
                balance = False
            return balance
        
    def Clear(self):
        self.last_Occ_weights = None
        self.last_Weakly_weights = None
        self.cur_Occ_weights = None
        self.cur_Weakly_weights = None
        self.cur_idx = dict()
    
    def Occ_to_Weakly(self,fragment=None):
        directory = self.data_path
        file_list = read_txt(self.args.txt_path)

        video_length = np.load(directory,allow_pickle=True).item()

        path = self.feature_path

        ab_ratio = self.ab_ratio

        ratio = self.nor_ratio

        samples = np.array([], dtype=np.float32)

        indices = {}

        frame_idx = {}

        start = 0

        fragment = -fragment
        assert fragment.min() >= 0

        video_list = list(open(path))
        features_length = {}
        for video in video_list:
            features = np.load(video.strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            # pseduo labels corresponds to features
            if self.args.feat_extractor=='i3d': 
                video = video.split('_i3d.npy')[0]
            elif self.args.feat_extractor=='c3d':
                video = video.split('_c3d.npy')[0]
            video = video.split('/')[-1]
            if self.args.dataset == 'UBnormal':
                type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)', video)[0]
                clip_id = type + clip_id
                video = scene_id+'_'+clip_id
            features_length[video] = features.shape[0]

        # Follow the file_list to origanize data
        for i in range(len(file_list)):
            video_name = file_list[i]
            length = video_length[video_name]
            hook = fragment[start:start+length]
            frame_idx[video_name] = hook
            hook = average_every_k(hook,16)
            feature_length = features_length[video_name]
            
            num = hook.shape[0] - feature_length
            if feature_length < hook.shape[0]:
                hook = hook[:-num]
            assert hook.shape[0] == feature_length
            indices[video_name] = hook
            samples = np.concatenate((samples,hook),axis=0)
            start += length
        
        fragment = samples

        frame_num = len(fragment)

        # if we have fixed selected num,then use it. Only use the length of the interseaction
        if self.fixed_selected_idx is not None:
            a_num = len(self.fixed_selected_idx)
            num = frame_num - a_num
        else:
            num = math.ceil(ratio * frame_num)
            a_num = math.ceil(ab_ratio * frame_num)

        fragment = torch.as_tensor(fragment)

        p_labels,fragment = produce_labels(arr=fragment,num=num,a_num=a_num)

        p_labels = p_labels.cpu().numpy()
        pseduo_label = {}
        start = 0
        for i in range(len(file_list)):
            dic = {}
            video_name = file_list[i]
            length = features_length[video_name]
            dic['scores'] = fragment[start:start+length]
            dic['p_labels'] = p_labels[start:start+length]
            pseduo_label[video_name] = dic
            start += length
        
        return pseduo_label,indices,frame_idx

    def Weakly_to_Occ(self,Weakly_scores=None):
        # convert feature scores to frame scores
        Weakly_scores = np.repeat(Weakly_scores,16)
        directory = self.data_path
        
        video_length = np.load(directory,allow_pickle=True).item()

        dictionary = self.feature_path
        file_list = list(open(dictionary))

        # normal weights higher, abnormal weights lower
        Weakly_scores = 1 - Weakly_scores

        pseudo_weight = {}
        start = 0
        weights = []
        for file in file_list:
            features = np.load(file.strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            file_name = file.split('.')[0].split('/')[-1]
            if self.args.feat_extractor=='i3d': 
                file_name = file_name.split('_i3d')[0]
            elif self.args.feat_extractor=='c3d':
                file_name = file_name.split('_c3d')[0]
            if self.args.dataset == 'UBnormal':
                type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)', file_name)[0]
                clip_id = type + clip_id
                file_name = scene_id+'_'+clip_id
            length = features.shape[0] * 16
            scores = Weakly_scores[start:start+length]
            # if we have the same length
            if len(scores) > video_length[file_name]:
                scores = scores[:video_length[file_name]]
            elif len(scores) < video_length[file_name]:
                dif = video_length[file_name] - len(scores)
                zeros = np.zeros(dif)
                scores = np.concatenate((scores,zeros),axis=0)
            assert len(scores) == video_length[file_name]
            # pseudo_weight[file_name] = scores
            weights.append(scores)
            start += length
        
        weights = np.concatenate(weights,axis=0)
        if self.fixed_ab_idx is not None and self.fixed_nor_idx is not None:
            weights[self.fixed_ab_idx] = 0
            weights[self.fixed_nor_idx] = 1

        
        start = 0
        for file in file_list:
            file_name = file.split('.')[0].split('/')[-1]
            if self.args.feat_extractor=='i3d':
                file_name = file_name.split('_i3d')[0]
            elif self.args.feat_extractor=='c3d':
                file_name = file_name.split('_c3d')[0]
            if self.args.dataset == 'UBnormal':
                type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)', file_name)[0]
                clip_id = type + clip_id
                file_name = scene_id+'_'+clip_id

            length = video_length[file_name]
            scores = Weakly_scores[start:start+length]
            pseudo_weight[file_name] = scores
            start += length

        return pseudo_weight








