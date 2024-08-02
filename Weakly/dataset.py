import torch.utils.data as data
import numpy as np
from Weakly.utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, bag_list=None,update=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            print('use list {}!'.format(args.test_rgb_list))
        # update the training set's scores
        if update:
            self.rgb_list_file = args.rgb_list
            print('use list {}!'.format(args.rgb_list))

        self.tranform = transform
        self.test_mode = test_mode
        # when training, use bag_list
        self.bag_list = bag_list
        if test_mode:
            self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))



    def __getitem__(self, index):
        label = self.get_label()  # get video level label 0/1
        if self.test_mode:
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
        else:
            features = self.bag_list[index]

        if self.tranform is not None:
            features = self.tranform(features)
            
        if self.test_mode:
            return features
        
        features = features.transpose(1, 0, 2)
        return features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        if self.test_mode:
            return len(self.list)
        else:
            return len(self.bag_list)

    def get_num_frames(self):
        return self.num_frame
