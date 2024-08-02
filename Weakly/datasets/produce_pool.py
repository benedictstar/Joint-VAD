import os
from Weakly.datasets.bag_creation import *
def pool_creation(data_path='list/shanghai-i3d-train-10crop.list',bag_size=16,test_mode=False,pseudo_idx=None,dataset='ShanghaiTech',extractor='i3d'):
    # data_path = os.path.join(root_path,data_path)
    if not test_mode:
        normal_pool,pseudo_normal_scores = get_pseudo_features(root_path=data_path,is_normal=True,test_mode=test_mode,pseudo_idx=pseudo_idx,dataset=dataset,extractor=extractor)
        abnormal_pool,pseudo_abnormal_scores = get_pseudo_features(root_path=data_path,is_normal=False,test_mode=test_mode,pseudo_idx=pseudo_idx,dataset=dataset,extractor=extractor)

        # in default is True
        shuffle = True
        normal_bag_list = create_bag(normal_pool,bag_size=bag_size,shuffle=shuffle,pseudo_scores=pseudo_normal_scores)
        abnormal_bag_list = create_bag(abnormal_pool,bag_size=bag_size,shuffle=shuffle,pseudo_scores=pseudo_abnormal_scores)

        return normal_bag_list,abnormal_bag_list



