from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from Weakly.utils import save_best_record
from Weakly.model import Model
from Weakly.dataset import Dataset
from Weakly.train import train
from Weakly.test_10crop import test
from Weakly.Weakly_args import init_parser,init_sub_args
from tqdm import tqdm
from Weakly.config import *
from Weakly.datasets.produce_pool import pool_creation
import time
import shutil

# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

def Weakly(Weakly_args=None,pseudo_idx=None,dir_best_record='',load=False,writer=None,steps=0):
    # parser = init_parser()
    # args = parser.parse_args()
    config = Config(Weakly_args)
    args = Weakly_args

    model = Model(args.feature_size, args.Weakly_batch_size,args.feat_extractor)

    if load:
        print('loading pretrained_ckpt')
        loaded_state_dict = torch.load(os.path.join(dir_best_record,'Weakly_epoch_final_checkpoint.pkl'))
        model.load_state_dict(loaded_state_dict)
    

    # for name, value in model.named_parameters():
    #     print(name)
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    normal_bag_list,abnormal_bag_list= pool_creation(data_path=args.rgb_list,bag_size=args.bag_size,test_mode=False,pseudo_idx=pseudo_idx,dataset=args.dataset,extractor=args.feat_extractor)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True,bag_list=normal_bag_list),
                               batch_size=args.Weakly_batch_size, shuffle=True,
                               num_workers=args.Weakly_workers, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False,bag_list=abnormal_bag_list),
                               batch_size=args.Weakly_batch_size, shuffle=True,
                               num_workers=args.Weakly_workers, pin_memory=False, drop_last=True)
    
    test_loader = DataLoader(Dataset(args, test_mode=True,bag_list=None),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    auc = test(test_loader, model, args,device)

    cur_best_Weakly_record = os.path.join(dir_best_record, 'Weakly_epoch_final_checkpoint.pkl')

    cur_record_best = os.path.join(dir_best_record, 'Weakly_inner_best_checkpoint.pkl')

    for step in tqdm(
            range(1, args.Weakly_max_epoch + 1),
            total=args.Weakly_max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.Weakly_batch_size, optimizer,device,writer=writer,step=step,rounds=steps*args.Weakly_max_epoch)

        if step % 5 == 0 and step >= 200:

            auc = test(test_loader, model, args, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), cur_record_best)
    torch.save(model.state_dict(), cur_best_Weakly_record)
    # test training set, provide scores for Occ
    pred = Weakly_update(cur_best_Weakly_record,args,device,update=True)
    return pred,best_AUC

def Weakly_update(cur_best_Weakly_record,args,device,update=True):
    model = Model(args.feature_size, args.Weakly_batch_size, args.feat_extractor)
    model = model.to(device)
    loaded_state_dict = torch.load(cur_best_Weakly_record)
    model.load_state_dict(loaded_state_dict)
    test_loader = DataLoader(Dataset(args, test_mode=True,bag_list=None,update=update),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    
    pred = test(test_loader, model, args, device, update=True)
    return pred


    


