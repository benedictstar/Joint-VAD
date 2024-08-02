import torch
from Occ.train_eval import Occ
from Weakly.train_eval import Weakly
import numpy as np
from main_args import init_parser,init_sub_args
from Occ.utils.data_utils import trans_list
from Occ.dataset import get_Occ_dataset
from utils.data_process import Balance
from torch.utils.tensorboard import SummaryWriter
import os
import time
import random
import shutil
if __name__ == '__main__':
    start_time = time.time()
    parser = init_parser()
    args = parser.parse_args()
    args = init_sub_args(args)
    pseudo_weight=None
    writer = SummaryWriter()
    root_checkpoint_path = 'checkpoints/'
    project_checkpoint_path = os.path.join(root_checkpoint_path, args.mode+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    if not os.path.exists(project_checkpoint_path):
        os.makedirs(project_checkpoint_path, exist_ok=True) 
    
    if args.seed == 1000:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    balance = Balance(args=args)

    occ_best_auc_path = os.path.join(project_checkpoint_path,'Occ_best_auc.pth.tar')
    ws_best_auc_path = os.path.join(project_checkpoint_path,'Ws_best_auc.pkl')
    
    Occ_dataset= get_Occ_dataset(trans_list=trans_list, only_test=False, pseudo_weight=pseudo_weight)

    first_dropping_num = 0
    for epoch in range(args.joint_epoch):
        # mkdirs for checkpoints
        round_checkpoint_path = os.path.join(project_checkpoint_path,'round '+str(epoch+1))
        if not os.path.exists(round_checkpoint_path):
            os.mkdir(round_checkpoint_path)
        dir_best_record = os.path.join(round_checkpoint_path,'dir_cur_best')
        if not os.path.exists(dir_best_record):
            os.mkdir(dir_best_record)
        
        # Interleaved Module
        for step in range(6):
            Occ_load = True if os.path.isfile(os.path.join(dir_best_record,'Occ_epoch_final_checkpoint.pth.tar')) else False
            Weakly_load = True if os.path.isfile(os.path.join(dir_best_record,'Weakly_epoch_final_checkpoint.pkl')) else False
            step_best = False
            Occ_scores, Occ_auc = Occ(Occ_args=args,dataset=Occ_dataset,pseudo_weight=pseudo_weight,dir_best_record=dir_best_record,load=Occ_load,writer=writer,steps=step)
            # update the global best checkpoint
            if Occ_auc > balance.best_occ_auc:
                shutil.copy(os.path.join(dir_best_record, 'Occ_epoch_final_checkpoint.pth.tar'),occ_best_auc_path)
                balance.best_occ_auc = Occ_auc
            pseudo_idx,Occ_indices,Occ_frame_idx = balance.Occ_to_Weakly(fragment=Occ_scores)
            Weakly_scores, ws_auc= Weakly(Weakly_args=args,pseudo_idx=pseudo_idx,dir_best_record=dir_best_record,load=Weakly_load,writer=writer,steps=step)
            if ws_auc > balance.best_ws_auc:
                shutil.copy(os.path.join(dir_best_record,'Weakly_inner_best_checkpoint.pkl'),ws_best_auc_path)
                balance.best_ws_auc = ws_auc
            pseudo_weight = balance.Weakly_to_Occ(Weakly_scores=Weakly_scores)
            balance.Update_weights(cur_Occ_weights=Occ_indices,cur_Weakly_weights=Weakly_scores,cur=True)
            # results = balance.Judge()
            # if results:
            #     break
            balance.Update_weights(cur_Occ_weights=Occ_indices,cur_Weakly_weights=Weakly_scores,cur=False)

        cur_dropping_num = balance.Find_consistent()
        balance.Clear()
        
        if first_dropping_num == 0:
            first_dropping_num = cur_dropping_num
        else:
            if cur_dropping_num <= first_dropping_num * args.stopping_criterion:
                break
    # estimate the training time of the whole procedure
    end_time = time.time()
    process_time = end_time - start_time

    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for Occ Models\033[0m".format(balance.best_occ_auc * 100))
    print("\033[92m Done with {}% AuC for Ws Models\033[0m".format(balance.best_ws_auc * 100))
    print("\033[92m The whole process time: {} seconds\033[0m".format(process_time))
    print("-------------------------------------------------------\n\n")



