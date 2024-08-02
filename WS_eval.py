from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from Weakly.model import Model
from Weakly.dataset import Dataset
from Weakly.test_10crop import test
from main_args import init_parser,init_sub_args
from tqdm import tqdm
from Weakly.config import *
import time
import shutil

def main():
    parser = init_parser()
    args = parser.parse_args()
    args = init_sub_args(args)

    model = Model(args.feature_size, args.Weakly_batch_size, args.feat_extractor)

    pretrained = vars(args).get('Weakly_pretrained_ckpt', None)

    if pretrained:
        print('loading pretrained_ckpt')
        loaded_state_dict = torch.load(args.Weakly_pretrained_ckpt)
        model.load_state_dict(loaded_state_dict)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_loader = DataLoader(Dataset(args, test_mode=True,bag_list=None),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    auc = test(test_loader, model, args,device)
    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC".format(auc * 100))
    print("-------------------------------------------------------\n\n")

if __name__ == '__main__':
    main()

    



    


