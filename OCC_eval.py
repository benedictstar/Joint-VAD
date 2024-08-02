import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from Occ.models.STG_NF.model_pose import STG_NF
from Occ.models.training import Trainer
from Occ.utils.data_utils import trans_list
from Occ.utils.optim_init import init_optimizer, init_scheduler
from Occ.Occ_args import create_exp_dirs
from main_args import init_parser
from Occ.Occ_args import init_sub_args
from Occ.dataset import get_dataset_and_loader
from Occ.utils.train_utils import dump_args, init_model_params
from Occ.utils.scoring_utils import score_dataset
from Occ.utils.train_utils import calc_num_of_params


def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('Occ_checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['test'],loader['test'],
                      optimizer_f=init_optimizer(args.Occ_model_optimizer, lr=args.Occ_model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.Occ_model_lr, epochs=args.Occ_epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained,args)
    else:
        writer = SummaryWriter()
        trainer.train(log_writer=writer)
        dump_args(args, args.ckpt_dir)

    normality_scores = trainer.test()
    auc, scores,_ = score_dataset(normality_scores, dataset["test"].metadata, args=args)

    # Logging and recording results
    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
    print("-------------------------------------------------------\n\n")


if __name__ == '__main__':
    main()