import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from Occ.models.STG_NF.model_pose import STG_NF
from Occ.models.training import Trainer
from Occ.utils.data_utils import trans_list
from Occ.utils.optim_init import init_optimizer, init_scheduler
from Occ.Occ_args import init_sub_args
from Occ.utils.train_utils import init_model_params
from Occ.utils.scoring_utils import score_dataset
# from Occ.utils.train_utils import calc_num_of_params
from torch.utils.data import DataLoader
from Occ.utils.training_scoring_utils import training_score_dataset
import os


def Occ(Occ_args=None,dataset=None,pseudo_weight=None,dir_best_record='',load=False,writer=None,steps=0):
    args, model_args = init_sub_args(Occ_args)

    pretrained = os.path.join(dir_best_record,'Occ_epoch_final_checkpoint.pth.tar') if load else None

    # update the training weights
    if pseudo_weight != None:
        dataset['train'].update_weights(pseudo_weight)

    loader = dict()
    loader_args = {'batch_size': args.Occ_batch_size, 'num_workers': args.Occ_num_workers, 'pin_memory': True}
    loader['train'] = DataLoader(dataset['train'], **loader_args, shuffle=True)
    # produce anomaly scores for training set to get pseudo labels
    loader['train_test'] = DataLoader(dataset['train_test'], **loader_args, shuffle=False)
    # valid on testing set
    loader['test'] = DataLoader(dataset['test'], **loader_args, shuffle=False)

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    # num_of_params = calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['train_test'],loader['test'],
                      optimizer_f=init_optimizer(args.Occ_model_optimizer, lr=args.Occ_model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.Occ_model_lr, epochs=args.Occ_epochs))

    if pretrained:
        trainer.load_checkpoint(pretrained,args)

    trainer.train(log_writer=writer,dir_best_record=dir_best_record,steps=steps)


    normality_scores = trainer.training_scores_producer()
    training_scores = training_score_dataset(normality_scores, dataset["train_test"].metadata, args=args)

    test_scores = trainer.test()
    test_auc,_,_= score_dataset(test_scores, dataset["test"].metadata, args=args)

    # Logging and recording results
    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for {} samples\033[0m".format(test_auc * 100, training_scores.shape[0]))
    print("-------------------------------------------------------\n\n")

    return training_scores, test_auc

# if __name__ == '__main__':
#     main()
