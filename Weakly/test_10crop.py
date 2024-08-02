import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args, device, update=False):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        # testing on testing set
        if not update:
            pred = np.repeat(np.array(pred), 16)
            gt = np.load(args.gt)
            fpr, tpr, threshold = roc_curve(list(gt), pred)
            # np.save('fpr.npy', fpr)
            # np.save('tpr.npy', tpr)
            rec_auc = auc(fpr, tpr)
            print('auc : ' + str(rec_auc))

            precision, recall, th = precision_recall_curve(list(gt), pred)
            pr_auc = auc(recall, precision)
            # np.save('precision.npy', precision)
            # np.save('recall.npy', recall)
            return rec_auc
        # testing on training set
        else:
            update_pred = np.array(pred)
            return update_pred
