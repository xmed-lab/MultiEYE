from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, multilabel_confusion_matrix, cohen_kappa_score
from collections import defaultdict
import numpy as np
import os
import torch


def f1_score(A, B):
    if A == 0 and B == 0:
        return 0
    return 2.0 * A * B / (A + B)


def accuracy_score(conf_mat):
    return float(conf_mat[0][0] + conf_mat[1][1]) / float(conf_mat[0][1] + conf_mat[0][0] + conf_mat[1][0] + conf_mat[1][1])


def specificity_score(confusion_matrix):
    if confusion_matrix[0][0] == 0 and confusion_matrix[0][1] == 0:
        return 0.0
    return float(confusion_matrix[0][0]) / float(confusion_matrix[0][1] + confusion_matrix[0][0])


def sensitivity_score(confusion_matrix):
    if confusion_matrix[1][0] == 0 and confusion_matrix[1][1] == 0:
        return 0.0
    return float(confusion_matrix[1][1]) / float(confusion_matrix[1][1] + confusion_matrix[1][0])


def precision_score(confusion_matrix):
    if confusion_matrix[1][1] == 0 and confusion_matrix[0][1] == 0:
        return 0.0
    return float(confusion_matrix[1][1]) / float(confusion_matrix[1][1] + confusion_matrix[0][1])


def auc_score(predicts_specific, expects_specific):
    auc_specific = roc_auc_score(expects_specific, predicts_specific)
    return auc_specific


def map_score(predicts_specific, expects_specific):
    return average_precision_score(expects_specific, predicts_specific)


def compute_metrics(predicts, expects, scores, cls_num):
    confusion_matrix = multilabel_confusion_matrix(expects, predicts)
    results = defaultdict(list)
    print(confusion_matrix)

    for i in range(cls_num):
        sen = sensitivity_score(confusion_matrix[i])
        spe = specificity_score(confusion_matrix[i])
        pre = precision_score(confusion_matrix[i])
        rec = sen
        acc = accuracy_score(confusion_matrix[i])
        results['sen'].append(sen)
        results['spe'].append(spe)
        results['pre'].append(pre)
        results['rec'].append(rec)
        results['f1_pr'].append(f1_score(pre, rec))
        results['f1_ss'].append(f1_score(sen, spe))
        results['acc'].append(acc)

        predicts_specific = scores[:, i].tolist()
        expects_specific = expects[:, i].tolist()
        predict_labels_specific = predicts[:, i].tolist()
        auc_roc = auc_score(predicts_specific, expects_specific)
        auc_pr = map_score(predicts_specific, expects_specific)
        kappa = cohen_kappa_score(predict_labels_specific, expects_specific)
        results['auc_roc'].append(auc_roc)
        results['auc_pr'].append(auc_pr)
        results['kappa'].append(kappa)

    return results

def print_results(results, alldiseases, cls_num):
    print('Class\tSen\tSpe\tPre\tRec\tF1_SS\tF1_PR\tAcc\tAucROC\tAucPR\tKappa')
    for idx in range(cls_num):
        print(alldiseases[idx] + '\t{sen:.4f}\t{spe:.4f}\t{pre:.4f}\t{rec:.4f}\t{f1_ss:.4f}\t{f1_pr:.4f}\t{acc:.4f}\t{auc:.4f}\t{map:.4f}\t{kappa:.4f}'
              .format(sen=results['sen'][idx],
                      spe=results['spe'][idx],
                      pre=results['pre'][idx],
                      rec=results['rec'][idx],
                      f1_ss=results['f1_ss'][idx],
                      f1_pr=results['f1_pr'][idx],
                      acc=results['acc'][idx],
                      auc=results['auc_roc'][idx],
                      map=results['auc_pr'][idx],
                      kappa=results['kappa'][idx]))

    sen_t = np.average(results['sen'])
    spe_t = np.average(results['spe'])
    pre_t = np.average(results['pre'])
    rec_t = np.average(results['rec'])
    f1_ss_t = np.average(results['f1_ss'])
    f1_pr_t = np.average(results['f1_pr'])
    acc_t = np.average(results['acc'])
    auc_t = np.average(results['auc_roc'])
    map_t = np.average(results['auc_pr'])
    kappa_t = np.average(results['kappa'])
    print('Average\tSen\tSpe\tPre\tRec\tF1_SS\tF1_PR\tAcc\tAucROC\tAucPR\tKappa')
    print('\t{sen:.4f}\t{spe:.4f}\t{pre:.4f}\t{rec:.4f}\t{f1_ss:.4f}\t{f1_pr:.4f}\t{acc:.4f}\t{auc:.4f}\t{map:.4f}\t{kappa:.4f}'
          .format(sen=sen_t, spe=spe_t, pre=pre_t, rec=rec_t, f1_ss=f1_ss_t, f1_pr=f1_pr_t, acc=acc_t, auc=auc_t, map=map_t, kappa=kappa_t))
    return f1_ss_t, f1_pr_t, acc_t, auc_t, map_t, kappa_t



def compute_metrics_threshold(predicts, expects, cls_num):
    confusion_matrix = multilabel_confusion_matrix(expects, predicts)
    results = defaultdict(list)
    print(confusion_matrix)

    for i in range(cls_num):
        sen = sensitivity_score(confusion_matrix[i])
        spe = specificity_score(confusion_matrix[i])
        pre = precision_score(confusion_matrix[i])
        acc = accuracy_score(confusion_matrix[i])
        results['sen'].append(sen)
        results['spe'].append(spe)
        results['pre'].append(pre)
        results['f1_score'].append(f1_score(pre, sen))
        results['acc'].append(acc)
        
        predict_labels_specific = predicts[:, i].tolist()
        expects_specific = expects[:, i].tolist()
        kappa = cohen_kappa_score(predict_labels_specific, expects_specific)
        results['kappa'].append(kappa)

    return results


def computer_metrics_score(predicts, expects, scores, cls_num):
    results = defaultdict(list)
    for i in range(cls_num):
        predicts_specific = scores[:, i].tolist()
        expects_specific = expects[:, i].tolist()
        
        auc_roc = auc_score(predicts_specific, expects_specific)
        auc_pr = map_score(predicts_specific, expects_specific)
        results['auc_roc'].append(auc_roc)
        results['auc_pr'].append(auc_pr)
        
    return results


def print_results_threshold(results, alldiseases, cls_num):
    print('Class\tSen\tSpe\tPre\tF1\tAcc\tKappa')
    for idx in range(cls_num):
        print(alldiseases[idx] + '\t{sen:.4f}\t{spe:.4f}\t{pre:.4f}\t{f1:.4f}\t{acc:.4f}\t{kappa:.4f}'
              .format(sen=results['sen'][idx],
                      spe=results['spe'][idx],
                      pre=results['pre'][idx],
                      f1=results['f1_score'][idx],
                      acc=results['acc'][idx],
                      kappa=results['kappa'][idx]))

    sen_t = np.average(results['sen'])
    spe_t = np.average(results['spe'])
    pre_t = np.average(results['pre'])
    f1_t = np.average(results['f1_score'])
    acc_t = np.average(results['acc'])
    kappa_t = np.average(results['kappa'])
    print('Average\tSen\tSpe\tPre\tF1\tAcc\tKappa')
    print('\t{sen:.4f}\t{spe:.4f}\t{pre:.4f}\t{f1:.4f}\t{acc:.4f}\t{kappa:.4f}'
          .format(sen=sen_t, spe=spe_t, pre=pre_t, f1=f1_t, acc=acc_t, kappa=kappa_t))
    return sen_t, spe_t, f1_t, acc_t, kappa_t


def save_model(args, epoch, model, optimizer, if_best=True):
    output_dir = args.output_dir
    epoch_name = str(epoch)
    if if_best:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
    else:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer,
        'epoch': epoch,
        'args': args,
    }
    torch.save(to_save, checkpoint_path)
    

def load_model(args, checkpoint, device, if_best=True):
    output_dir = checkpoint
    if if_best:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
    else:
        checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
    if args.device == 'cuda':
        model_state_dict = torch.load(checkpoint_path, map_location=device)['model']
    else:
        model_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['model']
    return model_state_dict
