from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from collections import defaultdict
import numpy as np
import os
import torch

def f1_score_ss(A, B):
    if A == 0 and B == 0:
        return 0
    return 2.0 * A * B / (A + B)


def compute_classwise_metrics(y_true, y_pred, y_pred_score=None):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    sen, spe, f1_ss, _, _, _ = compute_sen_spe(y_true, y_pred, [0,1,2,3,4,5,6,7,8])
    if y_pred_score is None:
        results = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "sen": sen,
                    "spe": spe,
                    "f1_ss": f1_ss
                }
        return results
    else:
        auc, map, _, _ = compute_auc_ap(y_true, y_pred_score, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sen": sen,
            "spe": spe,
            "f1_ss": f1_ss,
            "auc": auc,
            "map": map
        }
        return results

def compute_metrics(y_true, y_pred, y_pred_score=None):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    _, _, _, sen, spe, f1_ss = compute_sen_spe(y_true, y_pred, [0,1,2,3,4,5,6,7,8])
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    if y_pred_score is None:
        results = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "sen": sen,
                    "spe": spe,
                    "f1_ss": f1_ss,
                    "accuracy": accuracy,
                    "kappa": kappa
                }
        return results
    else:
        _, _, auc, map = compute_auc_ap(y_true, y_pred_score, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        results = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "sen": sen,
                    "spe": spe,
                    "f1_ss": f1_ss,
                    "accuracy": accuracy,
                    "kappa": kappa,
                    "auc": auc,
                    "map": map
                }
        return results
    
def compute_auc_ap(y_true, y_pred_prob, classes):
    cls_num = len(classes)
    y_true_binarized = label_binarize(y_true, classes=classes) # Binarize the labels for one-vs-rest computation

    auc_scores = []
    ap_scores = []
    for i in range(cls_num):
        # Compute AUC for each class
        auc_scores.append(roc_auc_score(y_true_binarized[:, i], y_pred_prob[:, i]))

        # Compute Average Precision for each class
        ap_scores.append(average_precision_score(y_true_binarized[:, i], y_pred_prob[:, i]))
    average_auc = np.mean(auc_scores)
    average_map = np.mean(ap_scores)

    return auc_scores, ap_scores, average_auc, average_map


def compute_sen_spe(y_true, y_pred, classes):
    cls_num = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sensitivity = []
    specificity = []
    f1_ss = []

    for i in range(cls_num):
        true_positives = cm[i, i]
        false_negatives = np.sum(cm[i, :]) - true_positives
        true_negatives = np.sum(cm) - np.sum(cm[:, i]) - np.sum(cm[i, :]) + true_positives
        false_positives = np.sum(cm[:, i]) - true_positives

        sens = true_positives / (true_positives + false_negatives)
        spec = true_negatives / (true_negatives + false_positives)
        sensitivity.append(sens)
        specificity.append(spec)
        f1_ss.append(f1_score_ss(sens, spec))
        
    average_sen = np.mean(sensitivity)
    average_spe = np.mean(specificity)
    average_f1_ss = np.mean(f1_ss)

    return sensitivity, specificity, f1_ss, average_sen, average_spe, average_f1_ss


def print_result(results_class, results, all_diseases):
    cls_num = len(all_diseases)
    print('Class\tPre\tRec\tF1_PR')
    for idx in range(cls_num):
        print(all_diseases[idx] + '\t{pre:.4f}\t{rec:.4f}\t{f1_pr:.4f}'
              .format(pre=results_class['precision'][idx],
                      rec=results_class['recall'][idx],
                      f1_pr=results_class['f1'][idx],))

    print('Average\tPre\tRec\tF1_PR\tAcc\tKappa')
    print('\t{pre:.4f}\t{rec:.4f}\t{f1_pr:.4f}\t{acc:.4f}\t{kappa:.4f}'
          .format(pre=results['precision'], rec=results['recall'], f1_pr=results['f1'], acc=results['accuracy'], kappa=results['kappa']))
    

def print_result_whole(results_class, results, all_diseases):
    cls_num = len(all_diseases)
    print('Class\tPre\tRec\tF1_PR\tSen\tSpe\tF1_SS\tAUC\tAP')
    for idx in range(cls_num):
        print(all_diseases[idx] + '\t{pre:.4f}\t{rec:.4f}\t{f1_pr:.4f}\t{sen:.4f}\t{spe:.4f}\t{f1_ss:.4f}\t{auc:.4f}\t{ap:.4f}'
              .format(pre=results_class['precision'][idx],
                      rec=results_class['recall'][idx],
                      f1_pr=results_class['f1'][idx],
                      sen=results_class['sen'][idx],
                      spe=results_class['spe'][idx],
                      f1_ss=results_class['f1_ss'][idx],
                      auc=results_class['auc'][idx],
                      ap=results_class['map'][idx]))

    print('Average\tPre\tRec\tF1_PR\tSen\tSpe\tF1_SS\tAUC\tMAP\tAcc\tKappa')
    print('\t{pre:.4f}\t{rec:.4f}\t{f1_pr:.4f}\t{sen:.4f}\t{spe:.4f}\t{f1_ss:.4f}\t{auc:.4f}\t{ap:.4f}\t{acc:.4f}\t{kappa:.4f}'
          .format(pre=results['precision'], rec=results['recall'], f1_pr=results['f1'], acc=results['accuracy'], kappa=results['kappa'],
                  sen=results['sen'], spe=results['spe'], f1_ss=results['f1_ss'], auc=results['auc'], ap=results['map']))
    