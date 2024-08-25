import torch
import torch.nn as nn
import torch.nn.functional as F


def KDloss(output, teacher_output, alpha, temperature):
    """
    from:
        https://github.com/peterliht/knowledge-distillation-pytorch
        
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = alpha
    T = temperature
    
    """
    kd_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(output/T, dim=1),
                                             F.softmax(teacher_output/T, dim=1)).type(torch.FloatTensor).cuda(gpu)
    
    kd_loss = kd_filter * torch.sum(kd_loss, dim=1) # kd filter is filled with 0 and 1.
    kd_loss = torch.sum(kd_loss) / torch.sum(kd_filter) * (alpha * T * T)
    """
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/T, dim=1),
                                                  F.softmax(teacher_output/T, dim=1)) * (alpha * T * T)
    
    return kd_loss

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def hard_nn(logits, num_cls):
    _, idx = (logits).max(dim=1)
    return idx
    # plan = F.one_hot(idx, num_cls).float()
    # return plan


def soft_nn(logits):
    plan = F.softmax(logits, dim=1)
    return plan
