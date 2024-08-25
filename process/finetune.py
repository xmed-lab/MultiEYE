import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd

from utils.lr_sched import adjust_learning_rate
from utils.logger import MetricLogger, SmoothedValue
from utils.eval_single import compute_metrics, compute_classwise_metrics, print_result
from utils.losses import KDloss

# fundus
alldiseases_f = ['NOR', 'AMD', 'CSC', 'DR', 'GLC', 'MEM', 'MYO', 'RVO', 'WAMD']
cls_names_f =  ['normal', 'dry age-related macular degeneration', 'central serous chorioretinopathy',
                'diabetic retinopathy', 'glaucoma',
                'epiretinal membrane', 'myopia',
                'retinal vein occlusion', 'wet age-related macular degeneration']
# OCT
alldiseases_o = ['NOR', 'AMD', 'CSC', 'DR', 'GLC', 'MEM', 'MYO', 'RVO', 'WAMD']
cls_names_o =  ['normal', 'dry age-related macular degeneration', 'central serous chorioretinopathy',
                'diabetic retinopathy', 'glaucoma',
                'epiretinal membrane', 'myopia',
                'retinal vein occlusion', 'wet age-related macular degeneration']

def dist_s_t(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)


def dist_s_label(y, q):

    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)

    return torch.mean(dist)


def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, args, oct_model, oct_optimizer):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if data_iter_step % accum_iter == 0:    
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        f_samples = samples[0].to(device)
        f_targets = targets[0].to(device)
        
        o_samples = samples[1].to(device)
        o_targets = targets[1].to(device)
        
        exist_classes = list(set(f_targets.detach().cpu().numpy().tolist()) & set(o_targets.detach().cpu().numpy().tolist()))
        concept_feats = model.get_concepts_feat()
        dim_layer = 2
        f_dist, o_dist = torch.zeros((len(exist_classes), dim_layer, concept_feats.shape[0])).to(device), torch.zeros((len(exist_classes), dim_layer, concept_feats.shape[0])).to(device)
        f_num, o_num = torch.zeros(len(exist_classes)).to(device), torch.zeros(len(exist_classes)).to(device)
        
        with torch.cuda.amp.autocast():
            outputs, concept_sims = model.forward_distill(f_samples)
            loss_cls_f = criterion(outputs, f_targets)
            preds = torch.softmax(outputs, dim=1)
            f_score = torch.gather(preds, 1, f_targets.unsqueeze(1)).squeeze(1).detach()
            
            oct_outputs, o_concept_sims = oct_model.forward_distill(o_samples)
            loss_cls_o = criterion(oct_outputs, o_targets)
            o_preds = torch.softmax(oct_outputs, dim=1)
            o_score = torch.gather(o_preds, 1, o_targets.unsqueeze(1)).squeeze(1).detach()
        
            oct_optimizer.zero_grad()
            loss_cls_o.backward()
            oct_optimizer.step()
            
            o_concept_sims = o_concept_sims.detach()
            
            target_onehot = torch.autograd.Variable((torch.zeros(f_samples.size()[0], args.n_classes).to(device)).scatter_(1, f_targets.view(f_targets.size()[0], 1).to(device), 1))
            s_label = dist_s_label(target_onehot, outputs.detach())
            t_label = dist_s_label(target_onehot, oct_outputs.detach())
    
            ps_pt = dist_s_t(oct_outputs.detach(), outputs.detach(), 1)
            epsilon = torch.exp(-1 * t_label / (s_label + t_label))
            delta = s_label - epsilon * t_label
            if ps_pt > delta and t_label < s_label:
                # ********* prototypical distill ************
                for o_sim, o_target in zip(o_concept_sims, o_targets):
                    if o_target in exist_classes:
                        o_dist[exist_classes.index(o_target)] += o_sim.detach()
                        o_num[exist_classes.index(o_target)] += 1
                        
                for f_sim, f_target, f_pred in zip(concept_sims, f_targets, preds):
                    if f_target in exist_classes:
                        f_dist[exist_classes.index(f_target)] += f_sim.detach()
                        f_num[exist_classes.index(f_target)] += 1
                
                f_dist = torch.div(f_dist, f_num.unsqueeze(-1).unsqueeze(-1))
                o_dist = torch.div(o_dist, o_num.unsqueeze(-1).unsqueeze(-1))
                loss_distill_sim = nn.MSELoss()(f_dist, o_dist) * args.beta_distill
            
                # ************** contrastive distill *************
                f_targets = f_targets.contiguous().view(-1, 1)
                mask = torch.eq(f_targets, f_targets.T).float().to(device)
                sims = torch.cat((concept_sims.unsqueeze(1), o_concept_sims.unsqueeze(1)), dim=1)
                
                contrast_count = sims.shape[1]
                contrast_feature = torch.cat(torch.unbind(sims, dim=1), dim=0)
                
                anchor_feature = contrast_feature
                anchor_count = contrast_count

                anchor_dot_contrast_a = torch.div(
                    torch.matmul(anchor_feature[:,0,:], contrast_feature[:,0,:].T),
                    args.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast_a, dim=1, keepdim=True)
                logits_a = anchor_dot_contrast_a - logits_max.detach()
                
                anchor_dot_contrast_b = torch.div(
                    torch.matmul(anchor_feature[:,1,:], contrast_feature[:,1,:].T),
                    args.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast_b, dim=1, keepdim=True)
                logits_b = anchor_dot_contrast_b - logits_max.detach()
                
                mask = mask.repeat(anchor_count, contrast_count)
                logits_mask = torch.scatter(
                    torch.ones_like(mask),
                    1,
                    torch.arange(args.batch_size * anchor_count).view(-1, 1).to(device),
                    0
                )
                mask = mask * logits_mask
                exp_logits_a = torch.exp(logits_a) * logits_mask
                log_prob_a = logits_a - torch.log(exp_logits_a.sum(1, keepdim=True))
                mean_log_prob_pos_a = (mask * log_prob_a).sum(1) / mask.sum(1)
                
                exp_logits_b = torch.exp(logits_b) * logits_mask
                log_prob_b = logits_b - torch.log(exp_logits_b.sum(1, keepdim=True))
                mean_log_prob_pos_b = (mask * log_prob_b).sum(1) / mask.sum(1)

                # loss
                loss_distill_contrast = - args.alpha_distill * mean_log_prob_pos_a * 0.5 - args.alpha_distill * mean_log_prob_pos_b * 0.5
                loss_distill_contrast = loss_distill_contrast.view(anchor_count, args.batch_size).mean()
                loss_distill = loss_distill_sim + loss_distill_contrast
                loss_dis_value = loss_distill.item()
            
            else:
                loss_distill = 0
                loss_dis_value = 0

            if not math.isnan(loss_dis_value):
                loss = loss_cls_f + loss_distill
            else:
                loss = loss_cls_f
                
            loss_cls_o_value = loss_cls_o.item()
            loss_cls_value = loss_cls_f.item()
            
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dis_value)
                print(loss_cls_f)
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metric_logger.update(loss_cls_o=loss_cls_o_value)
        metric_logger.update(loss_cls=loss_cls_value)
        metric_logger.update(loss_dis=loss_dis_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, num_class, write_pred=False):
    if args.modality == 'fundus':
        alldiseases = alldiseases_f
        cls_names = cls_names_f
    elif args.modality == 'oct':
        alldiseases = alldiseases_o
        cls_names = cls_names_o
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    prediction_decode_list = []
    prediction_prob_list = []
    true_label_decode_list = []
    img_name_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        images = batch[0]
        target = batch[1]
        img_names = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, target)
            
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        prediction_prob = torch.softmax(outputs, dim=1)
        prediction_decode = torch.argmax(prediction_prob, dim=1)
        true_label_decode = target
        
        prediction_decode_list.extend(prediction_decode.cpu().detach().numpy().tolist())
        true_label_decode_list.extend(true_label_decode.cpu().detach().numpy().tolist())
        prediction_prob_list.extend(prediction_prob.cpu().detach().numpy().tolist())
        img_name_list.extend(list(img_names))
        
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    prediction_prob_list = np.array(prediction_prob_list)
    
    if write_pred:
        init_dict = {'ImgName': [], 'Label': []}
        for d in alldiseases:
            init_dict[d] = []
        for img_name, pred, targets in zip(img_name_list, prediction_decode_list, true_label_decode_list):
            init_dict['ImgName'].extend([img_name, '', ])
            init_dict['Label'].extend(['pred', 'true'])
            for i, d in enumerate(alldiseases):
                init_dict[d].extend([pred[i], targets[i]])
        df = pd.DataFrame.from_dict(init_dict)
        df.to_excel(os.path.join(args.output_dir, 'test_result.xlsx'), float_format='%.4f')
    
    results = compute_metrics(true_label_decode_list, prediction_decode_list)
    class_wise_results = compute_classwise_metrics(true_label_decode_list, prediction_decode_list)

    print_result(class_wise_results, results, alldiseases)
    metric_logger.meters['kappa'].update(results['kappa'].item())
    metric_logger.meters['f1_pr'].update(results['f1'].item())
    metric_logger.meters['acc'].update(results['accuracy'].item())
    metric_logger.meters['precision'].update(results['precision'].item())
    metric_logger.meters['recall'].update(results['recall'].item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, results['f1']
