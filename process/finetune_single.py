import math
import sys
import os

import torch
import torch.nn as nn
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

def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, args):
    
    if args.modality == 'fundus':
        cls_names = cls_names_f
    elif args.modality == 'oct':
        cls_names = cls_names_o
    
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    accum_iter = args.accum_iter

    for data_iter_step, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        if data_iter_step % accum_iter == 0:    
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device)
        targets = targets.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            
            loss = criterion(outputs, targets)
            
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
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
    elif args.modality == 'oct':
        alldiseases = alldiseases_o
    
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
