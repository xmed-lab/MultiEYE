import os
import argparse
import random
import time
import datetime

import torch
import numpy as np
import wandb

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

from data.dataset import build_dataset_single
from model.flair_single import FLAIRConceptClassifier
from process.finetune_single import train_one_epoch, evaluate
from utils.eval import save_model


def get_args_parser():
    parser = argparse.ArgumentParser('Multi Eye CLIP', add_help=False)
    parser.add_argument('--modality', default='oct', type=str, help='modality for training backbone model')
    parser.add_argument('--device_id', default='2', type=str, help='select device id')
    parser.add_argument('--device', default='cuda', type=str, help='device: cuda or cpu')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    
    parser.add_argument('--data_path', default='multieye_data/assemble_oct', type=str,help='dataset path')
    parser.add_argument('--concept_path', default='concepts', type=str, help='concept path')
    
    # Augmentation parameters
    parser.add_argument('--input_size', default=512, type=int,
                    help='images input size')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_samples', default=36000, type=int, help='number of the sampled training data')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--n_classes', default=9, type=int, help='number of the classification types')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    
    parser.add_argument('--print_freq', default=100, type=int, help='batch size')

    parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
    parser.add_argument('--output_dir', default='checkpoint/oct_checkpoint', help='path where to save, empty for no saving')
    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device, int(args.device_id))

    torch.backends.cudnn.benchmark = True
    
    train_dataset = build_dataset_single('train', args=args, mod=args.modality)
    dev_dataset = build_dataset_single('dev', args=args, mod=args.modality)
    test_dataset = build_dataset_single('test', args=args, mod=args.modality)

    weights = train_dataset.label_weights_for_balance()
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=args.num_samples, replacement=True)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=args.batch_size, 
        pin_memory=args.pin_mem, 
        shuffle=False, num_workers=args.num_workers,
        drop_last=False)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        pin_memory=args.pin_mem, 
        shuffle=False, num_workers=args.num_workers,
        drop_last=False)
    
    concept_feat_path = os.path.join(args.concept_path, 'concepts_raw.npy')
    model = FLAIRConceptClassifier(args, device, concept_feat_path)
    
    for p in model.flair_model.parameters():
        p.requires_grad = False
    for p in model.flair_model.vision_model.parameters():
        p.requires_grad = True
    model.concept_classifier.requires_grad = True
    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters = %s' % str(n_parameters))
    
    criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("optimizer = %s" % str(optimizer))
    
    if args.eval:
        test_stats, test_metric = evaluate(args, data_loader_test, model, device, num_class=args.n_classes)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_metric = 0.0
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch,
            args=args)

        val_stats, val_metric = evaluate(args, data_loader_val, model, device, num_class=args.n_classes)
        if max_metric < val_metric:
            max_metric = val_metric
            
            if args.output_dir:
                save_model(args=args, model=model, optimizer=optimizer, epoch=epoch, if_best=True)
            print('------ best model ------')
            if args.modality == 'fundus':
                test_stats, test_metric = evaluate(args, data_loader_test, model, device, num_class=args.n_classes)
            

        if epoch==(args.epochs-1):
            print('------ last model ------')
            if args.modality == 'fundus':
                test_stats, test_metric = evaluate(args, data_loader_test, model, device, num_class=args.n_classes)
            if args.output_dir:
                save_model(args=args, model=model, optimizer=optimizer, epoch=epoch, if_best=False)
        
        wandb_log = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': v for k, v in val_stats.items()}}
        wandb.log(wandb_log)

                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="multieye",
        name="oct pre-train model",
        # track hyperparameters and run metadata
        config={
            "modality": args.modality,
            "learning_rate": args.lr,
            "output_dir": args.output_dir,
            "num_classes": args.n_classes,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs
        }
    )
    
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
    