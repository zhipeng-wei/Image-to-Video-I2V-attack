import numpy as np
import argparse
import os
import torch

import math
import json

import image_attacks
from dataset_ucf101 import attack_genearte_dataeset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, get_cfg_custom, OPT_PATH
import pickle as pkl

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    # parallel run
    parser.add_argument('--batch_nums', type=int, default=1)
    parser.add_argument('--batch_index', type=int, default=1)

    # parser.add_argument('--adv_path', type=str, default='', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for reference (default: 16)')
    parser.add_argument('--attack_method', type=str, default='ImageGuidedAttentionMap', help='')
    parser.add_argument('--step', type=int, default=10, metavar='N',
                    help='Multi-step or One-step in TI and SGM.')
    parser.add_argument('--file_prefix', type=str, default='')

    # for std
    parser.add_argument('--depth', type=int, default=1, help='1,2,3,4')
    parser.add_argument('--lamb', type=float, default=0.1, help='')

    parser.add_argument('--mode', type=str, default='direction', help='diff_norm\direction')
    parser.add_argument('--step_size', type=float, default=0.004, help='')
    
    # for dropout
    parser.add_argument('--dropout', type=float, default=0.1, help='')

    # for direction with changing image model
    parser.add_argument('--direction_image_model', type=str, default='resnet', help='resnet, densenet, squeezenet, vgg, alexnet')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, '{}-{}-{}-{}'.format('Image', args.attack_method, args.step, args.file_prefix))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print (args)

    # loading dataset and model.
    dataset_loader = attack_genearte_dataeset(args.batch_size)
    
    nums_contained = int(400 / args.batch_nums)
    left = (args.batch_index-1) * nums_contained
    right = args.batch_index * nums_contained

    # attack 
    if args.attack_method == 'ImageGuidedStd_Adam':
        model_name_lists = [args.direction_image_model]
        attack_method = getattr(image_attacks, args.attack_method)(model_name_lists, depth=args.depth, step_size=args.step_size, steps=args.step)
    elif args.attack_method == 'ImageGuidedFMDirection_Adam':
        model_name_lists = [args.direction_image_model]
        attack_method = getattr(image_attacks, args.attack_method)(model_name_lists, depth=args.depth, step_size=args.step_size, steps=args.step)
    elif args.attack_method == 'ImageGuidedFML2_Adam_MultiModels':
        model_name_lists = ['resnet', 'vgg', 'squeezenet', 'alexnet']
        depths = {
            'resnet':2,
            'vgg':3,
            'squeezenet':2,
            'alexnet':3
        }
        attack_method = getattr(image_attacks, args.attack_method)(model_name_lists, depths=depths, steps=args.step)

    for step, data in enumerate(dataset_loader):
        if step >= left and step < right:
            if step %1 == 0:
                print ('Running {}, {}/{}'.format(args.attack_method, step+1, len(dataset_loader)))
            val_batch = data[0]
            val_label = data[1]
            video_names = str(val_label)
            # np.save(os.path.join(args.adv_path, '{}-ori'.format(val_label[0].item())), val_batch[0].detach().cpu().numpy())
            adv_batches = attack_method(val_batch, val_label, video_names)
            for ind,label in enumerate(val_label):
                adv = adv_batches[ind].detach().cpu().numpy()            
                np.save(os.path.join(args.adv_path, '{}-adv'.format(label.item())), adv)
            
    with open(os.path.join(args.adv_path, 'loss_info_{}.json'.format(args.batch_index)), 'w') as opt:
        json.dump(attack_method.loss_info, opt)
