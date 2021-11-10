import argparse
import os
import torch
import numpy as np
import math

import base_attacks
import video_attacks
from dataset_ucf101 import attack_genearte_dataeset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, get_cfg_custom, OPT_PATH
from reference_ucf101 import MODEL_TO_CKPTS

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--adv_path', type=str, default='', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for reference (default: 16)')
    parser.add_argument('--model', type=str, default='i3d_resnet101', help='i3d_resnet101 | i3d_slow_resnet101 | slowfast_resnet101 | tpn_resnet101.')
    parser.add_argument('--attack_method', type=str, default='TemporalAugmentationMomentum', help='FGSM | BIM | MIFGSM | DIFGSM | TIFGSM | SGM')
    parser.add_argument('--attack_type', type=str, default='image', help='image | video')
    parser.add_argument('--step', type=int, default=10, metavar='N',
                    help='Multi-step or One-step in TI and SGM.')
    parser.add_argument('--sf_frame', type=int, default=32, metavar='N',
                    help='SFFGSM frame.')
    parser.add_argument('--cf_frame', type=str, default='small', metavar='N',
                    help='CFFGSM frame.')
    parser.add_argument('--kernlen', type=int, default=15, metavar='N',
                    help='SFFGSM frame.')
    parser.add_argument('--nsig', type=int, default=3, metavar='N',
                    help='SFFGSM frame.')
    parser.add_argument('--file_prefix', type=str, default='')
    parser.add_argument('--kernel_mode', type=str, default='gaussian')
    parser.add_argument('--iterative_momentum', action='store_true', default=False, help='Use iterative momentum in MFFGSM.')
    parser.add_argument('--frame_conv', action='store_true', default=False, help='Use frame_conv in MFFGSM.')
    # for TemporalAugmentationMomentum
    parser.add_argument('--augmentation_weight', type=float, default=1.0, help='')
    parser.add_argument('--frame_momentum', action='store_true', default=False, help='')
    parser.add_argument('--gamma', type=float, default=1.0, help='')
    # for combine momentum
    parser.add_argument('--no_iterative_momentum', action='store_true', default=False, help='')
    parser.add_argument('--weight_add', action='store_true', default=False, help='')
    parser.add_argument('--momentum_weight', type=float, default=0.5, help='')
    parser.add_argument('--iterative_first', action='store_true', default=False, help='')
    # for TemporalAugmentation
    parser.add_argument('--translation_invariant', action='store_true', default=False, help='')
    parser.add_argument('--temporal_augmentation', action='store_true', default=False, help='')
    parser.add_argument('--TI_First', action='store_true', default=False, help='')
    # for noise and shuffle
    parser.add_argument('--noise', action='store_true', default=False, help='')
    parser.add_argument('--shuffle_grads', action='store_true', default=False, help='')
    # for cycle move
    parser.add_argument('--move_type', type=str, default='adj',help='adj | large | random')
    args = parser.parse_args()
    if args.attack_type == 'video':
        args.adv_path = os.path.join(OPT_PATH, 'UCF101_Video-{}-{}-{}-{}'.format(args.model, args.attack_method, args.step, args.file_prefix))
    elif args.attack_type == 'image':
        args.adv_path = os.path.join(OPT_PATH, 'UCF101_Image-{}-{}-{}-{}'.format(args.model, args.attack_method, args.step, args.file_prefix))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print (args)

    # loading cfg.
    cfg_path = CONFIG_PATHS[args.model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)

    # loading dataset and model.
    dataset_loader = attack_genearte_dataeset(args.batch_size)
    ckpt_path = MODEL_TO_CKPTS[args.model]
    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.cuda()
    model.eval()

    # attack
    if args.attack_type == 'image':
        # FGSM, BIM, MIFGSM, DIFGSM, TIFGSM, SGM, SIM
        attack_method = getattr(base_attacks, args.attack_method)(model, steps=args.step)
    elif args.attack_type == 'video':
        if args.attack_method == 'TemporalTranslation':
            spe_params = {'kernlen':15, 'momentum':False, 'weight':1.0, 'move_type':'adj', 'kernel_mode':'gaussian'}
        print ('Used Params')
        print (spe_params)
        attack_method = getattr(video_attacks, args.attack_method)(model, params=spe_params, steps=args.step)

    for step, data in enumerate(dataset_loader):
        if step %1 == 0:
            print ('Running {}, {}/{}'.format(args.attack_method, step+1, len(dataset_loader)))
        val_batch = data[0].cuda()
        val_label = data[1].cuda()
        adv_batches = attack_method(val_batch, val_label)
        val_batch = val_batch.detach()
        for ind,label in enumerate(val_label):
            ori = val_batch[ind].cpu().numpy()
            adv = adv_batches[ind].cpu().numpy()
            np.save(os.path.join(args.adv_path, '{}-adv'.format(label.item())), adv)
            np.save(os.path.join(args.adv_path, '{}-ori'.format(label.item())), ori)
