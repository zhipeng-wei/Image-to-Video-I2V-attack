import numpy as np
import argparse
import os
import torch
import math
import json
from torch.utils.data import Dataset, DataLoader

import image_attacks
from datasets import get_dataset
from gluoncv.torch.model_zoo import get_model
from utils import CONFIG_PATHS, get_cfg_custom
import pickle as pkl
from reference_ucf101 import MODEL_TO_CKPTS

class AdvDataset(Dataset):
    def __init__(self, used_adv_path, used_ori_path):
        self.used_adv_path = used_adv_path
        files = os.listdir(self.used_adv_path)
        self.files = [i for i in files if 'adv' in i]
        self.used_ori_path = used_ori_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        vid_id = file.split('-')[0]
        ori_file = os.path.join(self.used_ori_path, '{}-ori.npy'.format(vid_id))
        vid = torch.from_numpy(np.load(os.path.join(self.used_adv_path, file)))
        vid = vid[None]
        ori_vid = torch.from_numpy(np.load(ori_file))
        ori_vid = ori_vid[None]
        label = [int(file.split('-')[0])]
        label = np.array(label).astype(np.int32)
        label = torch.from_numpy(label).long()
        return vid, ori_vid, label


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for reference (default: 16)')
    parser.add_argument('--attack_method', type=str, default='ILAF', help='')
    parser.add_argument('--opt_path', type=str, default='')
    # adv path
    parser.add_argument('--used_adv', type=str, default='', help='')
    parser.add_argument('--used_ori', type=str, default='', help='')
    # white-box model
    parser.add_argument('--white_model', type=str, default='i3d_resnet101', help='i3d_resnet101 | slowfast_resnet101 | tpn_resnet101')
    parser.add_argument('--dataset', type=str, default='Kinetics-400', help='Kinetics-400 | UCF-101')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print (args)
    # loading cfg.
    cfg_path = CONFIG_PATHS[args.white_model]
    cfg = get_cfg_custom(cfg_path, args.batch_size)
    model = get_model(cfg)
    if args.dataset == 'UCF-101':
        ckpt_path = MODEL_TO_CKPTS[args.white_model]
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.cuda()

    # loading dataset.
    dataset = AdvDataset(used_adv_path = args.used_adv, used_ori_path=args.used_ori)
    
    attack_method = getattr(image_attacks, args.attack_method)(model, args.white_model)
    for step in range(len(dataset)):
        if step %1 == 0:
            print ('Running {}, {}/{}'.format(args.attack_method, step+1, len(dataset)))
        # val_batch, val_label = generate_batch(files_batch[step])
        val_batch, ori_batch, val_label = dataset[step]
        video_names = ['...']
        adv_batches = attack_method(val_batch, ori_batch, val_label, video_names)
        for ind,label in enumerate(val_label):
            adv = adv_batches[ind].detach().cpu().numpy()
            np.save(os.path.join(args.opt_path, '{}-adv'.format(label.item())), adv)