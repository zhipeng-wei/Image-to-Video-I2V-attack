import os
import time
import numpy as np
import pandas as pd
import json

import torch
from gluoncv.torch.engine.config import get_cfg_defaults
from gluoncv.torch.model_zoo import get_model
from utils import AverageMeter, OPT_PATH, CONFIG_ROOT, UCF_CKPT_PATH
from datasets import get_dataset
import argparse
import math

CONFIG_PATHS = {
    'i3d_resnet50': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet50_v1_kinetics400.yaml'),
    'i3d_resnet101': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet101_v1_kinetics400.yaml'),
    'slowfast_resnet50': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet50_kinetics400.yaml'),
    'slowfast_resnet101': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet101_kinetics400.yaml'),
    'tpn_resnet50': os.path.join(CONFIG_ROOT, 'tpn_resnet50_f32s2_kinetics400.yaml'),
    'tpn_resnet101': os.path.join(CONFIG_ROOT, 'tpn_resnet101_f32s2_kinetics400.yaml')
    }

MODEL_TO_CKPTS = {
    'i3d_resnet50': os.path.join(UCF_CKPT_PATH, 'i3d_resnet50.pth'),
    'i3d_resnet101': os.path.join(UCF_CKPT_PATH, 'i3d_resnet101.pth'),
    'slowfast_resnet50': os.path.join(UCF_CKPT_PATH, 'slowfast_resnet50.pth'),
    'slowfast_resnet101': os.path.join(UCF_CKPT_PATH, 'slowfast_resnet101.pth'),
    'tpn_resnet50': os.path.join(UCF_CKPT_PATH, 'tpn_resnet50.pth'),
    'tpn_resnet101': os.path.join(UCF_CKPT_PATH, 'tpn_resnet101.pth')
}

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--adv_path', type=str, default='', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for reference (default: 16)')
    args = parser.parse_args()
    if 'DATACENTER' in args.adv_path:
        pass
    else:
        args.adv_path = os.path.join(OPT_PATH, args.adv_path)
    return args
    
def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t() # batch_size, 1
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0)
    return correct_k.mul_(100.0 / batch_size), torch.squeeze(pred)

def generate_batch(batch_files):
    batches = []
    labels = []
    for file in batch_files:
        batches.append(torch.from_numpy(np.load(os.path.join(args.adv_path, file))).cuda())
        labels.append(int(file.split('-')[0]))
    labels = np.array(labels).astype(np.int32)
    labels = torch.from_numpy(labels)
    return torch.stack(batches), labels

def reference(model, files_batch):
    data_time = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()

    predictions = []
    labels = []

    end = time.time()
    with torch.no_grad():
        for step, batch in enumerate(files_batch):
            data_time.update(time.time() - end)
            val_batch, val_label = generate_batch(batch)

            val_batch = val_batch.cuda()
            val_label = val_label.cuda()

            batch_size = val_label.size(0)
            outputs = model(val_batch)

            prec1a, preds = accuracy(outputs.data, val_label)

            predictions += list(preds.cpu().numpy())
            labels += list(val_label.cpu().numpy())

            top1.update(prec1a.item(), val_batch.size(0))   
            batch_time.update(time.time() - end)
            end = time.time()

            if step % 5 == 0:
                print('----validation----')
                print_string = 'Process: [{0}/{1}]'.format(step + 1, len(files_batch))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'top-1 accuracy: {top1_acc:.2f}%'.format(top1_acc = top1.avg)
                print (print_string)
    return predictions, labels, top1.avg

def load_model(model_name):
    cfg = get_cfg_defaults()
    cfg_path = CONFIG_PATHS[model_name]
    cfg.merge_from_file(cfg_path)
    cfg.CONFIG.MODEL.PRETRAINED = False
    ckpt_path = MODEL_TO_CKPTS[model_name]
    model = get_model(cfg)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.cuda()
    model.eval()
    return model

if __name__ == '__main__':
    global args
    args = arg_parse()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # loading adversarial examples.
    files = os.listdir(args.adv_path)
    files = [i for i in files if 'adv' in i]

    batch_times = math.ceil(len(files) / args.batch_size)
    files_batch = []
    for i in range(batch_times):
        batch = files[i*args.batch_size: min((i+1)*args.batch_size, len(files))]
        files_batch.append(batch)

    model_val_acc = {}
    info_df = pd.DataFrame()
    info_df['gt_label'] = [i for i in range(101)]
    for model_name in MODEL_TO_CKPTS.keys():
        print ('Model-{}:'.format(model_name))
        model = load_model(model_name)
        preds, labels, top1_avg = reference(model, files_batch)

        predd = np.zeros_like(preds)
        inds = np.argsort(labels)
        for i,ind in enumerate(inds):
            predd[ind] = preds[i]

        print (args.adv_path)
        info_df['{}-pre'.format(model_name)] = predd
        model_val_acc[model_name] = top1_avg
        del model
        torch.cuda.empty_cache()

    info_df.to_csv(os.path.join(args.adv_path, 'results_all_models_prediction.csv'), index=False)
    with open(os.path.join(args.adv_path, 'top1_acc_all_models.json'), 'w') as opt:
        json.dump(model_val_acc, opt)
    # delete the generated npy files.
    # command = os.path.join(args.adv_path, '*.npy')
    # os.system('rm {}'.format(command))

    

