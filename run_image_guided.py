import numpy as np
import os
import argparse

ImageGuidedFMDirection_Adam_attack_run = 'python image_main.py --gpu {gpu} --attack_method ImageGuidedFMDirection_Adam  --step {step} --step_size {step_size} --direction_image_model resnet --batch_size {batch_size} --batch_nums {batch_nums} --batch_index {batch_index} --file_prefix resnet_step_size_{step_size}_paper_study'
ImageGuidedFMDirection_Adam_reference_run = 'python reference.py --gpu {gpu} --adv_path Image-ImageGuidedFMDirection_Adam-{step}-resnet_step_size_{step_size}_paper_study'

Aba_layer_ImageGuidedFMDirection_Adam_attack_run = 'python image_main.py --gpu {gpu} --attack_method ImageGuidedFMDirection_Adam  --step 60 --step_size 0.005 --direction_image_model {image_model} --depth {depth} --file_prefix {image_model}-step_size-0.005-depth-{depth}_paper_study'
Aba_layer_ImageGuidedFMDirection_Adam_reference_run = 'python reference.py --gpu {gpu} --adv_path Image-ImageGuidedFMDirection_Adam-60-{image_model}-step_size-0.005-depth-{depth}_paper_study'

# Kinetic400
Per_Com_ImageGuidedFMDirection_Adam_attack_run = 'python image_main.py --gpu {gpu} --attack_method ImageGuidedFMDirection_Adam  --step 60 --step_size 0.005 --direction_image_model {image_model} --depth {depth} --file_prefix {image_model}-depth-{depth}_paper_per_com'
Per_Com_ImageGuidedFMDirection_Adam_reference_run = 'python reference.py --gpu {gpu} --adv_path Image-ImageGuidedFMDirection_Adam-60-{image_model}-depth-{depth}_paper_per_com'

Per_Com_ImageGuidedFML2_Adam_MultiModels_attack_run = 'python image_main.py --gpu {gpu} --attack_method ImageGuidedFML2_Adam_MultiModels  --step 60 --step_size 0.005 --file_prefix paper_per_com'
Per_Com_ImageGuidedFML2_Adam_MultiModels_reference_run = 'python reference.py --gpu {gpu} --adv_path Image-ImageGuidedFML2_Adam_MultiModels-60-paper_per_com'

Per_Com_ImageGuidedStd_Adam_attack_run = 'python image_main.py --gpu {gpu} --attack_method ImageGuidedStd_Adam  --step 60 --step_size 0.005 --direction_image_model {image_model} --depth {depth} --file_prefix {image_model}-depth-{depth}_paper_per_com'
Per_Com_ImageGuidedStd_Adam_reference_run = 'python reference.py --gpu {gpu} --adv_path Image-ImageGuidedStd_Adam-60-{image_model}-depth-{depth}_paper_per_com'

# UCF101
Per_Com_UCF_ImageGuidedFMDirection_Adam_attack_run = 'python image_main_ucf101.py --gpu {gpu} --attack_method ImageGuidedFMDirection_Adam  --step 60 --step_size 0.005 --direction_image_model {image_model} --depth {depth} --file_prefix {image_model}-depth-{depth}_paper_per_com'
Per_Com_UCF_ImageGuidedFMDirection_Adam_reference_run = 'python reference_ucf101.py --gpu {gpu} --adv_path Image-ImageGuidedFMDirection_Adam-60-{image_model}-depth-{depth}_paper_per_com'

Per_Com_UCF_ImageGuidedFML2_Adam_MultiModels_attack_run = 'python image_main_ucf101.py --gpu {gpu} --attack_method ImageGuidedFML2_Adam_MultiModels  --step 60 --step_size 0.005 --file_prefix paper_per_com'
Per_Com_UCF_ImageGuidedFML2_Adam_MultiModels_reference_run = 'python reference_ucf101.py --gpu {gpu} --adv_path Image-ImageGuidedFML2_Adam_MultiModels-60-paper_per_com'

Per_Com_UCF_ImageGuidedStd_Adam_attack_run = 'python image_main_ucf101.py --gpu {gpu} --attack_method ImageGuidedStd_Adam  --step 60 --step_size 0.005 --direction_image_model {image_model} --depth {depth} --file_prefix {image_model}-depth-{depth}_paper_per_com'
Per_Com_UCF_ImageGuidedStd_Adam_reference_run = 'python reference_ucf101.py --gpu {gpu} --adv_path Image-ImageGuidedStd_Adam-60-{image_model}-depth-{depth}_paper_per_com'



def arg_parse():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--adv_path', type=str, default='', help='the path of adversarial examples.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    # parser.add_argument('--kernlens', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()

    # ablation study for step size and iteration number (Figure 4)
    steps = [20, 40, 60, 80, 100]
    step_sizes = [0.001, 0.0025, 0.0050, 0.0075, 0.010]
    for step in steps:
        # step_size = 0.004
        for step_size in step_sizes:
            os.system(ImageGuidedFMDirection_Adam_attack_run.format(gpu=args.gpu, step=step, step_size=step_size, batch_nums=1, batch_index=1, batch_size=1))
            os.system(ImageGuidedFMDirection_Adam_reference_run.format(gpu=args.gpu, step=step, step_size=step_size))

    # ablation study for attacked layer (Table 2 and Figure 5)
    depths = [1,2,3,4]
    image_models = ['resnet', 'squeezenet', 'vgg', 'alexnet']
    for image_model in image_models:
        for depth in depths:
            os.system(Aba_layer_ImageGuidedFMDirection_Adam_attack_run.format(gpu=args.gpu, depth=depth, image_model=image_model))
            os.system(Aba_layer_ImageGuidedFMDirection_Adam_reference_run.format(gpu=args.gpu, image_model=image_model, depth=depth))

    # performance comparison for kinetics-400 (Table 3)
    step = 60
    step_size = 0.005
    image_models = ['squeezenet', 'vgg', 'alexnet', 'resnet'] 
    for image_model in image_models:
        if image_model == 'resnet' or image_model == 'squeezenet':
            depth = 2
        else:
            depth = 3
        # I2V attack
        os.system(Per_Com_ImageGuidedFMDirection_Adam_attack_run.format(gpu=args.gpu, image_model=image_model, depth=depth))
        os.system(Per_Com_ImageGuidedFMDirection_Adam_reference_run.format(gpu=args.gpu, image_model=image_model, depth=depth))
        # STD attack
        os.system(Per_Com_ImageGuidedStd_Adam_attack_run.format(gpu=args.gpu, image_model=image_model, depth=depth))
        os.system(Per_Com_ImageGuidedStd_Adam_reference_run.format(gpu=args.gpu, image_model=image_model, depth=depth))

    # ENS-I2V attack
    os.system(Per_Com_ImageGuidedFML2_Adam_MultiModels_attack_run.format(gpu=args.gpu))
    os.system(Per_Com_ImageGuidedFML2_Adam_MultiModels_reference_run.format(gpu=args.gpu))

    # performance compasison for ucf101 (Table 4)
    step = 60
    step_size = 0.005
    image_models = ['resnet', 'squeezenet', 'vgg', 'alexnet'] # '', 
    for image_model in image_models:
        if image_model == 'resnet' or image_model == 'squeezenet':
            depth = 2
        else:
            depth = 3
        # I2V attack
        os.system(Per_Com_UCF_ImageGuidedFMDirection_Adam_attack_run.format(gpu=args.gpu, image_model=image_model, depth=depth))
        os.system(Per_Com_UCF_ImageGuidedFMDirection_Adam_reference_run.format(gpu=args.gpu, image_model=image_model, depth=depth))
        # STD attack
        os.system(Per_Com_UCF_ImageGuidedStd_Adam_attack_run.format(gpu=args.gpu, image_model=image_model, depth=depth))
        os.system(Per_Com_UCF_ImageGuidedStd_Adam_reference_run.format(gpu=args.gpu, image_model=image_model, depth=depth))
        
    # ENS-I2V attack
    os.system(Per_Com_UCF_ImageGuidedFML2_Adam_MultiModels_attack_run.format(gpu=args.gpu))
    os.system(Per_Com_UCF_ImageGuidedFML2_Adam_MultiModels_reference_run.format(gpu=args.gpu))
    


