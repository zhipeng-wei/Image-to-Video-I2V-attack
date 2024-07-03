import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import random

from image_cam import GradCAM
from torch.autograd import Variable
from image_cam_utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer
import pickle as pkl

import time
from timm.models import create_model
import numpy as np

class Attack(object):
    """
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the model's training mode to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model=None):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        # mean and std values are used in pytorch pretrained models
        # they are also used in Kinetics-400.
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, *input):
        r"""
        It defines the computation performed at every call (attack forward).
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
    
    def _transform_perts(self, perts):
        dtype = perts.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype).cuda()
        std = torch.as_tensor(self.std, dtype=dtype).cuda()
        perts.div_(std[:, None, None])
        return perts

    def _transform_video(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype).cuda()
        std = torch.as_tensor(self.std, dtype=dtype).cuda()
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[:, None, None]).add_(mean[:, None, None])
        return video

    def _transform_video_ILAF(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype).cuda()
        std = torch.as_tensor(self.std, dtype=dtype).cuda()
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[None, :, None, None, None]).add_(mean[None, :, None, None, None])
        return video

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images

def get_vits():
    model = create_model(
                        'vit_base_patch16_224',
                        pretrained=True,
                        num_classes=1000,
                        in_chans=3,
                        global_pool=None,
                        scriptable=False)
    model.cuda()
    model.eval()
    return model

def get_model(model_name):
    '''
    ['alexnet', 'vgg', 'resnet', 'densenet', 'squeezenet']
    '''
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        # model.features[11/7/4/1] 
    elif model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        # model.features[29/20/11/1]
    elif model_name == 'resnet':
        model = models.resnet101(pretrained=True)
    elif model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        # model.features.denseblock1/2/3/4
        # model.features.transition1/2/3,norm5
    elif model_name == 'squeezenet':
        model = models.squeezenet1_1(pretrained=True)
        # model.features[12/9/6/3].expand3x3_activation
    model.cuda()
    model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         m.eval()
    return model

def get_models(model_name_lists):
    models = []
    for model_name in model_name_lists:
        model = get_model(model_name)
        models.append(model)
    return models
    
def get_GradCam(model_name_lists):
    gradcams = []
    for model_name in model_name_lists:
        model_dict = dict(type=model_name, arch=get_model(model_name), input_size=(224, 224))
        this_gradcam = GradCAM(model_dict, False)
        gradcams.append(this_gradcam)
    return gradcams 

class AENS_I2V_MF(Attack):
    '''
    The proposed adaptive I2V with multiple models and layers.
    Parameters:
        model_name_lists: the surrogate image model names. For example, model_name_lists = ['resnet', 'vgg', 'squeezenet', 'alexnet']
        depths: the layers used in each model. For example,  depths = {'resnet':[2,3], 'vgg':[2,3], 'squeezenet':[2,3], 'alexnet':[2,3]}
        step_size: the learning rate.
    Return:
        image_inps: video adversarial example.
        used_time: the time during attacking.
        cost_saved: the cost values of all steps
    '''
    def __init__(self, model_name_lists, depths, step_size, momentum=0, coef_CE=False, epsilon=16/255, steps=60):
        super(AENS_I2V_MF, self).__init__("AENS_I2V_MF")
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.loss_info = {}
        self.depths = depths
        self.momentum = momentum
        self.coef_CE = coef_CE
        self.models = get_models(model_name_lists)
        self.model_names = model_name_lists

        self.coeffs = torch.ones(len(model_name_lists)*2).cuda()
        # print ('using image models:', model_name_lists)

        for i in range(len(self.models)):
            self.models[i].train()
            for m in self.models[i].modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
            model_name = self.model_names[i]
            self._attention_hook(self.models[i], model_name)
            
    def _find_target_layer(self, model, model_name):
        used_depth = self.depths[model_name]
        if model_name == 'resnet':
            if isinstance(used_depth, list):
                return [getattr(model, 'layer{}'.format(this_depth))[-1] for this_depth in used_depth]
            else:
                return getattr(model, 'layer{}'.format(used_depth))[-1]
        elif model_name == 'alexnet':
            depth_to_layer = {1:1,2:4,3:7,4:11}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'vgg':
            depth_to_layer = {1:1,2:11,3:20,4:29}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'squeezenet':
            depth_to_layer = {1:3,2:6,3:9,4:12}
            if isinstance(used_depth, list):
                return [getattr(model, 'features')[depth_to_layer[this_depth]] for this_depth in used_depth]
            else:
                return getattr(model, 'features')[depth_to_layer[used_depth]].expand3x3_activation
            
    def _attention_hook(self, model, model_name):
        self.gradients = dict()
        self.gradients['value'] = []
        self.activations = dict()
        self.activations['value'] = []
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] += [grad_output[0]]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] += [output]
            return None
        target_layer = self._find_target_layer(model, model_name)
        # print (target_layer)
        if isinstance(target_layer, list):
            for i in target_layer:
                i.register_forward_hook(forward_hook)
                i.register_backward_hook(backward_hook)
        else:        
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)

    def forward(self, videos, labels, video_names):
        batch_size = videos.shape[0]
        b,c,f,h,w = videos.shape
        videos = videos.cuda()
        labels = labels.cuda()
        self.weights = []
        image_inps = videos.permute([0,2,1,3,4])
        image_inps = image_inps.reshape(b*f, c, h, w)

        # define modifer that updated by optimizer.
        modif = torch.Tensor(b*f, c, h, w).fill_(0.01/255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.step_size)

        unnorm_videos = self._transform_video(image_inps.clone().detach(), mode='back') # [0, 1]

        unnorm_videos = Variable(unnorm_videos, requires_grad=False)
        
        init_feature_maps = []
        for n in range(len(self.models)):
            this_feature_maps = []
            self.gradients = dict()
            self.gradients['value'] = []
            self.activations = dict()
            self.activations['value'] = []
            _ = self.models[n](image_inps)  
            for mm in range(len(self.activations['value'])):
                activations = self.activations['value'][mm]
                activations = Variable(activations, requires_grad=False)
                this_feature_maps.append(activations)
            init_feature_maps.append(this_feature_maps)

        begin = time.time()
        cost_saved = np.zeros(self.steps)
        previous_cs_loss = torch.ones_like(self.coeffs)
        for i in range(self.steps):
            # self.gradients = dict()
            # self.gradients['value'] = []
            # self.activations = dict()
            # self.activations['value'] = []

            # update coeff
            self.coeffs = torch.softmax(torch.softmax(previous_cs_loss, dim=0) + self.momentum * self.coeffs, dim=0)
            self.weights.append(self.coeffs.clone().cpu().numpy())
            # print (self.coeffs.clone().cpu().numpy())
            true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            true_image = self._transform_video(true_image, mode='forward') # norm

            losses = []
            for n in range(len(self.models)):
                self.gradients = dict()
                self.gradients['value'] = []
                self.activations = dict()
                self.activations['value'] = []
                _ = self.models[n](true_image)
                this_losses = []
                for mm in range(len(init_feature_maps[n])):
                    activations = self.activations['value'][mm]
                    init_activations = init_feature_maps[n][mm]
                    this_dir = activations.view(b*f, -1)
                    init_dir = init_activations.view(b*f, -1)
                    this_loss = F.cosine_similarity(this_dir, init_dir)
                    this_losses.append(this_loss)
                losses.append(torch.stack(this_losses)) # 2,32
            
            
            used_coeffs = torch.unsqueeze(self.coeffs, dim=1) # (lens_model*2) * 1
            each_features_loss = torch.sum(used_coeffs * torch.cat(losses, dim=0), dim=1) # 4*32
            cost = torch.mean(each_features_loss)
            
            if self.coef_CE:
                previous_cs_loss = each_features_loss.clone().detach()
            else:
                updated_features_loss = torch.sum(torch.cat(losses, dim=0).clone().detach(), dim=1)
                previous_cs_loss = updated_features_loss.clone().detach()

            # update previous_cs_loss
            
            # print (previous_cs_loss.clone().cpu().numpy())
            # print (cost)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            cost_saved[i] = cost.detach().item()

            for ind,vid_name in enumerate(video_names):
                if vid_name not in self.loss_info.keys():
                    self.loss_info[vid_name] = {}  
                self.loss_info[vid_name][i] = {'cost': str(cost.detach().cpu().numpy())}

        used_time = time.time()-begin

        true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
        image_inps = self._transform_video(true_image, mode='forward')
        image_inps = image_inps.reshape(b,f,c,h,w)
        image_inps = image_inps.permute([0,2,1,3,4])
        return image_inps, used_time, cost_saved
