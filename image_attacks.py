import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import random

from image_cam import GradCAM
from torch.autograd import Variable
from image_cam_utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer
import pickle as pkl

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

# *****************************************************************
# paper: Enhancing Cross-Task Black-Box Transferability of 
# Adversarial Examples with Dispersion Reduction
# *****************************************************************
class ImageGuidedStd_Adam(Attack):
    '''
    Dispersion Reduction (DR) attack.
    paper: Enhancing crosstask black-box transferability of adversarial examples with dispersion reduction
    parameters:
        depth: {1,2,3,4}
    '''
    def __init__(self, model_name_lists, depth, step_size, epsilon=16/255, steps=10):
        super(ImageGuidedStd_Adam, self).__init__("ImageGuidedStd_Adam")
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.loss_info = {}
        self.depth = depth
        self.model = get_models(model_name_lists)[0]
        self.model_name = model_name_lists[0]

        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

        self._attention_hook()

    def _find_target_layer(self):
        if self.model_name == 'resnet':
            return getattr(self.model, 'layer{}'.format(self.depth))[-1]
        elif self.model_name == 'alexnet':
            depth_to_layer = {1:1,2:4,3:7,4:11}
            return getattr(self.model, 'features')[depth_to_layer[self.depth]]
        elif self.model_name == 'vgg':
            depth_to_layer = {1:1,2:11,3:20,4:29}
            return getattr(self.model, 'features')[depth_to_layer[self.depth]]
        elif self.model_name == 'squeezenet':
            depth_to_layer = {1:3,2:6,3:9,4:12}
            return getattr(self.model, 'features')[depth_to_layer[self.depth]].expand3x3_activation

    def _attention_hook(self):
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
        target_layer = self._find_target_layer()
        print (target_layer)
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

        image_inps = videos.permute([0,2,1,3,4])
        image_inps = image_inps.reshape(b*f, c, h, w)

        # define modifer that updated by optimizer.
        modif = torch.Tensor(b*f, c, h, w).fill_(0.01/255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.step_size)

        unnorm_videos = self._transform_video(image_inps.clone().detach(), mode='back') # [0, 1]

        unnorm_videos = Variable(unnorm_videos, requires_grad=False)

        for i in range(self.steps):
            self.gradients = dict()
            self.gradients['value'] = []
            self.activations = dict()
            self.activations['value'] = []

            true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            true_image = self._transform_video(true_image, mode='forward') # norm

            _ = self.model(true_image)

            std_losses = []
            for mm in range(len(self.activations['value'])):
                activations = self.activations['value'][mm].std()
                std_losses.append(activations)
            cost = torch.sum(torch.stack(std_losses))
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            for ind,vid_name in enumerate(video_names):
                if vid_name not in self.loss_info.keys():
                    self.loss_info[vid_name] = {}  
                self.loss_info[vid_name][i] = {'cost': str(cost.detach().cpu().numpy())}

        true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
        image_inps = self._transform_video(true_image, mode='forward')
        image_inps = image_inps.reshape(b,f,c,h,w)
        image_inps = image_inps.permute([0,2,1,3,4])
        return image_inps

class ImageGuidedFMDirection_Adam(Attack):
    '''
    The proposed Image to Video (I2V) attack.
    parameters:
        depth: {1,2,3,4}
        model_name_lists: [a model name]
    '''
    def __init__(self, model_name_lists, depth, step_size, epsilon=16/255, steps=10):
        super(ImageGuidedFMDirection_Adam, self).__init__("ImageGuidedFMDirection_Adam")
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.loss_info = {}
        self.depth = depth
        self.model = get_models(model_name_lists)[0]
        self.model_name = model_name_lists[0]

        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.eval()

        self._attention_hook()

    def _find_target_layer(self):
            if self.model_name == 'resnet':
                return getattr(self.model, 'layer{}'.format(self.depth))[-1]
            elif self.model_name == 'alexnet':
                depth_to_layer = {1:1,2:4,3:7,4:11}
                return getattr(self.model, 'features')[depth_to_layer[self.depth]]
            elif self.model_name == 'vgg':
                depth_to_layer = {1:1,2:11,3:20,4:29}
                return getattr(self.model, 'features')[depth_to_layer[self.depth]]
            elif self.model_name == 'squeezenet':
                depth_to_layer = {1:3,2:6,3:9,4:12}
                return getattr(self.model, 'features')[depth_to_layer[self.depth]].expand3x3_activation
                
    def _attention_hook(self):
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
        target_layer = self._find_target_layer()
        print (target_layer)
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

        image_inps = videos.permute([0,2,1,3,4])
        image_inps = image_inps.reshape(b*f, c, h, w)

        # define modifer that updated by optimizer.
        modif = torch.Tensor(b*f, c, h, w).fill_(0.01/255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.step_size)

        unnorm_videos = self._transform_video(image_inps.clone().detach(), mode='back') # [0, 1]

        unnorm_videos = Variable(unnorm_videos, requires_grad=False)

        # initial feature map
        self.gradients = dict()
        self.gradients['value'] = []
        self.activations = dict()
        self.activations['value'] = []

        _ = self.model(image_inps)
        init_feature_maps = []
        for mm in range(len(self.activations['value'])):
            activations = self.activations['value'][mm]
            activations = Variable(activations, requires_grad=False)
            init_feature_maps.append(activations)

        for i in range(self.steps):
            self.gradients = dict()
            self.gradients['value'] = []
            self.activations = dict()
            self.activations['value'] = []

            true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            true_image = self._transform_video(true_image, mode='forward') # norm

            _ = self.model(true_image)

            losses = []
            for mm in range(len(init_feature_maps)):
                activations = self.activations['value'][mm]
                init_activations = init_feature_maps[mm]

                this_dir = activations.view(b*f, -1)
                init_dir = init_activations.view(b*f, -1)
                this_loss = F.cosine_similarity(this_dir, init_dir)
                flag = 1 # decrease this_loss

                losses.append(this_loss)
            cost = flag * torch.sum(torch.stack(losses))
            
            print (cost)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            for ind,vid_name in enumerate(video_names):
                if vid_name not in self.loss_info.keys():
                    self.loss_info[vid_name] = {}  
                self.loss_info[vid_name][i] = {'cost': str(cost.detach().cpu().numpy())}

        true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
        image_inps = self._transform_video(true_image, mode='forward')
        image_inps = image_inps.reshape(b,f,c,h,w)
        image_inps = image_inps.permute([0,2,1,3,4])
        return image_inps

class ImageGuidedFML2_Adam_MultiModels(Attack):
    '''
    The proposed ensemble Image to Video (ENS-I2V) attack.
    parameters:
        depth: {1,2,3,4}
    '''
    def __init__(self, model_name_lists, depths, epsilon=16/255, steps=60):
        super(ImageGuidedFML2_Adam_MultiModels, self).__init__("ImageGuidedFML2_Adam_MultiModels")
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = 0.005
        self.loss_info = {}
        self.depths = depths
        self.models = get_models(model_name_lists)
        self.model_names = model_name_lists
        print (model_name_lists)
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
            return getattr(model, 'layer{}'.format(used_depth))[-1]
        elif model_name == 'alexnet':
            depth_to_layer = {1:1,2:4,3:7,4:11}
            return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'vgg':
            depth_to_layer = {1:1,2:11,3:20,4:29}
            return getattr(model, 'features')[depth_to_layer[used_depth]]
        elif model_name == 'squeezenet':
            depth_to_layer = {1:3,2:6,3:9,4:12}
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
        print (target_layer)
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

        image_inps = videos.permute([0,2,1,3,4])
        image_inps = image_inps.reshape(b*f, c, h, w)

        # define modifer that updated by optimizer.
        modif = torch.Tensor(b*f, c, h, w).fill_(0.01/255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.step_size)

        unnorm_videos = self._transform_video(image_inps.clone().detach(), mode='back') # [0, 1]

        unnorm_videos = Variable(unnorm_videos, requires_grad=False)

        # initial feature map
        self.gradients = dict()
        self.gradients['value'] = []
        self.activations = dict()
        self.activations['value'] = []
        
        for n in range(len(self.models)):
            _ = self.models[n](image_inps)
        # _ = self.model(image_inps)
        init_feature_maps = []
        for mm in range(len(self.activations['value'])):
            activations = self.activations['value'][mm]
            activations = Variable(activations, requires_grad=False)
            init_feature_maps.append(activations)

        for i in range(self.steps):
            self.gradients = dict()
            self.gradients['value'] = []
            self.activations = dict()
            self.activations['value'] = []

            true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
            true_image = self._transform_video(true_image, mode='forward') # norm

            # _ = self.model(true_image)
            for n in range(len(self.models)):
                _ = self.models[n](true_image)
            losses = []
            for mm in range(len(init_feature_maps)):
                activations = self.activations['value'][mm]
                init_activations = init_feature_maps[mm]
                this_dir = activations.view(b*f, -1)
                init_dir = init_activations.view(b*f, -1)
                this_loss = F.cosine_similarity(this_dir, init_dir)
                flag = 1 # decrease this_loss
                losses.append(this_loss)
            cost = flag * torch.sum(torch.stack(losses))
            
            print (cost)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            for ind,vid_name in enumerate(video_names):
                if vid_name not in self.loss_info.keys():
                    self.loss_info[vid_name] = {}  
                self.loss_info[vid_name][i] = {'cost': str(cost.detach().cpu().numpy())}

        true_image = torch.clamp(unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)
        image_inps = self._transform_video(true_image, mode='forward')
        image_inps = image_inps.reshape(b,f,c,h,w)
        image_inps = image_inps.permute([0,2,1,3,4])
        return image_inps

class ILAF(Attack):
    '''
    ILAF. Paper: Enhancing adversarial example transferability with an intermediate level attack.
    '''
    def __init__(self, model, model_type, step_size=0.005, epsilon=16/255, steps=60):
        super(ILAF, self).__init__("ILAF")
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.loss_info = {}
        self.model_type = model_type
        self.model = model
        
        self._activation_hook()

    def _find_target_layer(self):
        if 'i3d' in self.model_type:
            return self.model.res_layers._modules['1']
        elif 'slowfast' in self.model_type:
            return [self.model._modules['slow_res2'], self.model._modules['fast_res2']] #[b,2048, 8, 7, 7], [b, 256, 32, 7, 7]
        elif 'tpn' in self.model_type:
            return self.model.layer2
                
    def _activation_hook(self):
        self.activations = dict()
        self.activations['value'] = []
        def forward_hook(module, input, output):
            self.activations['value'] += [output]
            return None
        target_layer = self._find_target_layer()
        if isinstance(target_layer, list):
            for i in target_layer:
                i.register_forward_hook(forward_hook)
        else:        
            target_layer.register_forward_hook(forward_hook)

    def forward(self, videos, ori_videos, labels, video_names):
        batch_size = videos.shape[0]
        b,c,f,h,w = videos.shape
        videos = videos.cuda()
        labels = labels.cuda()
        ori_videos = ori_videos.cuda()

        # ori feature map
        ori_feature_maps = []
        self.activations = dict()
        self.activations['value'] = []
        with torch.no_grad():
            _ = self.model(ori_videos)
        for mm in range(len(self.activations['value'])):
            activations = self.activations['value'][mm]
            ori_feature_maps.append(activations)

        # existed adv feature map
        adv_feature_maps = []
        self.activations = dict()
        self.activations['value'] = []
        with torch.no_grad():
            _ = self.model(videos)
        for mm in range(len(self.activations['value'])):
            activations = self.activations['value'][mm]
            adv_feature_maps.append(activations)

        init_directions = [] # normalized direction
        init_norms = [] # norm values
        for ori_di, adv_di in zip(ori_feature_maps, adv_feature_maps):
            init_direction = adv_di - ori_di
            norm = torch.norm(init_direction, p=2)
            init_norms.append(norm)
            init_directions.append(init_direction/torch.norm(init_direction,p=2,keepdim=True))

        
        adv_unnorm_videos = self._transform_video_ILAF(videos.clone().detach(), mode='back') # [0, 1]
        ori_unnorm_videos = self._transform_video_ILAF(ori_videos.clone().detach(), mode='back') # [0, 1]

        existed_perturbations = adv_unnorm_videos - ori_unnorm_videos
        modifier = torch.Tensor(existed_perturbations.cpu()).cuda()
        ori_unnorm_videos = Variable(ori_unnorm_videos, requires_grad=False)

        del adv_feature_maps, adv_unnorm_videos, videos, ori_videos
        torch.cuda.empty_cache() 
        for i in range(self.steps):
            modifier.requires_grad = True
            self.activations = dict()
            self.activations['value'] = []

            true_image = torch.clamp(ori_unnorm_videos + torch.clamp(modifier, min=-self.epsilon, max=self.epsilon), min=0, max=1)


            true_image = self._transform_video_ILAF(true_image, mode='forward') # norm

            step_feature_maps = []
            opt = self.model(true_image)
            
            for mm in range(len(self.activations['value'])):
                activations = self.activations['value'][mm]
                # activations = Variable(activations, requires_grad=False)
                step_feature_maps.append(activations)

            step_directions = [] # normalized direction
            step_norms = [] # norm values
            for ori_di, adv_di in zip(ori_feature_maps, step_feature_maps):
                step_direction = adv_di - ori_di
                step_norm = torch.norm(step_direction, p=2)
                step_norms.append(step_norm)
                step_directions.append(step_direction/torch.norm(step_direction,p=2,keepdim=True))

            losses = []
            for lens_fm in range(len(step_directions)):
                # magnitude
                magnitude_gain = step_norms[lens_fm] / init_norms[lens_fm]
                # angle
                angle_loss = torch.mm(init_directions[lens_fm].view(1,-1), step_directions[lens_fm].view(1,-1).transpose(1,0))
                this_loss = -(0.5 * magnitude_gain + angle_loss)
                losses.append(this_loss)
            cost = torch.sum(torch.stack(losses))

            grad = torch.autograd.grad(cost, modifier,
                                       retain_graph=False, create_graph=False)[0]
            modifier.data -= self.step_size * grad.sign()
            

            for ind,vid_name in enumerate(video_names):
                if vid_name not in self.loss_info.keys():
                    self.loss_info[vid_name] = {}  
                self.loss_info[vid_name][i] = {'cost': str(cost.detach().cpu().numpy())}

        true_image = torch.clamp(ori_unnorm_videos + torch.clamp(modifier.data, min=-self.epsilon, max=self.epsilon), min=0, max=1)
        image_inps = self._transform_video_ILAF(true_image, mode='forward')
        image_inps = image_inps.reshape(b,f,c,h,w)
        image_inps = image_inps.permute([0,2,1,3,4])
        return image_inps
