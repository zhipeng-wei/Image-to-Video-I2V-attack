import torch
import torch.nn as nn
import scipy.stats as st
import numpy as np
import torchvision
from PIL import Image
import random

from utils import norm_grads
# refer to https://github.com/Harry24k/adversarial-attacks-pytorch

class Attack(object):
    """
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the model's training mode to `test`
        by `.eval()` only during an attack process.
    """
    def __init__(self, name, model):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device
        
        self._targeted = 1
        self._attack_mode = 'default'
        self._return_type = 'float'
        self._target_map_function = lambda images, labels:labels

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, *input):
        r"""
        It defines the computation performed at every call (attack forward).
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
    def set_attack_mode(self, mode, target_map_function=None):
        r"""
        Set the attack mode.
  
        Arguments:
            mode (str) : 'default' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
                         
            target_map_function (function) :
        """
        if self._attack_mode is 'only_default':
            raise ValueError("Changing attack mode is not supported in this attack method.")
            
        if (mode is 'targeted') and (target_map_function is None):
            raise ValueError("Please give a target_map_function, e.g., lambda images, labels:(labels+1)%10.")
            
        if mode=="default":
            self._attack_mode = "default"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode=="targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._target_map_function = target_map_function
            self._transform_label = self._get_target_label
        elif mode=="least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(mode + " is not a valid mode. [Options : default, targeted, least_likely]")
            
    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == 'float':
            self._return_type = 'float'
        elif type == 'int':
            self._return_type = 'int'
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == 'int':
                adv_images = adv_images.float()/255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print('- Save Progress : %2.2f %% / Accuracy : %2.2f %%' % ((step+1)/total_batch*100, acc), end='\r')

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print('\n- Save Complete!')

        self._switch_model()
    
    def _transform_perts(self, perts):
        dtype = perts.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=self.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=self.device)
        perts.div_(std[:, None, None, None])
        return perts

    def _transform_video(self, video, mode='forward'):
        r'''
        Transform the video into [0, 1]
        '''
        dtype = video.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=self.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=self.device)
        if mode == 'forward':
            # [-mean/std, mean/std]
            video.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        elif mode == 'back':
            # [0, 1]
            video.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
        return video

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels
        
    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels
    
    def _get_target_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return self._target_map_function(images, labels)
    
    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels
    
    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images*255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()
        
        del_keys = ['model', 'attack']
        
        for key in info.keys():
            if key[0] == "_" :
                del_keys.append(key)
                
        for key in del_keys:
            del info[key]
        
        info['attack_mode'] = self._attack_mode
        if info['attack_mode'] == 'only_default' :
            info['attack_mode'] = 'default'
            
        info['return_type'] = self._return_type
        
        return self.attack + "(" + ', '.join('{}={}'.format(key, val) for key, val in info.items()) + ")"

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == 'int':
            images = self._to_uint(images)

        return images

class FGSM(Attack):
    '''Fast Gradient Sign Method'''
    def __init__(self, model, steps=None, epsilon=16/255):
        super(FGSM, self).__init__("FGSM", model)
        self.epsilon = epsilon

    def forward(self, videos, labels):
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()

        videos.requires_grad = True
        outputs = self.model(videos)
        cost = self._targeted*loss(outputs, labels).to(self.device)

        grad = torch.autograd.grad(cost, videos,
                                   retain_graph=False, create_graph=False)[0]

        adv_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = adv_videos + self.epsilon*grad.sign()
        adv_videos = torch.clamp(adv_videos, min=0, max=1).detach()
        adv_videos = self._transform_video(adv_videos, mode='forward') # norm

        return adv_videos    

class BIM(Attack):
    '''
    Basic Iterative Method
    Only iterative version.
    '''
    def __init__(self, model, epsilon=16/255, steps=10):
        super(BIM, self).__init__("FGSM", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            adv_videos.requires_grad = True
            outputs = self.model(adv_videos)
            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, adv_videos, 
                                       retain_graph=False, create_graph=False)[0]
            
            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm

        return adv_videos

class MIFGSM(Attack):
    '''
    Momentum Iterative Fast Gradient Sign Method
    Only iterative version.
    '''
    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0):
        super(MIFGSM, self).__init__("MIFGSM", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(videos).to(self.device)
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            adv_videos.requires_grad = True
            outputs = self.model(adv_videos)

            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, adv_videos, 
                                       retain_graph=False, create_graph=False)[0]
            # frame-level or clip-level
            grad = norm_grads(grad, True)
            # grad_norm = torch.norm(grad, p=1)
            # grad /= grad_norm
            grad += momentum*self.decay
            momentum = grad
            
            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm

        return adv_videos

class DIFGSM(Attack):
    '''
    Diverse Inputs Method.
    Only iterative version.
    Contain momentum or no momentum.
    '''
    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0, momentum=False):
        super(DIFGSM, self).__init__("DIFGSM", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.momentum = momentum

    def _input_diversity(self, videos):
        # r = torch.randint(1,10, size=(1,1)).item()
        # if r <= 5:
        if random.random() < 0.5:
            return videos
        else:
            rnd = torch.randint(224,250, size=(1,1)).item()
            rescaled = videos.view((-1, ) + videos.shape[2:])
            rescaled = torch.nn.functional.interpolate(rescaled, size=[rnd, rnd], mode='nearest')
            # rescaled = torchvision.transforms.functional.resize(videos,[rnd, rnd], Image.NEAREST)
            h_rem = 250 - rnd
            w_rem = 250 - rnd
            pad_top = torch.randint(0, h_rem, size=(1,1)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_rem, size=(1,1)).item()
            pad_right = w_rem - pad_left
            padded = nn.functional.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom])
            # return torchvision.transforms.functional.resize(padded,[224, 224], Image.NEAREST)
            padded = torch.nn.functional.interpolate(padded, size=[224, 224], mode='nearest')
            padded = padded.view(videos.shape)
            return padded

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(videos).to(self.device)
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            adv_videos.requires_grad = True
            outputs = self.model(self._input_diversity(adv_videos))

            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, adv_videos, 
                                       retain_graph=False, create_graph=False)[0]

            if self.momentum:
                grad_norm = torch.norm(grad, p=1)
                grad /= grad_norm
                grad += momentum*self.decay
                momentum = grad
            else:
                pass

            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm

        return adv_videos

class TIFGSM(Attack):
    '''Translation-Invariant Attack'''
    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0, momentum=False):
        super(TIFGSM, self).__init__("MIFGSM", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.momentum = momentum
        # generate start_kernel
        kernel = self._initial_kernel(15, 3).astype(np.float32) # (15,15)
        stack_kernel = np.stack([kernel, kernel, kernel]) # (3,15,15)
        self.stack_kernel = torch.from_numpy(np.expand_dims(stack_kernel, 1)).to(self.device) # 3,1,15,15
    
    def _initial_kernel(self, kernlen, nsig):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def _conv2d_frame(self, grads):
        '''
        grads: N, C, T, H, W
        '''
        frames = grads.shape[2]
        out_grads = torch.zeros_like(grads)
        for i in range(frames):
            this_grads = grads[:,:,i]
            out_grad = nn.functional.conv2d(this_grads, self.stack_kernel, groups=3, stride=1, padding=7)
            out_grads[:,:,i] = out_grad
        out_grads = out_grads / torch.mean(torch.abs(out_grads), [1,2,3], True)
        return out_grads

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(videos).to(self.device)
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            adv_videos.requires_grad = True
            outputs = self.model(adv_videos)

            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, adv_videos, 
                                       retain_graph=False, create_graph=False)[0]

            grad = self._conv2d_frame(grad)
            if self.momentum:
                grad += momentum*self.decay
                momentum = grad
            else:
                pass

            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm

        return adv_videos

class SGM(Attack):
    '''Skip Gradient Method'''
    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0, gamma=0.5, momentum=False):
        super(SGM, self).__init__("SGM", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.momentum = momentum
        self.gamma = gamma
        
        # register model
        self._register_hook_for_model(self.model)

    def _register_hook_for_model(self, model):    
        def backward_hook(gamma):
            # implement SGM through grad through ReLU
            def _backward_hook(module, grad_in, grad_out):
                if isinstance(module, nn.ReLU):
                    return (gamma * grad_in[0],)
            return _backward_hook

        def backward_hook_norm(module, grad_in, grad_out):
            # normalize the gradient to avoid gradient explosion or vanish
            std = torch.std(grad_in[0])
            return (grad_in[0] / std,)

        backward_hook_sgm = backward_hook(np.power(self.gamma, 0.5))
        for name, module in model.named_modules():
            if 'relu' in name and not '0.relu' in name:
                module.register_backward_hook(backward_hook_sgm)

            # e.g., 1.layer1.1, 1.layer4.2, ...
            # if len(name.split('.')) == 3:
            # refer to https://github.com/csdongxian/skip-connections-matter/issues/3
            # if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
                # module.register_backward_hook(backward_hook_norm)

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(videos).to(self.device)
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            adv_videos.requires_grad = True
            outputs = self.model(adv_videos)

            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, adv_videos, 
                                       retain_graph=False, create_graph=False)[0]

            if self.momentum:
                grad_norm = torch.norm(grad, p=1)
                grad /= grad_norm
                grad += momentum*self.decay
                momentum = grad
            else:
                pass

            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm
        return adv_videos

class SIM(Attack):
    '''Scale-Invariant Attack Method'''
    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0, sclae_step=5, momentum=False):
        super(SIM, self).__init__("SIM", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.momentum = momentum
        self.sclae_step = sclae_step

    def _multi_scale(self, adv_videos, labels, loss):    
        def obtain_grad(vid, labels):
            vid.requires_grad = True
            outputs = self.model(vid)
            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, vid, 
                                       retain_graph=False, create_graph=False)[0]
            return grad
        
        mean_grad = None
        for i in range(self.sclae_step):
            tmp_videos = 1 / 2**i * adv_videos
            grad = obtain_grad(tmp_videos, labels)
            if mean_grad is None:
                mean_grad = grad
            else:
                mean_grad += grad
        return mean_grad / self.sclae_step

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(videos).to(self.device)
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            grad = self._multi_scale(adv_videos, labels, loss)

            if self.momentum:
                grad_norm = torch.norm(grad, p=1)
                grad /= grad_norm
                grad += momentum*self.decay
                momentum = grad
            else:
                pass

            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm
        return adv_videos

class TIFGSM3D(Attack):
    '''Translation-Invariant Attack'''
    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0, momentum=False):
        super(TIFGSM3D, self).__init__("TIFGSM3D", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.momentum = momentum
        # generate start_kernel
        kernel = self._initial_kernel(15, 3).astype(np.float32) # (15,15,15)
        stack_kernel = np.stack([kernel, kernel, kernel]) # (3,15,15,15)
        self.stack_kernel = torch.from_numpy(np.expand_dims(stack_kernel, 1)).to(self.device) # 3,1,15,15,15
    
    def _initial_kernel(self, kernlen, nsig):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        used_kernel = np.zeros((kernlen, kernlen, kernlen))
        for i in range(kern1d.shape[0]):
            used_kernel[i] = kern1d[i] * kernel_raw
        used_kernel = used_kernel / used_kernel.sum()
        return used_kernel

    def _conv3d_frame(self, grads):
        '''
        grads: N, C, T, H, W
        '''
        out_grads = nn.functional.conv3d(grads, self.stack_kernel, groups=3, stride=1, padding=7)
        # frames = grads.shape[2]
        # out_grads = torch.zeros_like(grads)

        # for i in range(frames):
        #     this_grads = grads[:,:,i]
        #     out_grad = nn.functional.conv2d(this_grads, self.stack_kernel, groups=3, stride=1, padding=7)
        #     out_grads[:,:,i] = out_grad
        out_grads = norm_grads(out_grads, True)
        return out_grads

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(videos).to(self.device)
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            adv_videos.requires_grad = True
            outputs = self.model(adv_videos)

            cost = self._targeted*loss(outputs, labels).to(self.device)
            grad = torch.autograd.grad(cost, adv_videos, 
                                       retain_graph=False, create_graph=False)[0]

            grad = self._conv3d_frame(grad)
            if self.momentum:
                grad += momentum*self.decay
                momentum = grad
            else:
                pass

            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm

        return adv_videos
    
class TAP(Attack):
    '''Transferable Adversarial Perturbations
    params = {
        'kernlen': 3,
        'temporal_kernlen':3,
        'eta': 1e3,
        'conv3d': True
    }
    '''
    def __init__(self, model, params, epsilon=16/255, steps=10):
        super(TAP, self).__init__("TAP", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps

        for name, value in params.items():
            setattr(self, name, value)

        kernel = self._initial_kernel_uniform(self.kernlen).astype(np.float32) # (3,3)
        stack_kernel = np.stack([kernel, kernel, kernel]) # (3,3,3)
        self.stack_2d_kernel = torch.from_numpy(np.expand_dims(stack_kernel, 1)).to(self.device) # 3,1,3,3

        kernel_3d = self._initial_kernel_uniform_3d(self.kernlen, self.temporal_kernlen) # [t,h,h]
        stack_kernel_3d = np.stack([kernel_3d, kernel_3d, kernel_3d]) # (3,t,h,h)
        self.stack_3d_kernel = torch.from_numpy(np.expand_dims(stack_kernel_3d, 1)).to(self.device) # 3,1,t,h,h

        self._activation_hook()
        
    def _initial_kernel_uniform(self, kernlen):
        kern1d = np.ones(kernlen)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def _initial_kernel_uniform_3d(self, kernlen, temporal_kernel):
        kern3d = np.ones((temporal_kernel, kernlen, kernlen))
        kern3d = kern3d / kern3d.sum()
        return kern3d

    def _conv2d_frames(self, perts):
        frames = perts.shape[2]
        out_perts = torch.zeros_like(perts)
        for i in range(frames):
            this_perts = perts[:,:,i]
            out_pert = nn.functional.conv2d(this_perts, self.stack_2d_kernel, groups=3, stride=1, padding=[int((self.kernlen-1)/2), int((self.kernlen-1)/2)])
            out_perts[:,:,i] = out_pert
        return torch.sum(torch.abs(out_perts))

    def _conv3d_frames(self, perts):
        out_perts = nn.functional.conv3d(perts, self.stack_3d_kernel, groups=3, stride=1, padding=[int((self.temporal_kernlen-1)/2), int((self.kernlen-1)/2), int((self.kernlen-1)/2)])
        return torch.sum(torch.abs(out_perts))

    def _find_target_layer(self):
        if 'i3d' in self.model_type:
            return [self.model.res_layers._modules['0'], self.model.res_layers._modules['1']]
        elif 'slowfast' in self.model_type:
            return [self.model._modules['slow_res2'], self.model._modules['slow_res3'], self.model._modules['fast_res2'], self.model._modules['fast_res3']] #[b,2048, 8, 7, 7], [b, 256, 32, 7, 7]
        elif 'tpn' in self.model_type:
            return [self.model.layer1, self.model.layer2]

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

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        batch_size = videos.shape[0]
        self.loss_info = {}
        self.stack_3d_kernel = self.stack_3d_kernel.type(videos.dtype)
        videos = videos.to(self.device)
        labels = labels.to(self.device)

        self.activations = dict()
        self.activations['value'] = []
        outputs = self.model(videos)
        ori_feature_map = self.activations['value']

        loss = nn.CrossEntropyLoss()
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()

        for i in range(self.steps):
            self.activations = dict()
            self.activations['value'] = []
            adv_videos.requires_grad = True
            outputs = self.model(adv_videos)

            # CE loss
            cost1 = self._targeted*loss(outputs, labels).to(self.device)

            # l2 distance
            # this_feature_map = self._feature_map(adv_videos, True, False, labels)
            feat_distance = []
            for i,j in zip(self.activations['value'], ori_feature_map):
                this_distance = torch.norm((torch.sign(i) * torch.sqrt(torch.abs(i))).reshape(batch_size, -1) - (torch.sign(j) * torch.sqrt(torch.abs(j))).reshape(batch_size, -1), p=2, dim=1)
                feat_distance.append(this_distance)
            cost2 = torch.sum(torch.stack(feat_distance), 0)

            # L2 norm
            perts = self._transform_perts(adv_videos - videos).to(self.device)
            if self.conv3d:
                reg_cost = self._conv3d_frames(perts)
            else:
                reg_cost = self._conv2d_frames(perts)

            cost = cost1 + 1e3 * reg_cost + 0.05 * cost2

            grad = torch.autograd.grad(cost, adv_videos, 
                                       retain_graph=False, create_graph=False)[0]

            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm
            self.loss_info[i] = {'ce loss': cost1.detach().cpu().numpy(), 
                        'reg_cost': reg_cost.detach().cpu().numpy(),
                        'distance': cost2.detach().cpu().numpy()}
        return adv_videos

