import torch
import torch.nn.functional as F
import torchvision.models as models

from image_cam_utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer

# refer to https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/gradcam.py

def average_grad_cam_from_images(inps):
    '''
    inps: [b,c,f,h,w]
    '''
    b,c,f,h,w = inps.shape
    image_inps = inps.permute([0,2,1,3,4])
    image_inps = image_inps.reshape(b*f, c, h, w)
    model_lists = ['alexnet', 'vgg', 'resnet', 'densenet', 'squeezenet']
    masks = []
    for model_name in model_lists:
        if model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif model_name == 'vgg':
            model = models.vgg16(pretrained=True)
        elif model_name == 'resnet':
            model = models.resnet101(pretrained=True)
        elif model_name == 'densenet':
            model = models.densenet161(pretrained=True)
        elif model_name == 'squeezenet':
            model = models.squeezenet1_1(pretrained=True)
        model.eval()
        model.cuda()
        model_dict = dict(type=model_name, arch=model, input_size=(224, 224))
        
        gradcam = GradCAM(model_dict, False)
        mask, _ = gradcam(image_inps)
        masks.append(mask)
    average_mask = torch.stack(masks).mean(0, keepdim=False)
    return average_mask

class GradCAM(object):
    """Calculate GradCAM salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, 'features_29')
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, 'layer4')
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, 'features_norm5')
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, 'features_11')
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, 'features_12_expand3x3_activation')

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, ori_feature_mas, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward()
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        # saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        
        if self.update:
            print (saliency_map.shape)
            print (ori_feature_mas.shape)
            cost = torch.norm(saliency_map.reshape(b, -1) - ori_feature_mas.reshape(b, -1), p=2, dim=1)
            grad = torch.autograd.grad(cost, input, grad_outputs=torch.ones_like(cost),
                                       retain_graph=False, create_graph=False)[0]
            return grad
        else:
            return saliency_map

    def __call__(self, input, ori_feature_mas, update=False, class_idx=None, retain_graph=False):
        self.update = update
        return self.forward(input, ori_feature_mas, class_idx, retain_graph)