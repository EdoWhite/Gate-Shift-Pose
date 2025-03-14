import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import gsf

# EDITED by white

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1  # For BasicBlock, expansion is 1 (output channels same as input channels)

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_segments=8, gsf_ch_ratio=4):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride)  # First convolution (3x3)
        self.bn1 = norm_layer(planes)  # First batch normalization
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)  # Second convolution (3x3)
        self.bn2 = norm_layer(planes)  # Second batch normalization

        self.downsample = downsample  # Downsampling layer (if needed for dimension matching)
        self.stride = stride

        # Adding GSF module here for segment fusion
        self.gsf = gsf.GSF(fPlane=planes * self.expansion, num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio)

    def forward(self, x):
        identity = x

        # Forward through first conv -> batch norm -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Forward through second conv -> batch norm
        out = self.conv2(out)
        out = self.gsf(out)  # Apply GSF module before final batch norm
        out = self.bn2(out)

        # Add identity (skip connection) if downsample is used
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual connection and apply ReLU
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, num_segments=8, gsf_ch_ratio=4, gsf_ch_fusion=False,
                 gsf_enabled=False, temp_kern=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.gsf = gsf.GSF(fPlane=planes * self.expansion, num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gsf(out)
           
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_segments=8, gsf_ch_ratio=25, num_channels=3):
        super(ResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_segments = num_segments

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # Modified to n nun_channels
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                                dilate=replace_stride_with_dilation[0], num_segments=num_segments,
                                                gsf_ch_ratio=gsf_ch_ratio)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                                dilate=replace_stride_with_dilation[1], num_segments=num_segments,
                                                gsf_ch_ratio=gsf_ch_ratio)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                                dilate=replace_stride_with_dilation[2], num_segments=num_segments,
                                                gsf_ch_ratio=gsf_ch_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for n, m in self.named_modules():
            # if 'gsf' not in n:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, num_segments=8, gsf_ch_ratio=25):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, num_segments=num_segments,
                            gsf_ch_ratio=gsf_ch_ratio))
        self.inplanes = planes * block.expansion
        for blk in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, num_segments=num_segments,
                                gsf_ch_ratio=gsf_ch_ratio))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, temperature: float=100.0):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnetOriginal(arch, block, layers, pretrained, progress, num_segments, gsf_ch_ratio=25, **kwargs):
    """
    Used to load standard ResNet checkpoints
    """
    model = ResNet(block, layers, num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
        model.load_state_dict(state_dict, strict=False)
    gsf_cnt = 0
    for k, v in model.state_dict().items():
        if 'conv3D.weight' in k:
            gsf_cnt += 1
    print('No. of GSF modules = {}'.format(gsf_cnt))
    return model

def _resnet(arch, block, layers, pretrained, progress, num_segments, gsf_ch_ratio=25, num_channels=3, **kwargs):
    """
    Used to load 3-channels ResNet checkpoints into an a model with additional channels
    """
    model = ResNet(block, layers, num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio, num_channels=num_channels, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        
        # Modifica il primo livello convoluzionale per accettare 4 canali invece di 3
        if state_dict['conv1.weight'].shape[1] == 3:  # Se il modello è stato pre-addestrato per 3 canali
            print("Model pre-trained with 3 channels")
            # Carica i pesi pre-addestrati per i primi 3 canali
            pretrained_weights = state_dict['conv1.weight']
            
            # Crea un nuovo tensor per i pesi di conv1 con 4 canali
            new_weights = torch.zeros((pretrained_weights.shape[0], 4, 7, 7))  # Crea pesi per 4 canali
            
            # Copia i pesi pre-addestrati nei primi 3 canali
            new_weights[:, :3, :, :] = pretrained_weights
            
            # Inizializza il quarto canale
            nn.init.kaiming_normal_(new_weights[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            
            # Sostituisci i pesi nel state_dict
            state_dict['conv1.weight'] = new_weights

        # Carica il resto dello stato del modello
        model.load_state_dict(state_dict, strict=False)

    # Conta i moduli GSF per debug
    gsf_cnt = 0
    for k, v in model.state_dict().items():
        if 'conv3D.weight' in k:
            gsf_cnt += 1
    print('No. of GSF modules = {}'.format(gsf_cnt))

    return model

def resnet18(pretrained=False, progress=True, num_segments=8, gsf_ch_ratio=25, num_channels=3, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if num_channels == 3:
        return _resnetOriginal('resnet18', BasicBlock, [3, 4, 6, 3], pretrained, progress, num_segments=num_segments,
                   gsf_ch_ratio=gsf_ch_ratio, **kwargs)
    else:
        return _resnet('resnet18', BasicBlock, [3, 4, 6, 3], pretrained, progress, num_segments=num_segments,
                   gsf_ch_ratio=gsf_ch_ratio, num_channels=num_channels, **kwargs)
    

def resnet50(pretrained=False, progress=True, num_segments=8, gsf_ch_ratio=25, num_channels=3, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if num_channels == 3:
        return _resnetOriginal('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_segments=num_segments,
                   gsf_ch_ratio=gsf_ch_ratio, **kwargs)
    else:
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_segments=num_segments,
                   gsf_ch_ratio=gsf_ch_ratio, num_channels=num_channels, **kwargs)


def resnet101(pretrained=False, progress=True, num_segments=8, gsf_ch_ratio=25, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio, **kwargs)

# ADDED
def wide_resnet101_2(pretrained=False, progress=True, num_segments=8, gsf_ch_ratio=25, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, 
                   num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio, **kwargs)


