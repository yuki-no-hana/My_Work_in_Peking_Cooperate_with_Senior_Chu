import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .coordatt import CoordAtt, FAMv2
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class _BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, se=None,*, _conv3x3=None):
        super(_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _BasicBlock")
        if _conv3x3 is None:
            _conv3x3 = conv3x3
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = None if se is None else se(planes)

    def forward(self, x, **kwargs):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if self.se is not None:
            out = self.se(out,**kwargs)
        out += identity
        out = self.relu(out)

        return out

class FAM(nn.Module):
    def __init__(self, channels, h, w, *,rc=16, rs=32, L=32, bias=True, pool_types=('avg', 'max'), has_bn=True):
        super(FAM, self).__init__()
        assert isinstance(pool_types, tuple)
        mid_c = max(L, int(channels / rc))
        mid_s = max(L, int(h*w / rs))
        self.mlp_c = nn.Sequential(
            nn.Linear(channels, mid_c, bias=bias),
            nn.BatchNorm1d(mid_c) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, channels, bias=bias),
            nn.BatchNorm1d(channels) if has_bn else nn.Sequential(),
        )
        self.mlp_s = nn.Sequential(
            nn.Linear(h*w, mid_s, bias=bias),
            nn.BatchNorm1d(mid_s) if has_bn else nn.Sequential(),
            nn.ReLU(inplace=True),
            nn.Linear(mid_s, h*w, bias=bias),
            nn.BatchNorm1d(h*w) if has_bn else nn.Sequential(),
        )
        self.pool_types = pool_types
        self.f_c = nn.ReLU()
        self.f_s = nn.ReLU()

    def forward(self, x):
        origin_x = x
        n,c,h,w = x.shape
        x_c = self.f_c(x)
        xc_avg = F.adaptive_avg_pool2d(x_c,1).reshape(n,-1) if 'avg' in self.pool_types else None
        xc_max = F.adaptive_max_pool2d(x_c,1).reshape(n,-1) if 'max' in self.pool_types else None
        if len(self.pool_types) == 2:
            xc = xc_max + xc_avg
        elif 'avg' in self.pool_types:
            xc = xc_avg
        else:
            xc = xc_max
        ec = self.mlp_c(xc)

        x_s = self.f_s(x)
        xs_avg = torch.mean(x_s, dim=1).reshape(n,-1) if 'avg' in self.pool_types else None
        xs_max = torch.max(x_s, dim=1)[0].reshape(n,-1) if 'max' in self.pool_types else None
        if len(self.pool_types) == 2:
            xs = xs_max + xs_avg
        elif 'avg' in self.pool_types:
            xs = xs_avg
        else:
            xs = xs_max
        es = self.mlp_s(xs)
        
        attn_c = torch.sigmoid(ec.view(n,c,1,1))
        attn_s = torch.sigmoid(es.view(n,1,h,w))
        attn = attn_c*attn_s
        
        return origin_x*attn

class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers, ses=[None]*5, \
        norm_layer=nn.BatchNorm2d,proj_kernel_size=1):
        super(ResNet, self).__init__()
        if not isinstance(ses, list):
            ses = [ses]*4
        if not isinstance(block, list):
            block = [block]*4
        self.norm_layer = norm_layer 
        self.proj_kernel_size = proj_kernel_size

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = norm_layer(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block[0], self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.output_channel_block[0])
        self.se1 = nn.Sequential() if ses[0] is None else ses[0](self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block[1], self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(self.output_channel_block[1])
        self.se2 = nn.Sequential() if ses[1] is None else ses[1](self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block[2], self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = norm_layer(self.output_channel_block[2])
        self.se3 = nn.Sequential() if ses[2] is None else ses[2](self.output_channel_block[2])

        self.layer4 = self._make_layer(block[3], self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = norm_layer(self.output_channel_block[3])
        self.se4_1 = nn.Sequential() if ses[3] is None else ses[3](self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = norm_layer(self.output_channel_block[3])
        self.se4_2 = nn.Sequential() if ses[4] is None else ses[4](self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        expansion = block.func.expansion if isinstance(block,partial) else block.expansion
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=self.proj_kernel_size, stride=stride, padding=self.proj_kernel_size//2,bias=False),
                self.norm_layer(planes * expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.se1(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.se2(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.se3(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.se4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        x = self.se4_2(x)

        return x

class ResNetxx_FeatureExtractor(nn.Module):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512,*, ses=[None]*5, block=[_BasicBlock]*4 ,\
        norm_layer=nn.BatchNorm2d, proj_kernel_size=1):
        super(ResNetxx_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, block, [1, 2, 5, 3], ses=ses, \
            norm_layer=norm_layer,proj_kernel_size=proj_kernel_size)

    def forward(self, input):
        return self.ConvNet(input)
    
    def get_attn(self, x, **kwargs):
        self.ConvNet.get_attn(x,**kwargs)

def ResNetbn_famal_FeatureExtractor(input_channel, output_channel=512):
    spatial_size = [(16,50), (8,25), (4,26), (4,26),(2,27),(1,26)]
    re_1 = 16
    re_23 = 32
    pool_types = ('avg',)
    ses = [partial(FAM,h=h,w=w,rc=re_1,rs=re_23, pool_types=pool_types) for h,w in spatial_size]
    block = [partial(_BasicBlock,se=se) for se in ses[:4]]
    return ResNetxx_FeatureExtractor(input_channel,output_channel,ses=[None,None,None,None,ses[5]],block=block,\
        norm_layer=nn.BatchNorm2d)

def ResNetbn_famaml_FeatureExtractor(input_channel, output_channel=512):
    spatial_size = [(16,50), (8,25), (4,26), (4,26),(2,27),(1,26)]
    re_1 = 16
    re_23 = 32
    pool_types = ('avg','max')
    ses = [partial(FAM,h=h,w=w,rc=re_1,rs=re_23, pool_types=pool_types) for h,w in spatial_size]
    block = [partial(_BasicBlock,se=se) for se in ses[:4]]
    return ResNetxx_FeatureExtractor(input_channel,output_channel,ses=[None,None,None,None,ses[5]],block=block,\
        norm_layer=nn.BatchNorm2d)


def ResNetbn_coordatt_FeatureExtractor(input_channel, output_channel=512):
    ses = [partial(CoordAtt) for i in range(6)]
    block = [partial(_BasicBlock,se=se) for se in ses[:4]]
    return ResNetxx_FeatureExtractor(input_channel,output_channel,ses=[None,None,None,None,ses[5]],block=block,\
        norm_layer=nn.BatchNorm2d)

def ResNetbn_famw_FeatureExtractor(input_channel, output_channel=512):
    spatial_size = [(16,50), (8,25), (4,26), (4,26),(2,27),(1,26)]
    re_1 = 16
    re_23 = 32
    pool_types = ('avg',)
    ses = [partial(FAMv2,h=h,w=w,rc=re_1,rs=re_23, pool_types=pool_types) for h,w in spatial_size]
    block = [partial(_BasicBlock,se=se) for se in ses[:4]]
    return ResNetxx_FeatureExtractor(input_channel,output_channel,ses=[None,None,None,None,None],block=block,\
        norm_layer=nn.BatchNorm2d)
