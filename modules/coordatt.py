import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super(CoordAtt, self).__init__()
        inp = channels
        oup = channels
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class FAMv2(nn.Module):
    def __init__(self, channels, h, w, *,rc=16, rs=32, L=32, pool_types=('avg', 'max')):
        super(FAMv2, self).__init__()
        assert isinstance(pool_types, tuple)
        mid_c = max(L, int(channels / rc))
        mid_s = max(L, int(h*w / rs))
        self.mlp_c = nn.Sequential(
            nn.Linear(channels, mid_c, bias=False),
            nn.BatchNorm1d(mid_c),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, channels, bias=True),
        )
        self.mlp_s = nn.Sequential(
            nn.Linear(h*w, mid_s, bias=False),
            nn.BatchNorm1d(mid_s),
            nn.ReLU(inplace=True),
            nn.Linear(mid_s, h*w, bias=True),
        )
        self.pool_types = pool_types
        # self.f_c = nn.ReLU()
        # self.f_s = nn.ReLU()

    def forward(self, x):
        origin_x = x
        n,c,h,w = x.shape
        x_c = x#self.f_c(x)
        xc_avg = F.adaptive_avg_pool2d(x_c,1).reshape(n,-1) if 'avg' in self.pool_types else None
        xc_max = F.adaptive_max_pool2d(x_c,1).reshape(n,-1) if 'max' in self.pool_types else None
        if len(self.pool_types) == 2:
            xc = xc_max + xc_avg
        elif 'avg' in self.pool_types:
            xc = xc_avg
        else:
            xc = xc_max
        ec = self.mlp_c(xc)

        x_s = x#self.f_s(x)
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