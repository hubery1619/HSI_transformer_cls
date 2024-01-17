import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.odconv import ODConv2d
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math
from models.attention_module import *



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension for attention
        output: (B, N, C)        
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class Attention(nn.Module):
    r""" Self attention module.
    Args:
        dim (int): Number of input channels.
        #window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, num_heads=16, attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension for attention
        output: (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=nn.GELU, act2_layer=nn.Identity, 
        bias=False, kernel_size=3, padding=1,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


### MBConv
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, use_se=False):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        # self.identity = stride == 1 and inp == oup
        self.identity = False
        self.conv1 = nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.act1 = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(oup)
        self.att_module = SELayer(inp, hidden_dim)
        # self.att_module = csSE(inp)





        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        # if self.identity:
        #     return x + self.conv(x)
        # else:
        #     return self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        # x = self.norm1(x)
        x = self.act1(x)
        x = self.att_module(x)
        x = self.conv2(x)
        # x = self.norm2(x)
        x = x.permute(0, 2, 3, 1)
        return x

        


        nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
        nn.BatchNorm2d(hidden_dim),
        SiLU(),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),




### MBConv


class PixelBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ChannelBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True):
        super().__init__()
        dim_transpose = patch_size*patch_size
        self.head_channel = dim
        self.per_channel = 16
        # num_heads = self.head_channel/self.per_channel
        self.norm1 = nn.LayerNorm(dim_transpose)
        self.attn = Attention(dim_transpose, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim_transpose)
        mlp_hidden_dim = int(dim_transpose * mlp_ratio)
        self.mlp = Mlp(in_features=dim_transpose, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        B_channel, N_channel, C_channel = x.shape
        x = x.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        B_merge = int(x.shape[0])
        x = x.view(B_channel, B_merge//B_channel, self.per_channel, C_channel)
        x = x.permute(0, 3, 1, 2).contiguous().view(B_channel, C_channel, -1)
        #x = x.transpose(1, 2)
        return x


# modify channel block as multi-head
class ChannelMultiHeadBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True):
        super().__init__()
        self.dim_transpose = patch_size*patch_size
        self.head_channel = dim
        self.per_channel = 16
        num_heads = math.ceil(self.dim_transpose / self.per_channel)
        self.channel_pad = self.per_channel * num_heads
        self.channel_rep_pad = nn.ReplicationPad1d(padding=(0,self.channel_pad-self.dim_transpose))
        # num_heads = self.head_channel/self.per_channel
        self.norm1 = nn.LayerNorm(self.channel_pad)
        self.attn = Attention(self.channel_pad, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(self.channel_pad)
        mlp_hidden_dim = int(self.channel_pad * mlp_ratio)
        self.mlp = Mlp(in_features=self.channel_pad, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.channel_rep_pad(x)
        B_channel, N_channel, C_channel = x.shape
        x = x.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        B_merge = int(x.shape[0])
        x = x.view(B_channel, B_merge//B_channel, self.per_channel, C_channel)
        x = x.permute(0, 3, 1, 2).contiguous().view(B_channel, C_channel, -1)
        x = x[:, :self.dim_transpose, :]
        #x = x.transpose(1, 2)
        return x

class ChannelMultiHeadBlockUpdate(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True):
        super().__init__()
        self.dim_transpose = patch_size*patch_size
        self.head_channel = dim
        # self.per_channel = 16
        self.per_channel = math.ceil(self.dim_transpose / num_heads)
        self.channel_pad = self.per_channel * num_heads
        self.channel_rep_pad = nn.ReplicationPad1d(padding=(0,self.channel_pad-self.dim_transpose))
        # num_heads = self.head_channel/self.per_channel
        self.norm1 = nn.LayerNorm(self.channel_pad)
        self.attn = Attention(self.channel_pad, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(self.channel_pad)
        mlp_hidden_dim = int(self.channel_pad * mlp_ratio)
        self.mlp = Mlp(in_features=self.channel_pad, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.channel_rep_pad(x)
        # B_channel, N_channel, C_channel = x.shape
        # x = x.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        # x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # B_merge = int(x.shape[0])
        # x = x.view(B_channel, B_merge//B_channel, self.per_channel, C_channel)
        # x = x.permute(0, 3, 1, 2).contiguous().view(B_channel, C_channel, -1)
        x = x.transpose(1, 2)
        x = x[:, :self.dim_transpose, :]
        #x = x.transpose(1, 2)
        return x


class ChannelMultiHeadBlock_Conv_Update(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True, group=1):
        super().__init__()
        self.init_values = 1e-4
        self.dim_transpose = patch_size*patch_size
        self.head_channel = dim
        # self.per_channel = 16
        self.per_channel = math.ceil(self.dim_transpose / num_heads)
        self.channel_pad = self.per_channel * num_heads
        self.channel_rep_pad = nn.ReplicationPad1d(padding=(0,self.channel_pad-self.dim_transpose))
        # num_heads = self.head_channel/self.per_channel
        self.norm1 = nn.LayerNorm(self.channel_pad)
        self.attn = Attention(self.channel_pad, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(self.channel_pad)
        mlp_hidden_dim = int(self.channel_pad * mlp_ratio)
        self.mlp = Mlp(in_features=self.channel_pad, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.patch_size = patch_size
        self.conv_branch = nn.Sequential(
                    nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                    nn.BatchNorm2d(mlp_hidden_dim),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group)
                    )
        self.gamma_2 = nn.Parameter(self.init_values * torch.ones((self.channel_pad)),requires_grad=True)


    def forward(self, x):
        # convolution branch
        x1 = x
        B = x1.shape[0]
        # x1 = x1.reshape(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
        convX = self.drop_path(self.conv_branch(x1.view(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(B, self.patch_size*self.patch_size, -1))

        # self-attention branch
        x = x.transpose(1, 2)
        x = self.channel_rep_pad(x)
        # B_channel, N_channel, C_channel = x.shape
        # x = x.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        # x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.gamma_2 * convX.transpose(1, 2)
        x = x + self.channel_rep_pad(convX.transpose(1, 2))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # B_merge = int(x.shape[0])
        # x = x.view(B_channel, B_merge//B_channel, self.per_channel, C_channel)
        # x = x.permute(0, 3, 1, 2).contiguous().view(B_channel, C_channel, -1)
        x = x.transpose(1, 2)
        x = x[:, :self.dim_transpose, :]
        #x = x.transpose(1, 2)
        return x


# add convolution branch in Pixel-level attention module 
class PixelConvBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True, group=1, depConv_flag=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.init_values = 1e-4
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.patch_size = patch_size
        self.depConv_flag = depConv_flag
        if self.depConv_flag:
            # self.conv_branch = SepConv(dim) 
            # self.conv_branch = MBConv(dim, dim) 
            self.conv_branch = MBConv(dim)

        else:   
            self.conv_branch = nn.Sequential(
                                nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                                nn.BatchNorm2d(mlp_hidden_dim),
                                nn.SiLU(inplace=True),
                                nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group)
                                )
        self.gamma_1 = nn.Parameter(self.init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        # convolution branch
        x1 = x
        B = x1.shape[0]

        if self.depConv_flag:
            convX = self.drop_path(self.conv_branch(x1.view(B, self.patch_size, self.patch_size, -1)).view(B, self.patch_size*self.patch_size, -1))
        else:
            # x1 = x1.reshape(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
            convX = self.drop_path(self.conv_branch(x1.view(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(B, self.patch_size*self.patch_size, -1))

        # self_attention branch
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # merge convolution branch and self_attention branch
        # x = x + self.gamma_1 * convX
        x = x + convX
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# add convolution branch in Pixel-level attention module 
class PixelConvBlockNoAttention(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True, group=1, depConv_flag=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.init_values = 1e-4
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.patch_size = patch_size
        self.depConv_flag = depConv_flag
        if self.depConv_flag:
            # self.conv_branch = SepConv(dim)
            self.conv_branch = MBConv(dim, dim) 
        else:   
            self.conv_branch = nn.Sequential(
                                nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                                nn.BatchNorm2d(mlp_hidden_dim),
                                nn.SiLU(inplace=True),
                                nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group)
                                )
        self.gamma_1 = nn.Parameter(self.init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        # convolution branch
        x1 = x
        B = x1.shape[0]

        if self.depConv_flag:
            convX = self.drop_path(self.conv_branch(x1.view(B, self.patch_size, self.patch_size, -1)).view(B, self.patch_size*self.patch_size, -1))
        else:
            # x1 = x1.reshape(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
            convX = self.drop_path(self.conv_branch(x1.view(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(B, self.patch_size*self.patch_size, -1))

        # self_attention branch
        # x = x + self.drop_path(self.attn(self.norm1(x)))

        # merge convolution branch and self_attention branch
        # x = x + self.gamma_1 * convX
        x = x + convX

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x






# add convolution branch in Channel-level attention module 
class ChannelConvBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True, group=1):
        super().__init__()
        dim_transpose = patch_size*patch_size
        self.head_channel = dim
        self.per_channel = 16
        # num_heads = self.head_channel/self.per_channel
        self.norm1 = nn.LayerNorm(dim_transpose)
        self.attn = Attention(dim_transpose, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim_transpose)
        mlp_hidden_dim = int(dim_transpose * mlp_ratio)
        self.mlp = Mlp(in_features=dim_transpose, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.patch_size = patch_size
        self.conv_branch = nn.Sequential(
                            nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                            nn.BatchNorm2d(mlp_hidden_dim),
                            nn.SiLU(inplace=True),
                            nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group)
                            )

    def forward(self, x):
        # convolution branch
        x1 = x
        B = x1.shape[0]
        # x1 = x1.reshape(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
        convX = self.drop_path(self.conv_branch(x1.view(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(B, self.patch_size*self.patch_size, -1))

        # self_attention branch        
        x = x.transpose(1, 2)
        B_channel, N_channel, C_channel = x.shape
        x = x.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)      
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # merge convolution branch and self_attention branch
        convX = convX.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        convX = convX.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)      
        x = x + convX   
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        B_merge = int(x.shape[0])
        x = x.view(B_channel, B_merge//B_channel, self.per_channel, C_channel)
        x = x.permute(0, 3, 1, 2).contiguous().view(B_channel, C_channel, -1)
        #x = x.transpose(1, 2)
        return x


# add convolution branch in Channel-level attention module 
class ChannelMultiHeadConvBlock(nn.Module):
    def __init__(self, dim, num_heads, patch_size=7, mlp_ratio=4, drop=0., attn_drop=0., drop_path=0., qkv_bias=True, group=1):
        super().__init__()
        self.dim_transpose = patch_size*patch_size
        self.head_channel = dim
        self.per_channel = 16
        num_heads = math.ceil(self.dim_transpose / self.per_channel)
        self.channel_pad = self.per_channel * num_heads
        self.channel_rep_pad = nn.ReplicationPad1d(padding=(0,self.channel_pad-self.dim_transpose))
        # num_heads = self.head_channel/self.per_channel
        self.norm1 = nn.LayerNorm(self.channel_pad)
        self.attn = Attention(self.channel_pad, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(self.channel_pad)
        mlp_hidden_dim = int(self.channel_pad * mlp_ratio)
        self.mlp = Mlp(in_features=self.channel_pad, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.patch_size = patch_size
        self.conv_branch = nn.Sequential(
                            nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                            nn.BatchNorm2d(mlp_hidden_dim),
                            nn.SiLU(inplace=True),
                            nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group)
                            )

    def forward(self, x):
        # convolution branch
        x1 = x
        B = x1.shape[0]
        # x1 = x1.reshape(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
        convX = self.drop_path(self.conv_branch(x1.view(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2).contiguous()).view(B, self.head_channel, -1))

        # self_attention branch        
        x = x.transpose(1, 2)
        x = self.channel_rep_pad(x)
        B_channel, N_channel, C_channel = x.shape
        x = x.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)      
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # merge convolution branch and self_attention branch
        convX = self.channel_rep_pad(convX)
        convX = convX.view(B_channel, self.per_channel, N_channel//self.per_channel, C_channel)
        convX = convX.permute(0, 2, 1, 3).contiguous().view(-1, self.per_channel, C_channel)      
        x = x + convX
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        B_merge = int(x.shape[0])
        x = x.view(B_channel, B_merge//B_channel, self.per_channel, C_channel)
        x = x.permute(0, 3, 1, 2).contiguous().view(B_channel, C_channel, -1)
        x = x[:, :self.dim_transpose, :]
        #x = x.transpose(1, 2)
        return x



class BasicLayer(nn.Module):
    """A basic Transformer layer for one stage
    we need to add the API of drop_path
    
    
    """
    def __init__(self, dim=256, attention_type=0, num_heads_channel=1, num_heads_spatial=1, depth=2, patch_size=7, mlp_ratio=1, drop=0., attn_drop=0., drop_path=0., qkv_bias=True):
        super().__init__()

        if attention_type == 0:
            self.blocks = nn.ModuleList([PixelConvBlockNoAttention(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) for j in range(depth)])

        elif attention_type == 1:
            self.blocks = nn.ModuleList([PixelBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) for j in range(depth)])

        elif attention_type == 2:
            self.blocks = nn.ModuleList([ChannelMultiHeadBlockUpdate(
            dim=dim, 
            num_heads=num_heads_spatial,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            drop=0., 
            attn_drop=0.,
            drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
            qkv_bias=qkv_bias)
            for j in range(depth)])

        elif attention_type == 3:
            self.blocks = nn.ModuleList([PixelBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 0 else ChannelMultiHeadBlockUpdate(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])

        elif attention_type == 4:
            self.blocks = nn.ModuleList([PixelBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 1 else ChannelMultiHeadBlockUpdate(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])

        elif attention_type == 5:
            self.blocks = nn.ModuleList([PixelConvBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 0 else ChannelMultiHeadBlock_Conv_Update(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])
    
        elif attention_type == 6:
            self.blocks = nn.ModuleList([PixelConvBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 1 else ChannelMultiHeadBlock_Conv_Update(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])


        elif attention_type == 7:
            self.blocks = nn.ModuleList([PixelConvBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])

        elif attention_type == 8:
            self.blocks = nn.ModuleList([ChannelMultiHeadBlock_Conv_Update(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])

        elif attention_type == 9:
            self.blocks = nn.ModuleList([PixelConvBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 0 else ChannelMultiHeadBlockUpdate(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])
    
        elif attention_type == 10:
            self.blocks = nn.ModuleList([PixelConvBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 1 else ChannelMultiHeadBlockUpdate(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])

        elif attention_type == 11:
            self.blocks = nn.ModuleList([PixelBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 0 else ChannelMultiHeadBlock_Conv_Update(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])

        elif attention_type == 12:
            self.blocks = nn.ModuleList([PixelBlock(
                dim=dim, 
                num_heads=num_heads_channel,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias) if j % 2 == 1 else ChannelMultiHeadBlock_Conv_Update(
                dim=dim, 
                num_heads=num_heads_spatial,
                mlp_ratio=mlp_ratio,
                patch_size=patch_size,
                drop=0., 
                attn_drop=0.,
                drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
                qkv_bias=qkv_bias)
                for j in range(depth)])

        else:
            print("The selection of the transformer block is wrong!!!")


        ####build blocks
        # self.blocks = nn.ModuleList([PixelConvBlockNoAttention(
        #     dim=dim, 
        #     num_heads=num_heads_channel,
        #     mlp_ratio=mlp_ratio,
        #     patch_size=patch_size,
        #     drop=0., 
        #     attn_drop=0.,
        #     drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
        #     qkv_bias=qkv_bias) for j in range(depth)])


        # self.blocks = nn.ModuleList([ChannelBlock(
        #     dim=dim, 
        #     num_heads=1,
        #     mlp_ratio=mlp_ratio,
        #     patch_size=patch_size,
        #     drop=0., 
        #     attn_drop=0.,
        #     drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
        #     qkv_bias=qkv_bias)
        #     for j in range(depth)])


        # self.blocks = nn.ModuleList([PixelConvBlock(
        #     dim=dim, 
        #     num_heads=num_heads_channel,
        #     mlp_ratio=mlp_ratio,
        #     patch_size=patch_size,
        #     drop=0., 
        #     attn_drop=0.,
        #     drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
        #     qkv_bias=qkv_bias) if j % 2 == 0 else ChannelMultiHeadBlockUpdate(
        #     dim=dim, 
        #     num_heads=num_heads_spatial,
        #     mlp_ratio=mlp_ratio,
        #     patch_size=patch_size,
        #     drop=0., 
        #     attn_drop=0.,
        #     drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
        #     qkv_bias=qkv_bias)
        #     for j in range(depth)])



        # self.blocks = nn.ModuleList([ChannelMultiHeadBlock_Conv_Update(
        #     dim=dim, 
        #     num_heads=num_heads_spatial,
        #     mlp_ratio=mlp_ratio,
        #     patch_size=patch_size,
        #     drop=0., 
        #     attn_drop=0.,
        #     drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
        #     qkv_bias=qkv_bias)
        #     for j in range(depth)])


        # self.blocks = nn.ModuleList([PixelConvBlock(
        #     dim=dim, 
        #     num_heads=num_heads,
        #     mlp_ratio=mlp_ratio,
        #     patch_size=patch_size,
        #     drop=0., 
        #     attn_drop=0.,
        #     drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
        #     qkv_bias=qkv_bias) if j % 2 == 0 else ChannelMultiHeadBlock(
        #     dim=dim, 
        #     num_heads=1,
        #     mlp_ratio=mlp_ratio,
        #     patch_size=patch_size,
        #     drop=0., 
        #     attn_drop=0.,
        #     drop_path=drop_path[j] if isinstance(drop_path, list) else drop_path,
        #     qkv_bias=qkv_bias)
        #     for j in range(depth)])


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        return x


    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)



class TokenEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1, patch_norm_flag=False):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        # self.proj = ODConv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.patch_norm_flag = patch_norm_flag
        if patch_norm_flag:
            self.patch_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        if self.patch_norm_flag:
            x = self.patch_norm(x)
        
        x = x.flatten(2).transpose(1, 2)
        
        after_feature_map_size = self.ifm_size  
        
        return x, after_feature_map_size




class HyperTransformer(nn.Module):
    def __init__(self, img_size=224, block_type=0, in_chans=3, num_classes=1000, num_stages=4, 
                n_groups=[32, 32, 32, 32], embed_dims=[256, 128, 64, 32], num_heads_channel=[8, 4, 2, 2], num_heads_spatial=[1, 1, 1, 1], mlp_ratios=[1, 1, 1, 1], depths=[2, 2, 2, 2], qkv_bias=True, ape=False, patch_norm=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()

        self.num_stages = num_stages
        self.num_layers = len(depths)
        self.ape = ape
        self.img_size = img_size
        self.block_type = block_type
        
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        num_patches = img_size*img_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.attention_layers = nn.ModuleList()
        self.embedding_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            attention_layer = BasicLayer(dim=embed_dims[i_layer], 
                                        attention_type = self.block_type,
                                        num_heads_channel=num_heads_channel[i_layer], 
                                        num_heads_spatial=num_heads_spatial[i_layer], 
                                        depth=depths[i_layer], 
                                        patch_size=self.img_size, 
                                        mlp_ratio=mlp_ratios[i_layer], 
                                        drop=0., 
                                        attn_drop=0., 
                                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                        qkv_bias=qkv_bias)
            self.attention_layers.append(attention_layer)

            embedding_layer = TokenEmbedding(in_feature_map_size=img_size,
                                            in_chans=new_bands if i_layer == 0 else embed_dims[i_layer - 1],
                                            embed_dim=embed_dims[i_layer],
                                            n_groups=n_groups[i_layer],
                                            patch_norm_flag=patch_norm)
            self.embedding_layers.append(embedding_layer)

            norm = nn.LayerNorm(embed_dims[i_layer])
            self.norm_layers.append(norm) 

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self.apply(self._init_weights)
        for bly in self.attention_layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_layers):
            patch_embed = self.embedding_layers[i]
            block = self.attention_layers[i]
            norm = self.norm_layers[i]
            
            x, s = patch_embed(x)  # s = feature map size after patch embedding
            
            if self.ape and i == 0:
                x = x + self.absolute_pos_embed

            x = block(x)
            
            x = norm(x)

            
            if i != self.num_layers - 1: 
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        x = self.avgpool(x.transpose(1, 2))  # B C 1   
        x = torch.flatten(x, 1)             
        
        return x

    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def proposed(dataset, patch_size, trans_type):
    model = None
    if dataset == 'sa':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=204, num_classes=16, n_groups=[1, 1, 1, 1], depths=[2, 2, 6, 2])
    elif dataset == 'hu':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=144, num_classes=15, n_groups=[1, 1, 1, 1], depths=[3, 2, 4, 2], embed_dims=[96, 64, 32, 32], num_heads_spatial=[2, 2, 2, 2])
    elif dataset == 'indian':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=200, num_classes=16, n_groups=[1, 1, 1, 1], depths=[2, 2, 6, 2])
    elif dataset == 'bot':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=145, num_classes=14, n_groups=[1, 1, 1, 1], depths=[2, 2, 6, 2], embed_dims=[128, 64, 32, 16], num_heads_spatial=[2, 2, 2, 2])
    elif dataset == 'pu':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=103, num_classes=9, n_groups=[1, 1, 1, 1], depths=[2, 2, 6, 2], embed_dims=[128, 64, 32, 16], num_heads_spatial=[8, 8, 8, 8])
    elif dataset == 'ksc':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=176, num_classes=13, n_groups=[1, 1, 1, 1], depths=[2, 2, 6, 2])
    elif dataset == 'whulk':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=270, num_classes=9, n_groups=[1, 1, 1, 1], depths=[2, 2, 6, 2])
    elif dataset == 'hrl':
        model = HyperTransformer(img_size=patch_size, block_type = trans_type, in_chans=176, num_classes=14, n_groups=[1, 1, 1, 1], depths=[2, 2, 6, 2])
    return model

if __name__ == "__main__":
    t = torch.randn(size=(3, 1, 204, 7, 7))
    print("input shape:", t.shape)
    net = proposed(dataset='sa', patch_size=7)
    print("output shape:", net(t).shape)

