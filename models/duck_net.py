from torch import nn
import torch.nn.functional as F
import torch
from timm.models.layers import trunc_normal_
import math
from operator import __add__
from typing import List, Tuple, Optional





def conv2d_same(
    x,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dSame, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        return conv2d_same(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
    if isinstance(x, torch.Tensor):
        return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
    else:
        return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def pad_same_arg(
        input_size: List[int],
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1),
) -> List[int]:
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = get_same_padding(ih, kh, stride[0], dilation[0])
    pad_w = get_same_padding(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


def pad_same(
        x,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int] = (1, 1),
        value: float = 0,
):
    ih, iw = x.size()[-2:]
    pad_h = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return Conv2dSame(in_planes, out_planes, kernel_size=1, padding=0, dilation=1)


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    return Conv2dSame(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
    )


class Conv_Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, block_type, kernel=3, padding=1
    ):
        super(Conv_Block, self).__init__()
        self.wide = WideScope_Conv(in_channels, out_channels)
        self.mid = MidScope_Conv(in_channels, out_channels)
        self.res = ResNet_Conv(in_channels, out_channels)
        self.sep = Separated_Conv(in_channels, out_channels, kernel)
        self.duck = Duck_Block(in_channels, out_channels)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.block_type = block_type

    def forward(self, x):
        result = x
        
        if self.block_type == "separated":
            result = self.sep(result)
        elif self.block_type == "duckv2":
            result = self.duck(result)
        elif self.block_type == "midscope":
            result = self.mid(result)
        elif self.block_type == "widescope":
            result = self.wide(result)
        elif self.block_type == "resnet":
            result = self.res(result)
        elif self.block_type == "double_convolution":
            result = self.double_conv(result)
        else:
            return None

        return result

class Duck_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Duck_Block, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.wide = WideScope_Conv(in_channels, out_channels)
        self.mid = MidScope_Conv(in_channels, out_channels)

        self.res_1 = ResNet_Conv(in_channels, out_channels)

        self.res_2 = ResNet_Conv(in_channels, out_channels)
        self.res_2_1 = ResNet_Conv(out_channels, out_channels)

        self.res_3 = ResNet_Conv(in_channels, out_channels)
        self.res_3_1 = ResNet_Conv(out_channels, out_channels)
        self.res_3_2 = ResNet_Conv(out_channels, out_channels)

        self.sep = Separated_Conv(in_channels, out_channels, 6)
        self.norm_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.wide(x)
        x2 = self.mid(x)
        x3 = self.res_1(x)
        x4 = self.res_2_1(self.res_2(x))
        x5 = self.res_3_2(self.res_3_1(self.res_3(x)))
        x6 = self.sep(x)

        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.norm_out(x)
        return x


class Separated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(Separated_Conv, self).__init__()
        # self.conv1n = Conv2dSamePadding(
        #     in_channels, out_channels, kernel_size=(1, kernel)
        # )
        # self.convn1 = Conv2dSamePadding(
        #     out_channels,
        #     out_channels,
        #     kernel_size=(kernel, 1),
        # )
        # self.act = nn.ReLU(inplace=True)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.sep = nn.Sequential(
            Conv2dSame(
                in_channels, out_channels, kernel_size=(1, kernel)
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(
                out_channels,
                out_channels,
                kernel_size=(kernel, 1),
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # x = self.norm(self.act(self.conv1n(x)))
        # x = self.norm(self.act(self.convn1(x)))
        x = self.sep(x)
        return x


class MidScope_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MidScope_Conv, self).__init__()
        # self.conv33_1 = Conv2dSamePadding(
        #     in_channels,
        #     out_channels,
        #     kernel_size=3,
        #     dilation=1,
        # )
        # self.conv33_2 = Conv2dSamePadding(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     dilation=2,
        # )
        # self.act = nn.ReLU(inplace=True)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.mid = nn.Sequential(
            Conv2dSame(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(
                out_channels,
                out_channels,
                kernel_size=3,
                dilation=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # x = self.norm(self.act(self.conv33_1(x)))
        # x = self.norm(self.act(self.conv33_2(x)))
        x = self.mid(x)
        return x


class WideScope_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WideScope_Conv, self).__init__()
        # self.conv33_1 = Conv2dSamePadding(
        #     in_channels,
        #     out_channels,
        #     kernel_size=3,
        #     dilation=1,
        # )
        # self.conv33_2 = Conv2dSamePadding(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     dilation=2,
        # )
        # self.conv33_3 = Conv2dSamePadding(
        #     out_channels,
        #     out_channels,
        #     kernel_size=3,
        #     dilation=3,
        # )
        # self.act = nn.ReLU(inplace=True)
        # self.norm = nn.BatchNorm2d(out_channels)

        self.wide = nn.Sequential(
            Conv2dSame(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                dilation=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            Conv2dSame(
                out_channels,
                out_channels,
                kernel_size=3,
                dilation=3,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # x = self.norm(self.act(self.conv33_1(x)))
        # x = self.norm(self.act(self.conv33_2(x)))
        # x = self.norm(self.act(self.conv33_3(x)))
        x = self.wide(x)

        return x

    
class ResNet_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet_Conv, self).__init__()
        self.conv11 = conv1x1(in_channels, out_channels)
        self.conv33_1 = conv3x3(in_channels, out_channels)
        self.conv33_2 = conv3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = self.act(self.conv11(x))

        x = self.norm(self.act(self.conv33_1(x)))
        x = self.norm(self.act(self.conv33_2(x)))

        return self.norm(x + skip)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            Conv2dSame(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv2dSame(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
def UpsamplingNearest2d(x, scale_factor=2, mode='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)

class DUCK_Net(nn.Module):
    def __init__(self, in_channels = 17):
        super(DUCK_Net, self).__init__()
        self.name = "Duck_Net"
        self.conv1 = Conv2dSame(3, in_channels * 2, kernel_size=2, stride=2)
        self.conv2 = Conv2dSame(
            in_channels * 2, in_channels * 4, kernel_size=2, stride=2
        )
        self.conv3 = Conv2dSame(
            in_channels * 4, in_channels * 8, kernel_size=2, stride=2
        )
        self.conv4 = Conv2dSame(
            in_channels * 8, in_channels * 16, kernel_size=2, stride=2
        )
        self.conv5 = Conv2dSame(
            in_channels * 16, in_channels * 32, kernel_size=2, stride=2
        )

        self.t0 = Conv_Block(3, in_channels, "duckv2")

        self.l1i = Conv2dSame(
            in_channels, in_channels * 2, kernel_size=2, stride=2
        )
        self.l2i = Conv2dSame(
            in_channels * 2, in_channels * 4, kernel_size=2, stride=2
        )
        self.l3i = Conv2dSame(
            in_channels * 4, in_channels * 8, kernel_size=2, stride=2
        )
        self.l4i = Conv2dSame(
            in_channels * 8, in_channels * 16, kernel_size=2, stride=2
        )
        self.l5i = Conv2dSame(
            in_channels * 16, in_channels * 32, kernel_size=2, stride=2
        )

        self.t1 = Conv_Block(in_channels * 2, in_channels * 2, "duckv2")
        self.t2 = Conv_Block(in_channels * 4, in_channels * 4, "duckv2")
        self.t3 = Conv_Block(in_channels * 8, in_channels * 8, "duckv2")
        self.t4 = Conv_Block(in_channels * 16, in_channels * 16, "duckv2")
        self.t5_0 = Conv_Block(in_channels * 32, in_channels * 32, "resnet")
        self.t5_1 = Conv_Block(in_channels * 32, in_channels * 32, "resnet")
        self.t5_3 = Conv_Block(in_channels * 32, in_channels * 16, "resnet")
        self.t5_2 = Conv_Block(in_channels * 16, in_channels * 16, "resnet")

        self.q4 = Conv_Block(in_channels * 16, in_channels * 8, "duckv2")
        self.q3 = Conv_Block(in_channels * 8, in_channels * 4, "duckv2")
        self.q2 = Conv_Block(in_channels * 4, in_channels * 2, "duckv2")
        self.q1 = Conv_Block(in_channels * 2, in_channels, "duckv2")
        self.z1 = Conv_Block(in_channels, in_channels, "duckv2")

        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            fan_in // m.groups
            std = math.sqrt(2.0 / fan_in)
            m.weight.data.normal_(0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        t_0 = self.t0(x)
        l1_i = self.l1i(t_0)
        s_1 = p1 + l1_i
        t_1 = self.t1(s_1)

        l2_i = self.l2i(t_1)
        s_2 = p2 + l2_i
        t_2 = self.t2(s_2)

        l3_i = self.l3i(t_2)
        s_3 = p3 + l3_i
        t_3 = self.t3(s_3)

        l4_i = self.l4i(t_3)
        s_4 = p4 + l4_i
        t_4 = self.t4(s_4)

        l5_i = self.l5i(t_4)
        s_5 = p5 + l5_i
        t_51 = self.t5_1(s_5)
        t_51 = self.t5_0(s_5)
        t_53 = self.t5_3(t_51)
        t_52 = self.t5_2(t_53)

        l5_o = UpsamplingNearest2d(t_52)
        c4 = l5_o + t_4
        q_4 = self.q4(c4)

        l4_o = UpsamplingNearest2d(q_4)
        c3 = l4_o + t_3
        q_3 = self.q3(c3)

        l3_o = UpsamplingNearest2d(q_3)
        c2 = l3_o + t_2
        q_2 = self.q2(c2)

        l2_o = UpsamplingNearest2d(q_2)
        c1 = l2_o + t_1
        q_1 = self.q1(c1)

        l1_o = UpsamplingNearest2d(q_1)
        c0 = l1_o + t_0
        z_1 = self.z1(c0)

        x = self.out(z_1)
        x = torch.sigmoid(x)

        return x