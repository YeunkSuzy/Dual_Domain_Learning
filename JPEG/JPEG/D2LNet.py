import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        res = self.process(x)
        y = self.avg_pool(res)
        z = self.conv_du(y)
        return z *res + x
class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # ChannelAttention(n_feat, 4),
        ChannelAttention(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out

@ARCH_REGISTRY.register()
class D2LNet(nn.Module):
    def __init__(self, in_chn=3, wf=64, depth=4, relu_slope=0.2):
        super(D2LNet, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf),HinResBlock(wf,wf))
        self.conv_02 = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf),HinResBlock(wf,wf))
        self.pre_re = nn.Sequential(nn.Conv2d(in_chn, wf, 3, 1, 1),HinResBlock(wf,wf))
        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(FFTConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_FFT_AMP=False, use_FFT_PHASE=True))

            self.down_path_2.append(FFTConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_FFT_AMP=True, use_FFT_PHASE=False))
            prev_channels = (2**i) * wf
        # 8*wf-> wf
        # wf->2*wf
        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(HinResBlock((2**i)*wf,(2**i)*wf))
            self.skip_conv_2.append(HinResBlock((2**i)*wf,(2**i)*wf))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = Refine(prev_channels,in_chn)

        self.last2 = Refine(wf,in_chn)
        # self.last = conv3x3(prev_channels, in_chn, bias=True)
# in_chn是输入图像的通道数
    def forward(self, x):
        image = x #输入图像
        #stage 1 : amplitude JPEG restore stage
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1) #x1,x1_up是down之前和down之后
                encs.append(x1_up)
            else:
                x1 = down(x1)
        #以上对应图中的downpath
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        #以上对应图中的uppath
        sam_feature, out_1 = self.sam12(x1, image)


        #stage 2 : phase stage
        out_1_fft = torch.fft.rfft2(out_1, dim=(-2, -1))
        out_1_amp = torch.abs(out_1_fft)
        out_1_phase = torch.angle(out_1_fft)

        image_fft = torch.fft.rfft2(image, dim=(-2, -1))
        image_amp = torch.abs(image_fft)
        image_phase = torch.angle(image_fft)
        image_inverse = torch.fft.irfft2(image_amp*torch.exp(1j*out_1_phase), dim=(-2, -1))
        x2 = self.conv_02(image_inverse)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)
        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]))

        out_2 = self.last(x2) #调整通道为3
        tmp_2 = out_2 + image
        out_2 = tmp_2
        out_2 = self.pre_re(tmp_2)
        out_2 = self.last2(out_2)+tmp_2

        return [out_1, out_1_amp, out_1_phase, out_2]
    def get_input_chn(self, in_chn):
        return in_chn

class FFTConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_FFT_PHASE=False, use_FFT_AMP=False):
        super(FFTConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff
        self.use_FFT_PHASE = use_FFT_PHASE
        self.use_FFT_AMP = use_FFT_AMP
        self.sa = SpatialAttention()
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_fft_1 = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.relu_fft_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_fft_2 = nn.Conv2d(out_size, out_size, 1, 1, 0)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        if self.use_FFT_PHASE and self.use_FFT_AMP == False:
            x_res = self.identity(x)
            x_fft =torch.fft.fft2(x_res, dim=(-2, -1))
            x_amp = torch.abs(x_fft)
            x_phase = torch.angle(x_fft)

            x_phase = self.conv_fft_1(x_phase)
            x_phase = self.relu_fft_1(x_phase)
            x_phase = self.conv_fft_2(x_phase)
            x_fft_res = torch.fft.ifft2(x_amp*torch.exp(1j*x_phase), dim=(-2, -1))
            x_fft_res = x_fft_res.real
            dif = out-x_fft_res
            difsa = self.sa(dif)
            out = out*difsa
            out  = out + x_res + x_fft_res
        elif self.use_FFT_AMP and self.use_FFT_PHASE == False:
            x_res = self.identity(x)
            x_fft =torch.fft.fft2(x_res, dim=(-2, -1))
            x_amp = torch.abs(x_fft)
            x_phase = torch.angle(x_fft)

            x_amp = self.conv_fft_1(x_amp)
            x_amp = self.relu_fft_1(x_amp)
            x_amp = self.conv_fft_2(x_amp)
            x_fft_res = torch.fft.ifft2(x_amp*torch.exp(1j*x_phase), dim=(-2, -1))
            x_fft_res = x_fft_res.real
            dif = out-x_fft_res
            difsa = self.sa(dif)
            out = out*difsa
            out  = out + x_res + x_fft_res
        else:
            out = out + self.identity(x)

        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = FFTConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(FFTConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(FFTConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(FFTConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(FFTConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True,inc=3):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, inc, kernel_size, bias=bias)
        self.conv3 = conv(inc, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
