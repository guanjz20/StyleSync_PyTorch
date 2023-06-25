import cv2
import re
import math
import random
import itertools
import logging
import torch
from torch import nn
from torch.nn import functional as F

import utils
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

_logger = logging.getLogger('model')


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class Upsample(nn.Module):

    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad, device=self.device)

        return out


class Downsample(nn.Module):

    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad, device=self.device)

        return out


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1, device='cpu'):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, device=self.device)

        return out


class EqualConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
                f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})')


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation
        self.device = device
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul, device=self.device)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ScaledLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 style_dim,
                 demodulate=True,
                 upsample=False,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 device='cpu'):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor, device=device)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), device=device)

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
                f'upsample={self.upsample}, downsample={self.downsample})')

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):

    def __init__(self, isconcat=True):
        super().__init__()

        self.isconcat = isconcat
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, channel, height, width).normal_()

        if self.isconcat:
            return torch.cat((image, self.weight * noise), dim=1)
        else:
            return image + self.weight * noise


class ConstantInput(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 style_dim,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 isconcat=True,
                 device='cpu'):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel,
                                    out_channel,
                                    kernel_size,
                                    style_dim,
                                    upsample=upsample,
                                    blur_kernel=blur_kernel,
                                    demodulate=demodulate,
                                    device=device)

        self.noise = NoiseInjection(isconcat)
        feat_multiplier = 2 if isconcat else 1
        self.activate = FusedLeakyReLU(out_channel * feat_multiplier, device=device)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], device='cpu', out_channel=3):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel, device=device)

        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False, device=device)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConvLayer(nn.Sequential):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 activate=True,
                 device='cpu'):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), device=device))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(
            in_channel,
            out_channel,
            kernel_size,
            padding=self.padding,
            stride=stride,
            bias=bias and not activate,
        ))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel, device=device))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, device=device)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)

        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class audioConv2d(nn.Module):

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_groups = 32
        self.conv_block = nn.Sequential(nn.Conv2d(cin, cout, kernel_size, stride, padding),
                                        nn.GroupNorm(num_groups=num_groups, num_channels=cout))
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):

    def __init__(self, lr_mlp=0.01, device='cpu'):
        super().__init__()
        self.encoder = nn.Sequential(
            audioConv2d(1, 32, kernel_size=3, stride=1, padding=1),
            audioConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            audioConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(64, 128, kernel_size=3, stride=3, padding=1),
            audioConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            audioConv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            audioConv2d(256, 512, kernel_size=3, stride=1, padding=0),
            audioConv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
        self.linear = nn.Sequential(EqualLinear(512, 512, activation='fused_lrelu', device=device))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class Generator(nn.Module):

    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu',
    ):
        super().__init__()

        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', device=device))

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device)
        self.to_rgb1 = ToRGB(self.channels[4] * self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(in_channel * self.feat_multiplier,
                           out_channel,
                           3,
                           style_dim,
                           upsample=True,
                           blur_kernel=blur_kernel,
                           isconcat=isconcat,
                           device=device))

            self.convs.append(
                StyledConv(out_channel * self.feat_multiplier,
                           out_channel,
                           3,
                           style_dim,
                           blur_kernel=blur_kernel,
                           isconcat=isconcat,
                           device=device))

            self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        w_plus=False,
        delta_w=None,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            noise = [None] * (2 * (self.log_size - 2) + 1)

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t

        if len(styles) < 2:
            if not w_plus:
                inject_index = self.n_latent
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
                assert latent.shape[1] == self.n_latent
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for idx, (conv1, conv2, noise1, noise2,
                  to_rgb) in enumerate(zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs)):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        return image

    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True)
        return latent

    def get_latent(self, input):
        return self.style(input)


class FullGenerator(nn.Module):

    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu',
        mask_p='',
        mask_n_noise=None,
        face_z=False,
        tune_k=None,
        n_mlp_tune=2,
        noise_channel=False,
        noise_mask_p=None,
    ):
        super().__init__()

        self.size = size
        self.face_z = face_z
        self.mask_n_noise = mask_n_noise
        self.mask_p = mask_p
        self.noise_mask_p = noise_mask_p or self.mask_p
        self.act = nn.Sigmoid()

        self.audio_encoder = AudioEncoder(device=device)

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.log_size = int(math.log(size, 2))
        self.generator = Generator(
            size,
            style_dim,
            n_mlp,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            lr_mlp=lr_mlp,
            isconcat=isconcat,
            narrow=narrow,
            device=device,
        )

        conv = [ConvLayer(6, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d' % i for i in range(self.log_size - 1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2**(i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)]
            setattr(self, self.names[self.log_size - i + 1], nn.Sequential(*conv))
            in_channel = out_channel

        if self.mask_n_noise:
            size = self.size
            mask_mouth_region = cv2.imread(self.noise_mask_p)
            mask_mouth_region = cv2.resize(mask_mouth_region, (size, size))
            mask_back_region = 1. - mask_mouth_region[:, :, 0] / 255.
            mask_back_region_torch = torch.from_numpy(mask_back_region).float().view(1, 1, size, size)
            self.mask_back_region_list = [mask_back_region_torch]
            if self.mask_n_noise > 1:
                for _ in range(1, self.mask_n_noise):
                    size = size // 2
                    mask_back_region = cv2.resize(mask_back_region, (size, size))
                    mask_back_region_torch = torch.from_numpy(mask_back_region).float().view(1, 1, size, size)
                    self.mask_back_region_list.append(mask_back_region_torch)

        if self.face_z:
            self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu', device=device))
            self.cat_linear = nn.Sequential(EqualLinear(style_dim * 2, style_dim, activation='fused_lrelu', device=device))

        # if tune_k:
        #     for k in tune_k:
        #         _logger.info('Creating modules finetuned in [{}] ...'.format(k))
        #         self.finetune(k, freeze_other=False, n_mlp_tune=n_mlp_tune, noise_channel=noise_channel, device=device)

    def forward(
        self,
        face_sequences,
        audio_sequences,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        inversion_init=False,
        sm_audio_w=0.3,
        a_alpha=1.25,
    ):
        face = self.tensor5to4(face_sequences)
        audio = self.tensor5to4_audio(audio_sequences)
        inputs_masked = face[:, :3]
        inputs_ref = face[:, 3:]

        audio_feat = self.audio_encoder(audio)
        if sm_audio_w > 0:
            sm_audio_feat = getattr(self, 'sm_audio_feat', None)
            if sm_audio_feat is None:
                sm_audio_feat = audio_feat
            sm_audio_feat = sm_audio_w * sm_audio_feat + (1 - sm_audio_w) * audio_feat
            audio_feat = sm_audio_feat
            setattr(self, 'sm_audio_feat', sm_audio_feat)
        if a_alpha > 0:
            audio_feat *= a_alpha
        outs = audio_feat

        noise = []
        inputs = face
        for i in range(self.log_size - 1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
        face_feat_final = inputs

        if self.mask_n_noise:
            for j in range(self.mask_n_noise):
                noise_local = noise[j]
                mask_local = self.mask_back_region_list[j].type_as(noise_local).to(noise_local.device)
                noise[j] = noise_local * mask_local
        repeat_noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]

        if self.face_z:
            face_feat = self.final_linear(face_feat_final.view(face_feat_final.shape[0], -1))
            outs = self.cat_linear(torch.cat([outs, face_feat], dim=1))

        outs = self.generator(
            [outs],
            False,
            inject_index,
            truncation,
            truncation_latent,
            input_is_latent,
            noise=repeat_noise[1:],
        )
        image = self.act(outs)
        return image

    def tensor5to4(self, input):
        input_dim_size = len(input.size())
        if input_dim_size > 4:
            b, c, t, h, w = input.size()
            input = input.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return input

    def tensor5to4_audio(self, input):
        input_dim_size = len(input.size())
        if input_dim_size > 4:
            b, t, c, h, w = input.size()
            input = input.reshape(-1, c, h, w)
        return input


if __name__ == '__main__':
    model = FullGenerator(256, 512, 8)
    img_batch = torch.ones(2, 6, 5, 256, 256)
    mel_batch = torch.ones(2, 5, 1, 80, 16)
    x = model(img_batch, mel_batch)
    print(x.shape)
