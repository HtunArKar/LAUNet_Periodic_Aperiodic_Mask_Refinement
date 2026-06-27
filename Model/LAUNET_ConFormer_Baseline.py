# import necessary packages
from conformer import ConformerBlock
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy


def power_compress(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


def pad_to_multiple(x, mode="dim_2", multiple=16):
    # initialize
    pad_f = 0
    pad_t = 0

    # get dim
    _, _, T, Freq = x.shape

    if mode == "dim_2":

        pad_t = (multiple - T % multiple) % multiple

    elif mode == "dim_3":

        pad_f = (multiple - Freq % multiple) % multiple

    else:
        pad_f = (multiple - Freq % multiple) % multiple
        pad_t = (multiple - T % multiple) % multiple

    x = F.pad(x, (0, pad_f, 0, pad_t), mode='constant', value=0.0)  # pad F only
    return x


class HarmonicAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_final=False):
        super(HarmonicAttention, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding)

        self.scale_factor = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.channel_scale = nn.Parameter(torch.ones(out_channels))
        self.channel_alpha = nn.Parameter(torch.ones(out_channels))
        

        if not is_final:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
            self.activation = nn.PReLU(out_channels)

        self.is_final = is_final


    def forward(self, x, harmonic_mask):

        x = self.conv(x)

        if not self.is_final:
            x = self.norm(x)

        harmonic_mask = torch.sigmoid(harmonic_mask)
        harmonic_mask = harmonic_mask.expand(-1, x.shape[1], -1, -1)

        x_HA = x * harmonic_mask * self.channel_scale.view(1, -1, 1, 1) * self.scale_factor
        x = x + self.channel_alpha.view(1, -1, 1, 1) * x_HA

        if not self.is_final:
            x = self.activation(x)

        return x


class DownSampling(nn.Module):

    def __init__(self, in_channels, out_channels, mode="dim_2"):
        super(DownSampling, self).__init__()

        """
        Args:
                in_channels: input channel dim
                out_channels: output channel dim
                mode: dim_2, dim_3 and both, downsample along give dim
        """

        if mode == "dim_2":
            kernel_size = (3, 1)
            stride = (2, 1)
            padding = (1, 0)

        elif mode == "dim_3":
            kernel_size = (1, 3)
            stride = (1, 2)
            padding = (0, 1)

        else:
            kernel_size = (3, 3)
            stride = (2, 2)
            padding = (1, 1)

        # convolution block
        self.down_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        """
        Args:
                x: input 4D tensor, shape = [B?, C, _, _]
        Return:
                x: output 4D tensor with reduced associated dim
        """

        x = self.down_block(x)

        return x


class UpSampling(nn.Module):

    def __init__(self, in_channels, out_channels, mode="dim_2", r=2):
        super(UpSampling, self).__init__()
        """
        Args:
                in_channels: input channel dim
                out_channels: output channel dim
                mode: dim_2, dim_3 and both, upsample along give dim
                r: up-sampling factor
        """

        self.mode = mode
        self.r = r

        if mode == "dim_2":
            padding = (0, 0, 1, 1)
            kernel_size = (3, 1)
            out_channels = out_channels * r

        elif mode == "dim_3":
            padding = (1, 1, 0, 0)
            kernel_size = (1, 3)
            out_channels = out_channels * r

        else:
            padding = (1, 1, 1, 1)
            kernel_size = (3, 3)
            out_channels = out_channels * r * r

        self.pad = nn.ConstantPad2d(padding, value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=(1, 1)
        )

    def forward(self, x):
        """
        Args:
                x: input 4D tensor, shape = [B?, C, T, F]
        Return:
                x: output 4D tensor with reduced associated dim
        """
        x = self.pad(x)
        out = self.conv(x)
        B, C, T, Freq = out.shape

        if self.mode == "dim_2":
            out = out.view((B, self.r, C // self.r, T, Freq))
            out = out.permute(0, 2, 3, 1, 4)
            out = out.contiguous().view((B, C // self.r, -1, Freq))

        elif self.mode == "dim_3":
            out = out.view((B, self.r, C // self.r, T, Freq))
            out = out.permute(0, 2, 3, 4, 1)
            out = out.contiguous().view((B, C // self.r, T, -1))

        else:
            out = out.view(B, self.r, self.r, C // (self.r * self.r), T, Freq)
            out = out.permute(0, 3, 4, 1, 5, 2)  # (B, C, H, r, W, r)
            out = out.contiguous().view(B, C // (self.r * self.r), T * self.r, Freq * self.r)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attn_layers, mode, down=True):
        super(EncoderBlock, self).__init__()

        self.harmonic_attention = nn.ModuleList([])

        for i in range(attn_layers):

            if i == 0:
                block = HarmonicAttention(in_channels=in_channels, out_channels=out_channels)

            else:
                block = HarmonicAttention(in_channels=out_channels, out_channels=out_channels)

            self.harmonic_attention.append(block)


        if down:

            self.down_sampling = DownSampling(in_channels=out_channels, out_channels=out_channels, mode=mode)
            self.harmonic_downSampling = DownSampling(in_channels=1, out_channels=1, mode=mode)

        self.attn_layers = attn_layers
        self.down = down

    def forward(self, x, harmonic_mask):

        for i in range(self.attn_layers):

            x = self.harmonic_attention[i](x, harmonic_mask)

        if self.down:

            x_down = self.down_sampling(x)
            harmonic_mask_down = self.harmonic_downSampling(harmonic_mask)

        else:
            x_down = None
            harmonic_mask_down = None

        return x, x_down, harmonic_mask_down


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attn_layers, mode, up=True, is_final=False):
        super(DecoderBlock, self).__init__()

        self.harmonic_attention = nn.ModuleList([])

        for i in range(attn_layers):

            if i == 0:
                block = HarmonicAttention(in_channels=in_channels, out_channels=out_channels,
                                          is_final=is_final)

            else:
                block = HarmonicAttention(in_channels=out_channels, out_channels=out_channels,
                                          is_final=is_final)

            self.harmonic_attention.append(block)

        if up:

            self.up_sampling = UpSampling(in_channels=out_channels, out_channels=out_channels, mode=mode)

        self.attn_layers = attn_layers
        self.up = up

    def forward(self, x, harmonic_mask):

        for i in range(self.attn_layers):

            x = self.harmonic_attention[i](x, harmonic_mask)

        if self.up:

            x = self.up_sampling(x)

        return x


class LAUNET(nn.Module):

    def __init__(self, num_channels, attn_layers=2, conf_num_layers=4, mode="both", conv_kernel_size=17,
                 heads=8, ff_mult=2, expansion_factor=2, attn_dropout=0.0, ff_dropout=0.0, conv_dropout=0.0):
        super(LAUNET, self).__init__()

        """
        Args:
                num_channels: channel dims details used in the network
                attn_layers:
                mode: dim_2, dim_3 or both, to do downsampling and upsampling along provided dim mode
        """
        self.num_layers = len(num_channels) - 2
        self.conf_num_layers = conf_num_layers

        # initialize encoder module
        self.encoder_blocks = nn.ModuleList([])

        for i in range(len(num_channels) - 1):

            if i < (self.num_layers):
                down = True
            else:
                down = False

            block = EncoderBlock(in_channels=num_channels[i], out_channels=num_channels[i + 1], attn_layers=attn_layers,
                                 mode=mode, down=down)

            self.encoder_blocks.append(block)

        # Conformer module
        self.con_blocks = nn.ModuleList([])

        for i in range(conf_num_layers):

            self.con_blocks.append(
                copy.deepcopy(
                    ConformerBlock(
                        dim = num_channels[-1],
                        dim_head = num_channels[-1],
                        heads = heads,
                        ff_mult = ff_mult,
                        conv_expansion_factor = expansion_factor,
                        conv_kernel_size = conv_kernel_size,
                        attn_dropout = attn_dropout,
                        ff_dropout = ff_dropout,
                        conv_dropout = conv_dropout
                    )
                )
            )

        # initialize decoder module
        self.decoder_blocks = nn.ModuleList([])

        # reverse order for decoder
        num_channels = num_channels[::-1]

        for i in range(len(num_channels) - 1):

            if i < (self.num_layers):
                up = True
                is_final = False
            else:
                up = False
                is_final = True

            block = DecoderBlock(in_channels=num_channels[i], out_channels=num_channels[i + 1], attn_layers=attn_layers,
                                 mode=mode, up=up, is_final=is_final)

            self.decoder_blocks.append(block)


    def forward(self, x, h_mask):
        """
        Args:
                x: input tensor of shape [B?, C, T, F],
                h_mask: harmonic mask of shape [B?, C, T, F]
        """
        # initialize list to collect data
        feature_skips = []
        harmonic_skips = [h_mask]

        for i in range(len(self.encoder_blocks)):

            x_enc, x, h_mask = self.encoder_blocks[i](x, h_mask)

            feature_skips.append(x_enc)

            if i < (len(self.encoder_blocks) - 1):

                harmonic_skips.append(h_mask)

        # reversed order for decoder module
        feature_skips = feature_skips[::-1]
        harmonic_skips = harmonic_skips[::-1]

        # print(f"[INFO] Number of feature skips: {len(feature_skips)}")
        # print(f"[INFO] Number of harmonic skips: {len(harmonic_skips)}")

        # get the output of last Harmonic Attention Layer (without downsampling)
        enc_out = x_enc

        # reshape into [B? * F, T, C]
        b, c, t, f = enc_out.size()
        x_t = enc_out.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)

        # pass into conformer block
        for i in range(self.conf_num_layers):

            x_t = self.con_blocks[i](x_t) + x_t

        # reshape back to original shape
        x = x_t.view(b, f, t, c).permute(0, 3, 2, 1) # [B?, C, T, F]

        # print(f"[INFO] Shape of x: {x.shape}")
        # print(f"[INFO] Shape of skip: {feature_skips[0].shape}")

        # skip connection for conformer
        x = x + feature_skips[0]

        assert x != None, "Input for Decoder should not be None"

        for i in range(len(self.decoder_blocks)):

            x = self.decoder_blocks[i](x, harmonic_skips[i])

            if i < len(self.decoder_blocks) - 1:

                # print(f"[INFO] Shape of x: {x.shape}")
                # print(f"[INFO] Shape of skip: {feature_skips[i + 1].shape}")

                # add features
                x += feature_skips[i + 1]

        return x