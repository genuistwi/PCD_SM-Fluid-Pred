import sys
import math
import torch

from inspect import isfunction
from functools import partial
from einops import rearrange
import numpy as np


from torch import nn, einsum
import lightning as lt


def exists(x):
    return x is not None


def default(val, d):
    """ Designed to return a given value val if it exists; otherwise, it returns the default value d.
    It also checks whether the default value d is a callable (i.e., a function), and if so, it calls its result. """

    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(lt.LightningModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(lt.LightningModule):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = (time * embeddings[None, :])  # time is already in the shape [batchsize,1]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GaussianFourierProjection(lt.LightningModule):  # for continuous training
    """ Gaussian Fourier embeddings for noise levels. Taken and adapted from:
    https://github.com/yang-song/score_sde_pytorch/blob/
    cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layerspp.py#L32. """

    def __init__(self, dim=256, scale=16.0):
        super(GaussianFourierProjection, self).__init__()
        dim = dim//2
        self.W = nn.Parameter(torch.randn(dim) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Block(lt.LightningModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(lt.LightningModule):
    """ Adapted from https://arxiv.org/abs/1512.03385. """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):

        super(ResnetBlock, self).__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if exists(time_emb_dim) else None


        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(lt.LightningModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(lt.LightningModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(lt.LightningModule):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Network(lt.LightningModule):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8),
                 channels=1,  # First dimension of image
                 cond_channels=1,  # Same for conditioning
                 embedding='sinusoidal',
                 with_time_emb=True,
                 resnet_block_groups=2,
                 img_cond=None,
                 ):

        super(Network, self).__init__()

        self.embedding = embedding
        assert self.embedding in ['fourier', 'sinusoidal', ]

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embedding
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                GaussianFourierProjection(dim) if self.embedding == 'fourier' else SinusoidalPositionEmbeddings(dim),
                # SinusoidalPositionEmbeddings(dim), include option which embedding type to use
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            raise Exception("Time embedding is set to False, wrong for time conditional score matching")

        # determine dimensions
        self.channels = channels
        self.cond_channels = cond_channels
        self.is_cond = img_cond

        # Conditioning segmentation diffusion, see https://arxiv.org/abs/2303.05966
        if img_cond == 1:
            self.encoder_img = nn.Sequential(
                block_klass(channels, init_dim, time_emb_dim=time_dim)
            )
            self.encoder_mask = nn.Sequential(
                block_klass(cond_channels, init_dim, time_emb_dim=time_dim)
            )
            init_dim *= 2
            channels = init_dim

        elif img_cond == 0:
            self.encoder_img = nn.Sequential(
                block_klass(channels, init_dim, time_emb_dim=time_dim)
            )
            init_dim = init_dim
            channels = init_dim

        init_dim = default(init_dim, dim // 3 * 2)  # if no cond, init dim should around 10 (latent space dim proj)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Layers of the Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim, ),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim, ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim, ),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim, ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, self.channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim, time_emb_dim=None), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, img_cond=None):

        # Concatenation encoding conditioning
        if img_cond is not None and self.is_cond == 1:
            # x = self.encoder_img(x) + self.encoder_mask(cond)
            x = torch.cat((self.encoder_img(x), self.encoder_mask(img_cond)), 1)
        elif img_cond is None and self.is_cond == 0:
            x = self.encoder_img(x)
        else:
            print('Unconditional mode is set to ON but conditional input is fed (or vis-versa), exiting now...')
            sys.exit()

        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # Downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # Final conv to have the proper shape of data in output
        return self.final_conv(x)
