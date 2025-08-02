import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score


from .actionformer_proj import get_sinusoid_encoding
from ..bricks import ConvModule, AffineDropPath ,TransformerBlock,SGPBlock
from ..builder import PROJECTIONS

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#from iTransformer import iTransformer

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from flash_attn import flash_attn_qkvpacked_func

    FLASHATTN_AVAILABLE = True
except ImportError:
    FLASHATTN_AVAILABLE = False

try:
    from mamba_ssm.modules.mamba2_simple1 import Mamba2Simple as ViM
    #from mamba_ssm.modules.mamba_simple import Mamba as ViM
    from mamba_ssm.modules.mamba2_non import Mamba2Simple as DBM
    #from mamba_ssm.modules.mamba_new import Mamba as DBM

    MAMBA_AVAILABLE = True

except ImportError:
    MAMBA_AVAILABLE = False


import causal_conv1d_cuda

import warnings

warnings.filterwarnings("ignore", message="cumsum_cuda_kernel does not have a deterministic implementation")
warnings.filterwarnings("ignore", message="upsample_linear1d_backward_out_cuda does not have a deterministic implementation")


@PROJECTIONS.register_module()
class MambaProj(nn.Module):
    """Implementation of Video-Mamba-Suite: https://arxiv.org/abs/2403.09626"""

    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
        mamba_cfg=dict(kernel_size=4, drop_path_rate=0.3, use_mamba_type="dbm",use_mamba_type2="dbm"),  # default to DBM
    ):
        super().__init__()
        assert (
            MAMBA_AVAILABLE
        ), "Please install mamba-ssm to use this module. Check: https://github.com/OpenGVLab/video-mamba-suite"

        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len
        self.n_mha_win_size=19
        self.sgp_win_size = [1, 1, 1, 1, 1, 1]

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None
        
        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                    ConvModule(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type="relu"),
                    )
                )

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for i in range(arch[1]):
            self.stem.append(MaskMambaBlock(out_channels, **mamba_cfg))

        
        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(MaskMambaBlock(out_channels, n_ds_stride=2,**mamba_cfg))
        
            

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)
        #print('x.shape:',x.shape)
        #print(mask.shape)

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)
            #print('idx:',idx)
            #print('x.shape:',x.shape)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            #print(x.shape)
            #x = highpass_filter(x, cutoff=10.0, fs=30.0)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
        

class MaskMambaBlock(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=4,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        drop_path_rate=0.3,  # drop path rate
        use_mamba_type="dbm",
        use_mamba_type2="dbm",
        dilation=8
    ):
        super().__init__()
        if use_mamba_type == "dbm":
            self.mamba = DBM(n_embd, d_conv=kernel_size)
        elif use_mamba_type == "vim":
            # vim
            self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
        elif use_mamba_type == "dim":
            # dim
            self.mamba = DilatedMamba(n_embd, d_conv=kernel_size,expand=1,d_state=8,dilation=8)
            self.mamba2 = DilatedMamba(n_embd, d_conv=kernel_size,expand=1,d_state=8,dilation=16)
            #self.mamba3 = MambaVisionMixer(n_embd, d_conv=kernel_size,expand=1,d_state=8,dilation=1)
        else:
            raise NotImplementedError
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None


        self.norm = nn.LayerNorm(n_embd)
        
        #self.gamma = nn.Parameter(torch.ones(1))

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        
        res = x
        x_ = x.transpose(1, 2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)+self.mamba2(x_).transpose(1, 2)
        x = x_ * mask.unsqueeze(1).to(x.dtype)

        x = res + self.drop_path(x)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return x, mask
        '''
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.mamba(self.norm(x)))
        x = x.permute(0, 2, 1)
        x = x * mask.unsqueeze(1).to(x.dtype)

        if self.downsample is not None:
            mask = self.downsample(mask.float()).bool()
            x = self.downsample(x) * mask.unsqueeze(1).to(x.dtype)
        return x, mask
        '''



class MaskMambaBlock2(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=4,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        drop_path_rate=0.3,  # drop path rate
        use_mamba_type="dbm",
        use_mamba_type2="dim",
        dilation=1,
    ):
        super().__init__()
        if use_mamba_type2 == "dbm":
            self.mamba = DBM(n_embd, d_conv=kernel_size, use_fast_path=True, expand=1)
        elif use_mamba_type2 == "vim":
            # vim
            self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
        elif use_mamba_type2 == "dim":
            # mxm
            self.mamba = DilatedMamba(n_embd, d_conv=kernel_size,expand=1,d_state=8,dilation=dilation)
        else:
            raise NotImplementedError
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        self.norm = nn.LayerNorm(n_embd)

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1, 2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)
        x = x_ * mask.unsqueeze(1).to(x.dtype)

        x = res + self.drop_path(x)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return x, mask

class MaxPooler(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x, mask, **kwargs):
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = self.ds_pooling(mask.float()).bool()
        else:
            # masking out the features
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.unsqueeze(1).to(x.dtype)

        return out, out_mask.bool()



class DilatedMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
        dilation = 1,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.dilation = dilation

        self.in_proj = nn.Linear(self.d_model, self.d_inner*2, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner*2, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        


    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")


        xz_f, xz_b = torch.chunk(xz, 2, dim=1)  # (B, D, L)
        xz_b = xz_b.flip([-1])
        xz = torch.cat([xz_f, xz_b], dim=0)

        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        #z = F.silu(z)
        #print(self.d_conv)
        
        padding = (self.d_conv - 1) * self.dilation
        x_padded = F.pad(x, (padding, 0))
        z_padded = F.pad(z, (self.d_conv - 1, 0))
        '''
          #dtad-i
        x_t = x
        x_t_padded = F.pad(x_t, ((self.d_conv - 1), 0))
        x_t = F.silu(F.conv1d(input=x_t_padded, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, dilation=1, groups=self.d_inner//2))
        '''

        x = F.silu(F.conv1d(input=x_padded, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, dilation=self.dilation, groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z_padded, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, dilation=1, groups=self.d_inner//2))
        #x = x+x_t      #dtad-i

        #x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))  #Ablation


        
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        #dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        

        y = y.chunk(2)
        y = torch.cat([y[0], y[1].flip([-1])], dim=1)

        out = F.linear(rearrange(y, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        return out
