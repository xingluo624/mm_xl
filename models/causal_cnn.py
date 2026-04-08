import torch
import torch.nn as nn
from models.resnet import CausalResnet1D
from models.modules import Patcher1D, UnPatcher1D


    
    
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1-stride)
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=0,  
            dilation=dilation
        )

    def forward(self, x):
        # x的尺寸为 (batch_size, channels, sequence_length)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)
    



class CausalAttention1D(nn.Module):
    def __init__(self, in_channels: int, norm=None) -> None:
        super().__init__()

        self.norm = norm
        # if self.norm == "LN":
        #     x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
            
        if norm == "LN":
            self.norm1 = nn.LayerNorm(in_channels)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=in_channels, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            
        self.q = CausalConv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv1d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv1d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        if self.norm == "LN":
            h_ = self.norm1(h_.transpose(-2, -1)).transpose(-2, -1)
        else:
            h_ = self.norm1(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        # q, batch_size, height = space2batch(q)
        # k, _, _ = space2batch(k)
        # v, _, _ = space2batch(v)

        # bhw, c, t = q.shape
        
        b, c, t = q.shape
        
        q = q.permute(0, 2, 1)  # (b, t, c)
        k = k.permute(0, 2, 1)  # (b, t, c)
        v = v.permute(0, 2, 1)  # (b, t, c)

        w_ = torch.bmm(q, k.permute(0, 2, 1))  # (b, t, t)
        w_ = w_ * (int(c) ** (-0.5))

        # Apply causal mask
        mask = torch.tril(torch.ones_like(w_))
        w_ = w_.masked_fill(mask == 0, float("-inf"))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        h_ = torch.bmm(w_, v)  # (b, t, c)
        h_ = h_.permute(0, 2, 1).reshape(b, c, t)  # (b, c, t)

        # h_ = batch2space(h_, batch_size, height)
        h_ = self.proj_out(h_)
        return x + h_


class CausalEncoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 kernel_size=3,
                 use_patcher=False,
                 patch_size=1,
                 patch_method="haar",
                 use_attn=False):
        super().__init__()
        
        self.use_attn = use_attn
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if use_patcher:
            blocks.append(Patcher1D(patch_size, patch_method))
        # blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(CausalConv1d(input_emb_width, width, kernel_size, 1, (kernel_size-1)//2))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, pad_t),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            ) if not use_attn else nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, pad_t),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
                CausalAttention1D(width, norm=norm)
            )
            blocks.append(block)
        
        
        if use_attn:
            # blocks.append(CausalConv1d(width, output_emb_width, 3, 1, 1))
            # middle
            middle_blocks = []
            end_blocks = []
            
            middle_blocks.append(CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm))
            middle_blocks.append(CausalAttention1D(width, norm=norm))
            middle_blocks.append(CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm))
            
            # end
            if norm == "LN":
                end_norm = nn.LayerNorm(width)
            elif norm == "GN":
                end_norm = nn.GroupNorm(num_groups=32, num_channels=width, eps=1e-6, affine=True)
            elif norm == "BN":
                end_norm = nn.BatchNorm1d(num_features=width, eps=1e-6, affine=True)
            else:
                end_norm = nn.Identity()
                
            end_blocks.append(nn.ReLU())
            end_blocks.append(CausalConv1d(width, output_emb_width, kernel_size, 1, (kernel_size-1)//2))
            
            self.middle_blocks = nn.Sequential(*middle_blocks)
            self.end_norm = end_norm
            self.end_blocks = nn.Sequential(*end_blocks)
            
        else:
            blocks.append(CausalConv1d(width, output_emb_width, kernel_size, 1, (kernel_size-1)//2))
        
        self.model = nn.Sequential(*blocks)
        

    def forward(self, x):
        # x.shape (b, 263, len)
        output = self.model(x)
        if self.use_attn:
            output = self.middle_blocks(output)
            output = self.end_norm(output.transpose(-2, -1)).transpose(-2, -1)
            output = self.end_blocks(output)
        return output

class CausalDecoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None,
                 kernel_size=3,
                 use_patcher=False,
                 patch_size=1,
                 patch_method="haar",
                 use_attn=False):
        super().__init__()
        blocks = []
        
        self.use_attn = use_attn
        filter_t, pad_t = stride_t * 2, stride_t // 2
        # blocks.append(CausalConv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(CausalConv1d(output_emb_width, width, kernel_size, 1, (kernel_size-1)//2))
        
        # middle
        if use_attn:
            blocks.append(CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm))
            blocks.append(CausalAttention1D(width, norm=norm))
            blocks.append(CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm))
        else:
            blocks.append(nn.ReLU())
            
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1)
            ) if not use_attn else nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1),
                CausalAttention1D(width, norm=norm)
            )
            blocks.append(block)
            
        # end
        # blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(CausalConv1d(width, width, kernel_size, 1, (kernel_size-1)//2))
        blocks.append(nn.ReLU())
        # blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))
        blocks.append(CausalConv1d(width, input_emb_width, kernel_size, 1, (kernel_size-1)//2))
        
        if use_patcher:
            blocks.append(UnPatcher1D(patch_size, patch_method))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
