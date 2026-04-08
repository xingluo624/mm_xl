import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


from einops import rearrange

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = True

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

            
class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)



class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=False)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class Patcher1D(torch.nn.Module):
    """A module to convert 1D signal tensors into patches using torch operations."""
    
    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._haar(x)
        elif self.patch_method == "rearrange":
            return self._arrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # 1D padding
        x = F.pad(x, pad=(n - 2, n - 1), mode=mode).to(dtype)
        
        # 1D小波变换
        xl = F.conv1d(x, hl, groups=g, stride=2)  # 低通滤波
        xh = F.conv1d(x, hh, groups=g, stride=2)  # 高通滤波

        out = torch.cat([xl, xh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x):
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x):
        x = rearrange(
            x,
            "b c (l p) -> b (c p) l",
            p=self.patch_size,
        ).contiguous()
        return x


class UnPatcher1D(torch.nn.Module):
    """A module to convert 1D patches back into signal tensors."""
    
    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._ihaar(x)
        elif self.patch_method == "rearrange":
            return self._iarrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = x.shape[1] // 2  # 分成低频和高频两部分
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xl, xh = torch.chunk(x.to(dtype), 2, dim=1)

        # 1D逆变换
        yl = F.conv_transpose1d(xl, hl, groups=g, stride=2, padding=(n - 2))
        yh = F.conv_transpose1d(xh, hh, groups=g, stride=2, padding=(n - 2))
        y = yl + yh

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p) l -> b c (l p)",
            p=self.patch_size,
        )
        return x