from math import exp
import torch
import torch.nn as nn
from denseblock import Dense
import config as c
from kan_layers import KANCouplingNet

class INV_block_addition(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()

        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp

        # ρ
        # self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # self.s2 = subnet_constructor(self.split_len2, self.split_len1)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            t1 = self.y(y1)
            y2 = x2 + t1

        else:  # names of x and y are swapped!

            t1 = self.y(x1)
            y2 = (x2 - t1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3, imp_map=True):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        if imp_map:
            self.imp = 12
        else:
            self.imp = 0
        chunk_size = getattr(c, "kan_chunk_size", 4096)
        identity_init = getattr(c, "kan_identity_init", True)
        identity_jitter = getattr(c, "kan_identity_jitter", 1e-3)
        kan_hidden_dims = getattr(c, "kan_hidden_dims", None)
        normalize_input = getattr(c, "kan_normalize_input", False)
        normalization_eps = getattr(c, "kan_normalization_eps", 1e-6)

        def make_kan(in_channels, out_channels):
            return KANCouplingNet(
                in_channels,
                out_channels,
                hidden_dims=kan_hidden_dims,
                identity_init=identity_init,
                identity_jitter=identity_jitter,
                verbose=c.kan_verbose,
                chunk_size=chunk_size,
                normalize_input=normalize_input,
                normalization_eps=normalization_eps,
            )

        if imp_map:
            use_scale_kan = getattr(c, "kan_stage2_use_scale_nets", True)
            use_translate_kan = getattr(c, "kan_stage2_use_translation_nets", False)
        else:
            use_scale_kan = getattr(c, "kan_stage1_use_scale_nets", False)
            use_translate_kan = getattr(c, "kan_stage1_use_translation_nets", False)

        scale_constructor = make_kan if use_scale_kan else subnet_constructor
        translate_constructor = make_kan if use_translate_kan else subnet_constructor

        # ρ
        self.r = scale_constructor(self.split_len1 + self.imp, self.split_len2)
        # η
        self.y = translate_constructor(self.split_len1 + self.imp, self.split_len2)
        # φ
        self.f = translate_constructor(self.split_len2, self.split_len1 + self.imp)
        # ψ
        self.p = scale_constructor(self.split_len2, self.split_len1 + self.imp)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = (x.narrow(1, 0, self.split_len1 + self.imp),
                  x.narrow(1, self.split_len1 + self.imp, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2)

        return torch.cat((y1, y2), 1)
