from layer import *

import torch
import torch.nn as nn

class BS3LNet(nn.Module):
    def __init__(self, nch_in=189, nch_out=189, nch_ker=64, norm='bnorm', nblk=2):
        super(BS3LNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.encoder = CNR2d(self.nch_in, self.nch_ker, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm=[], relu=0.0)

        resconv = []
        for i in range(self.nblk):
            resconv += [ResBlock(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm=self.norm, relu=0.0)]
        self.resconv = nn.Sequential(*resconv)

        self.decoder = CNR2d(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm=self.norm, relu=[])

        self.conv = Conv2d(self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1, padding_mode='reflection')

    def forward(self, x):
        x = self.encoder(x)
        x0 = x

        x = self.resconv(x)

        x = self.decoder(x)
        x = x + x0

        x = self.conv(x)

        return x