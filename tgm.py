import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import temporal_structure_filter as tsf

 
class TGM(tsf.TSF):
    """
    Subevents as temporal conv
    combine input channels and output channels with soft-attn
    """
    def __init__(self, inp, num_f,  length, c_in, c_out, soft=False):
        super(TGM, self).__init__(num_f)
        
        self.inp = inp
        self.length = length
        self.c_in = c_in
        self.c_out = c_out
        self.soft = soft
        
        self.soft_attn = nn.Parameter(torch.Tensor(c_out*c_in, num_f))
        # learn c_out combinations of the c_in channels
        if self.c_in > 1 and not soft:
            self.convs = nn.ModuleList([nn.Conv2d(self.c_in, 1, (1,1)) for c in range(self.c_out)])
        if self.c_in > 1 and soft:
            self.cls_attn = nn.Parameter(torch.Tensor(1,self.c_out, self.c_in,1,1))
    
    def forward(self, x):
        # overwrite the forward pass to get the TSF as conv kernels
        t = x.size(2)
        k = super(TGM, self).get_filters(torch.tanh(self.delta), torch.tanh(self.gamma), torch.tanh(self.center), self.length, self.length)
        # k is shape 1xNxL
        k = k.squeeze()
        # is k now NxL
        # apply soft attention to conver (C_out*C_in x N)*(NxL) to C_out*C_in x L

        # make attn sum to 1 along the num_gaussians
        soft_attn = F.softmax(self.soft_attn, dim=1)
        #print soft_attn
        k = torch.mm(soft_attn, k)

        # make k C_out*C_in x 1x1xL
        k = k.unsqueeze(1).unsqueeze(1)
        p = compute_pad(1, self.length, t)
        pad_f = p // 2
        pad_b = p - pad_f

        # x is shape CxDxT
        x = F.pad(x, (pad_f, pad_b))
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.expand(-1, self.c_in, -1, -1)

        # use groups to separate the class channels
        # but apply it in groups of C_out
        chnls = []
        for i in range(self.c_out):
            # gives k of C_in x1x1xL
            # output of C_in xDxT
            r = F.conv2d(x, k[i*self.c_in:(i+1)*self.c_in], groups=self.c_in).squeeze(1)
            # 1x1 conv to combine C_in to 1
            if self.c_in > 1 and not self.soft:
                r = F.relu(self.convs[i](r)).squeeze(1)
                #print 'r2', r.size()
            chnls.append(r)
        # get C_out x DxT
        f = torch.stack(chnls, dim=1)
        if self.c_in > 1 and self.soft:
            a = F.softmax(self.cls_attn, dim=2).expand(f.size(0), -1, -1, f.size(3), f.size(4))
            f = torch.sum(a * f, dim=1)
        return f



class TGMModel(nn.Module):
    def __init__(self, inp, classes=8):
        super(TGMModel, self).__init__()

        self.classes = classes
        self.dropout = nn.Dropout()
        self.add_module('d', self.dropout)

        self.sub_event1 = SubConv3(inp, 16, 5, c_in=1, c_out=8, soft=False)
        self.sub_event2 = SubConv3(inp, 16, 5, c_in=8, c_out=8, soft=False)
        self.sub_event3 = SubConv3(inp, 16, 5, c_in=8, c_out=8, soft=False)

        self.h = nn.Conv1d(inp+1*inp, 512, 1)
        self.classify = nn.Conv1d(512, classes, 1)
        self.inp = inp
        
    def forward(self, inp):
        val = False
        dim = 1
        if inp[0].size()[0] == 1:
            val = True
            dim = 0

        f = inp[0].squeeze()
        if val:
            f = f.unsqueeze(0)

        sub_event = self.sub_event1(f)
        sub_event = self.sub_event2(sub_event)
        sub_event = self.sub_event3(sub_event)
        sub_event = self.dropout(torch.max(sub_event, dim=1)[0])

        cls = F.relu(torch.cat([self.dropout(f), sub_event], dim=1))
        cls = F.relu(self.h(cls))
        return self.classify(cls)

    
