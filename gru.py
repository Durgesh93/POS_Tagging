# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:56:11 2019

@author: durgesh singh
"""

import math
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.init as init

    

class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_ih)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        
        init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        

    def forward(self, x, hx):
        dim_h = self.hidden_size
        
        wxr = self.weight_ih[:dim_h]
        wxz = self.weight_ih[dim_h:2 * dim_h]
        wxh = self.weight_ih[2 * dim_h:]
        
        whr =self.weight_hh[:dim_h]
        whz = self.weight_hh[dim_h:2 * dim_h]
        whh=self.weight_hh[2 * dim_h:]
        
        br = self.bias[:,:dim_h]
        bz = self.bias[:,dim_h:2 * dim_h]
        bh = self.bias[:,2 * dim_h:]
        
        qr = torch.sigmoid(x@wxr.t()+hx@whr.t()+br)
        qz = torch.sigmoid(x@wxz.t()+hx@whz.t()+bz)
        htil = torch.tanh(x@wxh.t()+(qr*hx)@whh.t()+bh)
        
        ht = qz*hx +(1-qz)*htil
        return ht

class GRUCellM1(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        
        
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_ih)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        
        init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        

    def forward(self, x, hx):
        dim_h = self.hidden_size
        
        wxr = self.weight_ih[:dim_h]
        wxz = self.weight_ih[dim_h:2 * dim_h]
        wxh = self.weight_ih[2 * dim_h:]
        
        whr =self.weight_hh[:dim_h]
        whz = self.weight_hh[dim_h:2 * dim_h]
        whh=self.weight_hh[2 * dim_h:]
        
        br = self.bias[:,:dim_h]
        bz = self.bias[:,dim_h:2 * dim_h]
        bh = self.bias[:,2 * dim_h:]
        
        qr = torch.sigmoid(x@wxz.t()+bz)
        qz = torch.sigmoid(x@wxr.t()+hx@whr.t()+br)
        ht = torch.tanh((qr*hx)@whh.t()+torch.tanh(x)+bh)*qz + hx*(1-qz)
        
        return ht
    
    

class GRUCellM2(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_ih)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        
        init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        

    def forward(self, x, hx):
        dim_h = self.hidden_size
        
        wxr = self.weight_ih[:dim_h]
        wxz = self.weight_ih[dim_h:2 * dim_h]
        wxh = self.weight_ih[2 * dim_h:]
        
        whr =self.weight_hh[:dim_h]
        whz = self.weight_hh[dim_h:2 * dim_h]
        whh=self.weight_hh[2 * dim_h:]
        
        br = self.bias[:,:dim_h]
        bz = self.bias[:,dim_h:2 * dim_h]
        bh = self.bias[:,2 * dim_h:]
        
        qz = torch.sigmoid(x@wxz.t()+hx@whz.t()+bz)
        qr = torch.sigmoid(x+hx@whr.t()+br)
        ht = torch.tanh((qr*hx)@whh.t()+x@wxh.t()+bh)*qz + hx*(1-qz)
        
        return ht
        


class GRUCellM3(nn.Module):

    def __init__(self, input_size, hidden_size,batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh =Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = Parameter(torch.Tensor(batch_size,3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_ih)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        
        init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        

    def forward(self, x, hx):
        dim_h = self.hidden_size
        
        wxr = self.weight_ih[:dim_h]
        wxz = self.weight_ih[dim_h:2 * dim_h]
        wxh = self.weight_ih[2 * dim_h:]
        
        whr =self.weight_hh[:dim_h]
        whz = self.weight_hh[dim_h:2 * dim_h]
        whh=self.weight_hh[2 * dim_h:]
        
        br = self.bias[:,:dim_h]
        bz = self.bias[:,dim_h:2 * dim_h]
        bh = self.bias[:,2 * dim_h:]
        
        qz = torch.sigmoid(x@wxz.t()+torch.tanh(hx)@whz.t()+bz)
        qr = torch.sigmoid(x@wxr.t()+hx@whr.t()+br)
        ht = torch.tanh((qr*hx)@whh.t()+x@wxh.t()+bh)*qz + hx*(1-qz)
        
        return ht              
        