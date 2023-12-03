import math
import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np

from basicsr.utils import get_root_logger

# try:
#     from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
#                                         modulated_deform_conv)
# except ImportError:
#     # print('Cannot import dcn. Ignore this warning if dcn is not used. '
#     #       'Otherwise install BasicSR with compiling dcn.')
#

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class FLC_Pooling_conv_blurred(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, channels = None, test_wo_drop_alpha=False, transpose=True, test_drop_alpha=False, stop = False, half_precision = False, padding = "reflect"):
        self.transpose = transpose
        self.window2d = None
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.stop = stop
        self.half_precision = half_precision
        self.channels = channels
        self.padding = padding

        super(FLC_Pooling_conv_blurred, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.conv = nn.conv2d(self.channels, 4*self.channels, 1, 0, bias=False)
        self.downsample_high = nn.PixelUnshuffle(2)
        self.drop = 0

    def forward(self, x):        
        device = x.device

        orig_x_size = x.shape
        orig_x = x.clone()

        x = F.pad(x, (3*x.shape[-1]//4 +1, 3*x.shape[-1]//4, 3*x.shape[-2]//4 +1, 3*x.shape[-2]//4), mode=self.padding)


        if self.transpose:
            x = x.transpose(2,3)

        in_freq = torch.fft.fftshift(torch.fft.fft2(x.to(torch.float32), norm='forward'))#.half()

        low_part = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]

        low_part =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward'))#.half()
        if self.half_precision:
            low_part = low_part.half()

        if self.transpose:
            low_part = low_part.transpose(2,3)     

        #low_part = torch.cat((low_part, low_part, low_part, low_part), dim=1)
        low_part = self.conv(low_part)

        self.drop = torch.tensor(np.random.choice(2, replace=True, p=[0.3, 0.7])).to(device)
        if self.test_drop_alpha:
            return T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)
        elif self.drop == 0 and not self.test_wo_drop_alpha:
            return T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)
        else:
            zeroed_high = torch.zeros_like(in_freq)
            zeroed_high[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)] = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
                        
            zeroed_high =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(zeroed_high), norm='forward'))#.half()
            if self.half_precision:
                zeroed_high = zeroed_high.half()
            if self.transpose:
                zeroed_high = zeroed_high.transpose(2,3)
            zeroed_high = T.CenterCrop((orig_x_size[-2], orig_x_size[-1]))(zeroed_high)

            high_part = orig_x - zeroed_high
            

            high_part = self.downsample_high(high_part)


            

            high_part = high_part
            
            return (T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)) + high_part


class FLC_Pooling_conv_blurred_alpha_dropout(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, channels = None, test_wo_drop_alpha=False, transpose=True, test_drop_alpha=False, stop = False, half_precision = False, padding = "reflect"):
        self.transpose = transpose
        self.window2d = None
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.stop = stop
        self.half_precision = half_precision
        self.channels = channels
        self.padding = padding

        super(FLC_Pooling_conv_blurred_alpha_dropout, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.conv = nn.conv2d(self.channels, 4*self.channels, 1, 0, bias=False)
        self.downsample_high = nn.PixelUnshuffle(2)
        self.drop = 0

    def forward(self, x):        
        device = x.device

        orig_x_size = x.shape
        orig_x = x.clone()

        x = F.pad(x, (3*x.shape[-1]//4 +1, 3*x.shape[-1]//4, 3*x.shape[-2]//4 +1, 3*x.shape[-2]//4), mode=self.padding)


        if self.transpose:
            x = x.transpose(2,3)

        in_freq = torch.fft.fftshift(torch.fft.fft2(x.to(torch.float32), norm='forward'))#.half()

        low_part = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]

        low_part =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward'))#.half()
        if self.half_precision:
            low_part = low_part.half()

        if self.transpose:
            low_part = low_part.transpose(2,3)     

        #low_part = torch.cat((low_part, low_part, low_part, low_part), dim=1)
        low_part = self.conv(low_part)

        self.drop = torch.tensor(np.random.choice(2, replace=True, p=[0.3, 0.7])).to(device)
        if self.test_drop_alpha:
            return T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)
        elif self.drop == 0 and not self.test_wo_drop_alpha:
            return T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)
        else:
            zeroed_high = torch.zeros_like(in_freq)
            zeroed_high[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)] = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
                        
            zeroed_high =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(zeroed_high), norm='forward'))#.half()
            if self.half_precision:
                zeroed_high = zeroed_high.half()
            if self.transpose:
                zeroed_high = zeroed_high.transpose(2,3)
            zeroed_high = T.CenterCrop((orig_x_size[-2], orig_x_size[-1]))(zeroed_high)

            high_part = orig_x - zeroed_high
            

            high_part = self.downsample_high(high_part)


            self.alpha = self.alpha.to(device)

            high_part = high_part*self.alpha
            
            return (T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)*(1- self.alpha)) + high_part

class FLC_Pooling_random_alpha_blurred(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, channels = None, test_wo_drop_alpha=False, transpose=True, test_drop_alpha=False, stop = False, half_precision = False, padding = "reflect"):
        self.transpose = transpose
        self.window2d = None
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.stop = stop
        self.half_precision = half_precision
        self.padding = padding

        super(FLC_Pooling_random_alpha_blurred, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.downsample_high = nn.PixelUnshuffle(2)
        self.drop = 0

    def forward(self, x):        
        #x = x.cuda()
        device = x.device

        orig_x_size = x.shape
        orig_x = x.clone()
        x = F.pad(x, (3*x.shape[-1]//4 +1, 3*x.shape[-1]//4, 3*x.shape[-2]//4 +1, 3*x.shape[-2]//4), mode=self.padding)

        if self.transpose:
            x = x.transpose(2,3)

        in_freq = torch.fft.fftshift(torch.fft.fft2(x.to(torch.float32), norm='forward'))
        #if self.half_precision:
        #    in_freq = in_freq.half()
        low_part = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
        low_part =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward'))

        if self.half_precision:
            low_part = low_part.half()

        if self.transpose:
            low_part = low_part.transpose(2,3)     

        low_part = torch.cat((low_part, low_part, low_part, low_part), dim=1)        
        if self.test_drop_alpha:
            return T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)

        zeroed_high = torch.zeros_like(in_freq)
        zeroed_high[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)] = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]

        zeroed_high =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(zeroed_high), norm='forward'))
        if self.half_precision:
            zeroed_high = zeroed_high.half()
        if self.transpose:
            zeroed_high = zeroed_high.transpose(2,3)
        zeroed_high = T.CenterCrop((orig_x_size[-2], orig_x_size[-1]))(zeroed_high)

        high_part = orig_x - zeroed_high     
        high_part = self.downsample_high(high_part)
        self.alpha = self.alpha.to(device)
        high_part = high_part*self.alpha    
        return (T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)*(1- self.alpha)) + high_part

class FLC_Pooling_learn_alpha_blurred_alpha_dropout(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, channels = None, test_wo_drop_alpha=False, transpose=True, test_drop_alpha=False, stop = False, half_precision = False, padding = "reflect"):
        self.transpose = transpose
        self.window2d = None
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.stop = stop
        self.half_precision = half_precision
        self.padding = padding

        super(FLC_Pooling_learn_alpha_blurred_alpha_dropout, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.downsample_high = nn.PixelUnshuffle(2)
        self.drop = 0

    def forward(self, x):        
        #x = x.cuda()
        device = x.device

        orig_x_size = x.shape
        orig_x = x.clone()

        #import ipdb;ipdb.set_trace()
        x = F.pad(x, (3*x.shape[-1]//4 +1, 3*x.shape[-1]//4, 3*x.shape[-2]//4 +1, 3*x.shape[-2]//4), mode=self.padding)


        if self.transpose:
            x = x.transpose(2,3)
            #orig_x = orig_x.transpose(2,3)
        
        #try:
        in_freq = torch.fft.fftshift(torch.fft.fft2(x.to(torch.float32), norm='forward'))#.half()
        #if self.half_precision:
        #    in_freq = in_freq.half()
        #except Exception as exp:
        #    x = x.to(torch.float32)
        #    #orig_x = orig_x.to(torch.float32)
        #    in_freq = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        #    in_freq = in_freq.half()
            #import ipdb;ipdb.set_trace()
        

        low_part = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
        #if self.stop:
        #    import ipdb;ipdb.set_trace()
        #low_part =  torch.fft.ifft2(torch.fft.ifftshift(low_part.to(torch.float32)), norm='forward').real.half()
        low_part =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward'))#.half()
        if self.half_precision:
            low_part = low_part.half()

        if self.transpose:
            low_part = low_part.transpose(2,3)     

        low_part = torch.cat((low_part, low_part, low_part, low_part), dim=1)
        self.drop = torch.tensor(np.random.choice(2, replace=True, p=[0.3, 0.7])).to(device)
        if self.test_drop_alpha:
            return T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)
        elif self.drop == 0 and not self.test_wo_drop_alpha:
            return T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)
        else:
            zeroed_high = torch.zeros_like(in_freq)
            zeroed_high[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)] = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]

                           
            #zeroed_high =  torch.fft.ifft2(torch.fft.ifftshift(zeroed_high.to(torch.float32)), norm='forward').real.half()
            zeroed_high =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(zeroed_high), norm='forward'))#.half()
            if self.half_precision:
                zeroed_high = zeroed_high.half()
            if self.transpose:
                zeroed_high = zeroed_high.transpose(2,3)
            zeroed_high = T.CenterCrop((orig_x_size[-2], orig_x_size[-1]))(zeroed_high)

            high_part = orig_x - zeroed_high
            

            high_part = self.downsample_high(high_part)


            self.alpha = self.alpha.to(device)
            high_part = high_part*self.alpha                        
            
            return (T.CenterCrop((orig_x_size[-2]//2, orig_x_size[-1]//2))(low_part)*(1- self.alpha)) + high_part



class FLC_Pooling_learn_alpha_blurred(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, transpose=True, channels = None, half_precision = False, padding = "reflect"):
        self.transpose = transpose
        self.window2d = None
        self.half_precision = half_precision
        self.padding = padding
        
        super(FLC_Pooling_learn_alpha_blurred, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.downsample_high = nn.Conv2d(channels, channels, kernel_size=7, stride=2, bias=False, padding=3)

    def forward(self, x):        
        #x = x.cuda()
        device = x.device

        orig_x_size = x.shape
        orig_x = x.clone()

        x = F.pad(x, (3*x.shape[-1]//4, 3*x.shape[-1]//4, 3*x.shape[-2]//4, 3*x.shape[-2]//4), mode=self.padding)


        if self.transpose:
            x = x.transpose(2,3)
            orig_x = orig_x.transpose(2,3)
        
        in_freq = torch.fft.fftshift(torch.fft.fft2(x.to(torch.float32), norm='forward'))
        #if self.half_precision:
        #    in_freq = in_freq.half()
        

        low_part = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
        zeroed_high = torch.zeros_like(in_freq)
        zeroed_high[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)] = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]

        
        #low_part =  torch.fft.ifft2(torch.fft.ifftshift(low_part.to(torch.float32)), norm='forward').real        
        low_part =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward'))        
        if self.half_precision:
            low_part = low_part.half()
        #zeroed_high =  torch.fft.ifft2(torch.fft.ifftshift(zeroed_high.to(torch.float32)), norm='forward').real
        zeroed_high =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(zeroed_high), norm='forward'))
        if self.half_precision:
            zeroed_high = zeroed_high.half()
        zeroed_high = T.CenterCrop((orig_x_size[-1], orig_x_size[-2]))(zeroed_high)
        high_part = orig_x - zeroed_high

        high_part = self.downsample_high(high_part)

        ### NEED TO BRING HIGH PART AND LOW PART TO THE SAME RESOLUTION !!!!

        self.alpha = self.alpha.to(device)

        return (T.CenterCrop((orig_x_size[-1]//2, orig_x_size[-2]//2))(low_part)*(1- self.alpha)) + high_part*self.alpha


class FLC_Pooling_learn_alpha(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, transpose=True):
        self.transpose = transpose
        self.window2d = None
        
        super(FLC_Pooling_learn_alpha, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad = True)

    def forward(self, x):        
        #x = x.cuda()
        device = x.device

        orig_x_size = x.shape

        x = F.pad(x, (3*x.shape[-1]//4, 3*x.shape[-1]//4, 3*x.shape[-2]//4, 3*x.shape[-2]//4), mode="reflect")


        if self.transpose:
            x = x.transpose(2,3)
        
        in_freq = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        

        low_part = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
        padded_low_part = F.pad(low_part, (low_part.shape[-1]//2, low_part.shape[-1]//2, low_part.shape[-2]//2, low_part.shape[-2]//2))
        try:
            high_part = in_freq - padded_low_part
        except:
            #import ipdb;ipdb.set_trace()
            try:
                high_part = in_freq - F.pad(padded_low_part, (in_freq.shape[-1]-padded_low_part.shape[-1], in_freq.shape[-2]-padded_low_part.shape[-2]))
            except:
                high_part = in_freq - T.CenterCrop((in_freq[-1], in_freq[-2]))(padded_low_part)
        high_part = torch.fft.ifftshift(high_part)
        high_part = high_part[:,:,high_part.shape[2]//2, high_part.shape[3]//2]
        high_part = torch.fft.ifft2(high_part, norm='forward').real
        low_part =  torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real        

        self.alpha = self.alpha.to(device)

        return (T.CenterCrop((orig_x_size[-1]//2, orig_x_size[-2]//2))(low_part)*(1- self.alpha)) + (T.CenterCrop((orig_x_size[-1]//2, orig_x_size[-2]//2))(high_part)*self.alpha)



class FLC_Pooling_random_alpha(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, transpose=True):
        self.transpose = transpose
        self.window2d = None
        self.alpha = None
        super(FLC_Pooling_random_alpha, self).__init__()

    def forward(self, x):        
        #x = x.cuda()
        device = x.device

        orig_x_size = x.shape

        x = F.pad(x, (3*x.shape[-1]//4, 3*x.shape[-1]//4, 3*x.shape[-2]//4, 3*x.shape[-2]//4), mode="reflect")


        if self.transpose:
            x = x.transpose(2,3)
        
        in_freq = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        

        low_part = in_freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
        padded_low_part = F.pad(low_part, (low_part.shape[-1]//2, low_part.shape[-1]//2, low_part.shape[-2]//2, low_part.shape[-2]//2))
        #high_part = in_freq - padded_low_part
        try:
            high_part = in_freq - padded_low_part
        except:
            #import ipdb;ipdb.set_trace()
            try:
                high_part = in_freq - F.pad(padded_low_part, (in_freq.shape[-1]-padded_low_part.shape[-1], in_freq.shape[-2]-padded_low_part.shape[-2]))
            except:
                high_part = in_freq - T.CenterCrop((in_freq[-1], in_freq[-2]))(padded_low_part)
        high_part = torch.fft.ifftshift(high_part)
        high_part = high_part[:,:,high_part.shape[2]//2, high_part.shape[3]//2]
        high_part = torch.fft.ifft2(high_part, norm='forward').real
        low_part =  torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real        

        self.alpha = torch.tensor(np.random.choice(10, replace=True, p=[0.35, 0.15, 0.1, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04])).to(device).to(torch.float32)
        self.alpha /= 10

        return (T.CenterCrop((orig_x_size[-1]//2, orig_x_size[-2]//2))(low_part)*(1- self.alpha)) + (T.CenterCrop((orig_x_size[-1]//2, orig_x_size[-2]//2))(high_part)*self.alpha)


class FLC_Pooling(nn.Module):
    # pooling through selecting only the low frequent part in the fourier domain and only using this part to go back into the spatial domain
    # save computations as we do not need to do the downsampling trough conv with stride 2
    def __init__(self, transpose=True):
        self.transpose = transpose
        self.window2d = None
        super(FLC_Pooling, self).__init__()

    def forward(self, x):        
        #x = x.cuda()
        device = x.device
        x = x.to(torch.float32).to(device)
        #import ipdb;ipdb.set_trace()
        if self.window2d is None:
            size=x.size(2)
            window1d = np.abs(np.hamming(size))
            window2d = np.sqrt(np.outer(window1d,window1d))
            window2d = torch.Tensor(window2d)#.cuda()
            self.window2d = window2d.unsqueeze(0).unsqueeze(0)

        orig_x_size = x.shape
        x = F.pad(x, (x.shape[-1]//1, x.shape[-1]//1, x.shape[-2]//1, x.shape[-2]//1))


        if self.transpose:
            x = x.transpose(2,3)
        
        low_part = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'))
        #low_part = low_part.cuda()*self.window2d
        try:
            assert low_part.size(2) == self.window2d.size(2)
            assert low_part.size(3) == self.window2d.size(3)
            low_part = low_part*self.window2d
        except Exception:
            try:
                assert low_part.size(2) == self.window2d.size(2)
                assert low_part.size(3) == self.window2d.size(3)
                low_part = low_part.cuda()*self.window2d.cuda()
            except Exception:
                #import ipdb;ipdb.set_trace()
                window1d = np.abs(np.hamming(x.shape[2]))
                window1d_2 = np.abs(np.hamming(x.shape[3]))
                window2d = np.sqrt(np.outer(window1d,window1d_2))
                window2d = torch.Tensor(window2d)
                self.window2d = window2d.unsqueeze(0).unsqueeze(0)
                low_part = low_part.to(device)*self.window2d.to(device)
        
        #low_part = low_part[:,:,int(x.size()[2]/4):int(x.size()[2]/4*3),int(x.size()[3]/4):int(x.size()[3]/4*3)]
        low_part = low_part[:,:,int(orig_x_size[2]/4):int(orig_x_size[2]/4*3),int(orig_x_size[3]/4):int(orig_x_size[3]/4*3)]
        
        return torch.fft.ifft2(torch.fft.ifftshift(low_part), norm='forward').real.half()#.cuda()

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# class DCNv2Pack(ModulatedDeformConvPack):
#     """Modulated deformable conv for deformable alignment.
#
#     Different from the official DCNv2Pack, which generates offsets and masks
#     from the preceding features, this DCNv2Pack takes another different
#     features to generate offsets and masks.
#
#     Ref:
#         Delving Deep into Deformable Alignment in Video Super-Resolution.
#     """
#
#     def forward(self, x, feat):
#         out = self.conv_offset(feat)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#
#         offset_absmean = torch.mean(torch.abs(offset))
#         if offset_absmean > 50:
#             logger = get_root_logger()
#             logger.warning(
#                 f'Offset abs mean is {offset_absmean}, larger than 50.')
#
#         return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
#                                      self.stride, self.padding, self.dilation,
#                                      self.groups, self.deformable_groups)
