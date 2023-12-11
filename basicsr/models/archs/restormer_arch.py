## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from basicsr.models.archs.arch_util import FLC_Pooling, FLC_Pooling_learn_alpha, FLC_Pooling_random_alpha, FLC_Pooling_learn_alpha_blurred, FLC_Pooling_random_alpha_blurred, FLC_Pooling_learn_alpha_blurred_alpha_dropout



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.previous_q = None
        self.previous_k = None
        self.previous_temp = None
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        try:
            attn = (q @ k.transpose(-2, -1)) * self.temperature
        except Exception as e:
            print(str(e))
            import ipdb;ipdb.set_trace()
        self.previous_q, self.previous_k, self.previous_temp = q.clone().detach(), k.clone().detach(), self.temperature.clone().detach()
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class FLC_Downsample(nn.Module):
    def __init__(self, n_feat, use_conv, use_alpha, learn_alpha, use_blur, drop_alpha, test_wo_drop_alpha, test_drop_alpha, transpose = False, stop=False, half=False, padding="reflect"):
        super(FLC_Downsample, self).__init__()
        self.use_conv = use_conv
        self.use_alpha = use_alpha
        self.learn_alpha = learn_alpha
        self.use_blur = use_blur
        self.drop_alpha = drop_alpha
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.stop = stop
        self.transpose = transpose
        self.channel_multiplier = 1 #if self.transpose else 1
        self.half_precision = half
        self.padding = padding

        if self.use_conv:
            if self.use_alpha:
                #self.body = nn.Sequential(FLC_Pooling_conv_blurred_alpha_dropout(channels=n_feat, transpose=self.transpose, test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, stop = stop, half_precision = self.half_precision, padding=self.padding), )                        
                self.body = nn.Sequential(nn.Conv2d(n_feat*self.channel_multiplier, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_conv_blurred_alpha_dropout(channels=n_feat, transpose=self.transpose, test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, stop = stop, half_precision = self.half_precision, padding=self.padding), )                        
            else:
                #self.body = nn.Sequential(FLC_Pooling_conv_blurred(channels=n_feat, transpose=self.transpose, test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, stop = stop, half_precision = self.half_precision, padding=self.padding))                        
                self.body = nn.Sequential(nn.Conv2d(n_feat*self.channel_multiplier, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_conv_blurred(channels=n_feat, transpose=self.transpose, test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, stop = stop, half_precision = self.half_precision, padding=self.padding))                        
        elif self.use_alpha:
            if self.learn_alpha:
                if self.use_blur:
                    if self.drop_alpha:
                        self.body = nn.Sequential(nn.Conv2d(n_feat*self.channel_multiplier, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_learn_alpha_blurred_alpha_dropout(channels=n_feat, transpose=self.transpose, test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, stop = stop, half_precision = self.half_precision, padding=self.padding),)                        
                    else:
                        #self.body = nn.Sequential(FLC_Pooling_random_alpha_blurred(channels=n_feat, transpose=self.transpose, test_drop_alpha=self.test_drop_alpha, half_precision = self.half_precision, padding=self.padding), nn.Conv2d(n_feat*self.channel_multiplier, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),)
                        self.body = nn.Sequential(nn.Conv2d(n_feat*self.channel_multiplier, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_random_alpha_blurred(channels=n_feat, transpose=self.transpose, test_drop_alpha=self.test_drop_alpha, half_precision = self.half_precision, padding=self.padding), )
                else:
                    #self.body = nn.Sequential(FLC_Pooling_learn_alpha(transpose=self.transpose), nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),)
                    self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_learn_alpha(transpose=self.transpose), )
            else:
                if self.use_blur:
                    self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_random_alpha_blurred(channels=n_feat, transpose=self.transpose),)
                else:
                    self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling_random_alpha(transpose=self.transpose), )
        else:
            self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False), FLC_Pooling(transpose=self.transpose), )

    def forward(self, x):
        return self.body(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class TransposedUpsample(nn.Module):
    def __init__(self, n_feat, kernel_size, para_kernel_size):
        super(TransposedUpsample, self).__init__()
        self.kernel_size = kernel_size
        self.para_kernel_size = para_kernel_size
        output_padding = 0 if self.kernel_size==2 else 1
        padding = 0
        para_padding = 0
        if kernel_size !=2:
            padding = (kernel_size-1)//2
        if para_kernel_size !=2:
            para_padding = (para_kernel_size-1)//2
        groups = n_feat//2
        self.body = nn.ConvTranspose2d(n_feat, n_feat//2, kernel_size = self.kernel_size, stride=2, padding=padding,
                                                    groups=groups, output_padding=output_padding)
        self.para = nn.ConvTranspose2d(n_feat, n_feat//2, kernel_size = self.para_kernel_size, stride=2, padding=para_padding,
                                                    groups=groups, output_padding=output_padding) if self.para_kernel_size > 0 else None

    def forward(self, x):
        return self.body(x) if self.para_kernel_size == 0 else self.body(x)+self.para(x)

class FreqAvgUpsample(nn.Module):
    def __init__(self, n_feat, padding='zero'):
        super(FreqAvgUpsample, self).__init__()
        self.padding = 'constant' if padding =='zero' else 'mirror'
        self.body = nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv1 = nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, groups=n_feat//2, bias=False)
        self.beta = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.shuffle = nn.PixelShuffle(2)        

    def forward(self, x):
        dtype = x.dtype
        x = self.body(x)
        #import ipdb;ipdb.set_trace()
        channels = x.shape[1]
        freq = torch.fft.fft2(x.to(torch.float32), norm='forward')

        avg_list, avg_channel_list = [], []
        for i in range(0, freq.shape[1], 4):
            avg = torch.mean(freq[:,i:i+4,:,:], dim=1)
            avg = torch.unsqueeze(avg, dim=1)
            avg_channels = torch.cat([avg]*4, dim=1)
            avg_list.append(avg)
            avg_channel_list.append(avg_channels)
        #import ipdb;ipdb.set_trace()
        tmp = torch.cat(avg_list, dim=1)
        avg_list = torch.cat(avg_list, dim=1)
        avg_channel_list = torch.cat(avg_channel_list, dim=1)
                
        #avg = torch.mean(freq, dim=1)
        
        #padding = F.pad(freq, (x.shape[-2], x.shape[-1]), mode=self.padding)
        #freqUp = torch.fft.ifft2(padding, norm='forward').to(dtype)
        #freqUp = self.conv1(freqUp)
        
        #avg = torch.unsqueeze(avg, dim=1)
        #avg = torch.cat([avg]*(channels//avg.shape[1]), dim=1)
        
        freq = freq - avg_channel_list
        freq = torch.fft.ifft2(freq, norm='forward').to(dtype)
        highFreq = self.shuffle(freq)

        padding = F.pad(avg_list, (x.shape[-1], 0, x.shape[-2], 0), mode=self.padding)
        freqUp = torch.fft.ifft2(padding, norm='forward').to(dtype)
        #freqUp = self.conv1(freqUp)

        return freqUp*(1-self.beta) + self.beta*highFreq

class SplitUpsampled(nn.Module):
    def __init__(self, n_feat, padding='zero'):
        super(SplitUpsampled, self).__init__()
        self.padding = 'constant' if padding=='zero' else 'mirror'
        self.beta = nn.Parameter(torch.tensor(0.3), requires_grad = True)
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    
    def forward(self, x):
        dtype = x.dtype
        x = self.body(x)
        freq = torch.fft.fftshift(torch.fft.fft2(x.to(torch.float32), norm='forward'))

        low = freq[:,:,int(x.shape[2]/4):int(x.shape[2]/4*3),int(x.shape[3]/4):int(x.shape[3]/4*3)]
        low = F.pad(low, (x.shape[2]//4, x.shape[2]//4, x.shape[3]//4, x.shape[3]//4), mode=self.padding)

        low =  torch.abs(torch.fft.ifft2(torch.fft.ifftshift(low), norm='forward')).to(dtype)
        high = x - low

        return (1 - self.beta)*low + self.beta*high





##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias', ## Other option 'BiasFree'
        dual_pixel_task = False,     ##  True for dual-pixel defocus deblurring only. Also set inp_channels=6
        flc_pooling = False,
        use_conv = False,
        use_alpha = False,   
        learn_alpha = False,
        drop_alpha = False,
        first_drop_alpha = False,
        test_wo_drop_alpha = False,
        test_drop_alpha = False,
        use_blur = False,
        kernel_size = 0,
        para_kernel_size = 0,
        half_precision = False,
        padding = "reflect",
        upsampling_method = 'pixel'
    ):

        super(Restormer, self).__init__()
        self.kernel_size = kernel_size
        self.para_kernel_size = para_kernel_size
        self.flc_pooling = flc_pooling
        self.use_conv = use_conv   
        self.use_alpha = use_alpha   
        self.learn_alpha = learn_alpha
        self.use_blur = use_blur
        self.drop_alpha = drop_alpha
        self.first_drop_alpha = first_drop_alpha
        self.test_wo_drop_alpha = test_wo_drop_alpha
        self.test_drop_alpha = test_drop_alpha
        self.dim = dim
        self.half_precision = half_precision
        self.padding = padding
        self.upsampling_method = upsampling_method

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = FLC_Downsample(dim, self.use_conv, self.use_alpha, self.learn_alpha, self.use_blur, drop_alpha= True if self.first_drop_alpha else self.drop_alpha, test_wo_drop_alpha = self.test_wo_drop_alpha, test_drop_alpha=self.test_drop_alpha, transpose=True, half=self.half_precision, padding = self.padding) if self.flc_pooling else Downsample(dim) ## From Level 1 to Level 2
        #self.down1_2_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = FLC_Downsample(int(dim*2**1), self.use_conv, self.use_alpha, self.learn_alpha, self.use_blur, self.drop_alpha, self.test_wo_drop_alpha, self.test_drop_alpha, transpose=False, half=self.half_precision, padding = self.padding) if self.flc_pooling else Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        #self.down2_3_2 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = FLC_Downsample(int(dim*2**2), self.use_conv, self.use_alpha, self.learn_alpha, self.use_blur, self.drop_alpha, self.test_wo_drop_alpha, self.test_drop_alpha, transpose=True, stop = False, half=self.half_precision, padding = self.padding) if self.flc_pooling else Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        #print('\n\n\t\t{}\n\n'.format(self.upsampling_method))
        if self.upsampling_method == 'pixel':
            self.up4_3 = Upsample(int(dim*2**3))
        elif self.upsampling_method == 'FreqAvgUp':
            self.up4_3 = FreqAvgUpsample(n_feat=int(dim*2**3))
        elif self.upsampling_method == 'SplitUp':
            self.up4_3 = SplitUpsampled(n_feat=int(dim*2**3))
        elif kernel_size>2:
            self.up4_3 = TransposedUpsample(int(dim*2**3), kernel_size=self.kernel_size, para_kernel_size = self.para_kernel_size)
        #self.up4_3 = Upsample(int(dim*2**3)) if kernel_size<2 else TransposedUpsample(int(dim*2**3), kernel_size=self.kernel_size, para_kernel_size = self.para_kernel_size) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        if self.upsampling_method == 'pixel':
            self.up3_2 = Upsample(int(dim*2**2))
        elif self.upsampling_method == 'FreqAvgUp':
            self.up3_2 = FreqAvgUpsample(n_feat=int(dim*2**2))
        elif self.upsampling_method == 'SplitUp':
            self.up3_2 = SplitUpsampled(n_feat=int(dim*2**2))
        elif kernel_size>2:
            self.up3_2 = TransposedUpsample(int(dim*2**2), kernel_size=self.kernel_size, para_kernel_size = self.para_kernel_size)
        
        #self.up3_2 = Upsample(int(dim*2**2)) if kernel_size<2 else TransposedUpsample(int(dim*2**2), kernel_size=self.kernel_size, para_kernel_size = self.para_kernel_size) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        if self.upsampling_method == 'pixel':
            self.up2_1 = Upsample(int(dim*2**1))
        elif self.upsampling_method == 'FreqAvgUp':
            self.up2_1 = FreqAvgUpsample(n_feat=int(dim*2**1))
        elif self.upsampling_method == 'SplitUp':
            self.up2_1 = SplitUpsampled(n_feat=int(dim*2**1))
        elif kernel_size>2:
            self.up2_1 = TransposedUpsample(int(dim*2**1), kernel_size=self.kernel_size, para_kernel_size = self.para_kernel_size)
        
        #self.up2_1 = Upsample(int(dim*2**1)) if kernel_size<2 else TransposedUpsample(int(dim*2**1), kernel_size=self.kernel_size, para_kernel_size = self.para_kernel_size)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        #import ipdb;ipdb.set_trace()
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        #import ipdb;ipdb.set_trace()
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        #import ipdb;ipdb.set_trace()
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        try:
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        except Exception:
            inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3.permute(0, 1, 3, 2)], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        try:
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        except Exception:
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2.permute(0, 1, 3, 2)], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        try:
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        except Exception:
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1.permute(0, 1, 3, 2)], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            try:
                out_dec_level1 = self.output(out_dec_level1) + inp_img
            except Exception:
                out_dec_level1 = self.output(out_dec_level1.permute(0, 1, 3, 2)) + inp_img


        return out_dec_level1

