## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
import logging

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from ipdb import set_trace as stx
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

import torch
from torchvision import transforms, datasets
from ipdb import set_trace as stx

from utils import MyDataset
import pynvml

import os
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

import importlib
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--result_folder', default='corrected', type=str, help='folder Directory for results')
parser.add_argument('--weights', default='./pretrained_models/motion_deblurring.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--flc', action='store_true')
parser.add_argument('--attack', action='store_true')
parser.add_argument('--zero_padding', action='store_true')
parser.add_argument('--adversarial', action="store_true", help="Adversarially train the model")
parser.add_argument('--adv_trained', action='store_true')
parser.add_argument('--method', default='cospgd', type=str)
parser.add_argument('--iterations', default=5, type=int)
parser.add_argument('--epsilon', default=0.03, type=float)
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument("--use_conv", help="use 1x1 convolution to increase the number of channels in low freqs", action="store_true")
parser.add_argument("--use_alpha", help="use flc_pooling with alpha blending while training", action="store_true")
parser.add_argument("--learn_alpha", help="use flc_pooling with learnable alpha blending instead of random while training", action="store_true")
parser.add_argument("--drop_alpha", help="use flc_pooling with learnable alpha blending, SUCH THAT ALPHA IS RANDOMLY ZERO AT TIMES instead of random while training", action="store_true")
parser.add_argument("--first_drop_alpha", help="use flc_pooling with learnable alpha blending, SUCH THAT ALPHA IS RANDOMLY ZERO AT TIMES instead of random while training", action="store_true")
parser.add_argument("--test_wo_drop_alpha", help="use flc_pooling with learnable alpha blending, SUCH THAT ALPHA IS RANDOMLY ZERO AT TIMES instead of random while training", action="store_true")
parser.add_argument("--test_drop_alpha", help="use flc_pooling with learnable alpha blending, SUCH THAT ALPHA IS RANDOMLY ZERO AT TIMES instead of random while training", action="store_true")
parser.add_argument("--blur", help="Use gaussian blur on high frequencies", action="store_true")
parser.add_argument('--kernel_size', type=int, help="kernel size for transposed convolution, if kernel size<2 then uses pixel shuffle", default=0)
parser.add_argument('--para_kernel_size', type=int, help="kernel size for parallel transposed convolution, if parallel kernel size<2 then not used", default=0)
parser.add_argument("--half_precision", help="use half precision", action="store_true")
parser.add_argument('--upsample_method', type=str, default='pixel')

args = parser.parse_args()
if args.adversarial:
    args.adv_trained = args.adversarial

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_id)

def get_memory_free_GB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 3


def init_linf(
            images,
            epsilon,
            clamp_min = 0,
            clamp_max = 1,
            half = False,
        ):
        noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(images.device).cuda()#.half()
        if half:
            noise = noise.half()
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images

def cospgd_scale(
        predictions,
        labels,
        loss,
        num_classes=None,
        targeted=False,
        detach = True,
    ):

    cossim = torch.nn.functional.cosine_similarity(
        torch.nn.functional.softmax(predictions, dim=1),
        torch.nn.functional.softmax(labels, dim=1),
        dim = 1
    )
    if targeted:
        cossim = 1 - cossim
    
    loss = cossim.detach() * loss if detach else cossim * loss
    return loss

def step_inf(
        perturbed_image,
        epsilon,
        data_grad,
        orig_image,
        alpha,
        targeted=False,
        clamp_min = 0,
        clamp_max = 1,
        grad_scale = None
    ):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = alpha*data_grad.sign()
    if targeted:
        sign_data_grad *= -1
    if grad_scale is not None:
        sign_data_grad *= grad_scale
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = perturbed_image.detach() + sign_data_grad
    # Adding clipping to maintain [0,1] range
    delta = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
    perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
    return perturbed_image


####### Load yaml #######
yaml_file = 'Options/Deblurring_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

if args.flc:
    x['network_g']['flc_pooling'] = True
    x['network_g']['use_conv'] = args.use_conv
    x['network_g']['use_alpha'] = args.use_alpha
    x['network_g']['learn_alpha'] = args.learn_alpha
    x['network_g']['use_blur'] = args.blur
    x['network_g']['padding'] = "constant" if args.zero_padding else "reflect"
x['network_g']['kernel_size']=args.kernel_size
x['network_g']['para_kernel_size']=args.para_kernel_size
x['network_g']['drop_alpha'] = args.drop_alpha
x['network_g']['test_wo_drop_alpha'] = args.test_wo_drop_alpha
x['network_g']['test_drop_alpha'] = args.test_drop_alpha
x['network_g']['half_precision'] = args.half_precision
x['network_g']['first_drop_alpha'] = args.first_drop_alpha
s = x['network_g'].pop('type')
##########################

model_restoration = Restormer(**x['network_g'])

x['attack']['method'] = args.method
x['attack']['alpha'] = args.alpha
x['attack']['epsilon'] = args.epsilon
x['attack']['iterations'] = args.iterations
x['attack']['targeted'] = False
x['attack']['adv_attack'] = args.attack

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'], strict = False)
print("===>Testing using weights: ",args.weights)
#model_restoration.cuda()#.half()
if args.half_precision:
    model_restoration.cuda().half()
else:
    model_restoration.cuda()#.half()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

"""
for name, module in model_restoration.named_parameters():
    if 'alpha' in name:
        print(module)
"""
if args.use_conv:
    concat_word = "conv"
else:
    concat_word = "concat"

if args.test_drop_alpha:
    test_drop_word = 'always'
elif args.test_wo_drop_alpha:
    test_drop_word = 'without'
else:
    test_drop_word = 'with'

if args.first_drop_alpha:
    drop_word = 'first_layer'
elif args.drop_alpha:
    drop_word = 'with'
else:
    drop_word = 'without'

factor = 8
dataset = args.dataset
if args.attack:
    if args.kernel_size > 0:
        result_dir  = os.path.join(args.result_dir, dataset, args.result_folder,  'trans_{}_para_trans_{}_flc_{}_adv_train_{}'.format(args.kernel_size, args.para_kernel_size, args.flc, args.adv_trained), args.method, 'iterations_{}'.format(args.iterations), 'alpha_{}'.format(args.alpha), 'epsilon_{}'.format(args.epsilon))
    else:    
        result_dir  = os.path.join(args.result_dir, dataset, args.result_folder, 'flc_low_freqs_{}_{}_alpha_{}_blur_{}_alpha_{}_drop_alpha_testing_{}_drop_alpha_adv_train_{}_{}_padding'.format(concat_word, 'without' if not args.use_alpha else 'with', 'with' if args.blur else 'without', 'learned' if args.learn_alpha else 'random', drop_word, test_drop_word, args.adv_trained, "zero" if args.zero_padding else "mirror"), args.method, 'iterations_{}'.format(args.iterations), 'alpha_{}'.format(args.alpha), 'epsilon_{}'.format(args.epsilon))
else:
    if args.kernel_size > 0:
        result_dir  = os.path.join(args.result_dir, dataset, args.result_folder, 'no_attack', 'trans_{}_para_trans_{}_flc_{}_adv_train_{}'.format(args.kernel_size, args.para_kernel_size, args.flc, args.adv_trained))
    else:    
        result_dir  = os.path.join(args.result_dir, dataset, args.result_folder, 'no_attack', 'flc_low_freqs_{}_{}_alpha_{}_blur_{}_alpha_{}_drop_alpha_testing_{}_drop_alpha_adv_train_{}_{}_padding'.format(concat_word, 'without' if not args.use_alpha else 'with', 'with' if args.blur else 'without', 'learned' if args.learn_alpha else 'random', drop_word, test_drop_word, args.adv_trained, "zero" if args.zero_padding else "mirror"))
os.makedirs(result_dir, exist_ok=True)

logging.basicConfig(filename='{}/log.log'.format(result_dir), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

logger.info(x)
logger.info(args)

data_folder = os.path.join(args.input_dir, 'test', dataset)
data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(360, 640))])
data = MyDataset(image_paths=data_folder, transform=data_transforms)

data_loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle=False, num_workers=8)

# define losses
pixel_opt={
    "type": "L1Loss",
    "loss_weight": 1,
    "reduction": "none"
}
pixel_type = pixel_opt.pop('type')
cri_pix_cls = getattr(loss_module, pixel_type)
cri_pix = cri_pix_cls(**pixel_opt).to('cuda')
psnr_value, ssim_value = 0, 0

with torch.enable_grad():
    #for i, (file_, targets) in enumerate(tqdm(zip(files, target_files))):
    #     
    pbar = tqdm(data_loader)
    for i, (input, gt, images_name) in enumerate(pbar):
        loss = 0
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        #stx()        
        #pbar.set_postfix_str("mem free: {} mem total: {}".format(get_memory_free_GB(args.gpu_id), (torch.cuda.get_device_properties(args.gpu_id).total_memory)//1024**3))
        input = input.cuda()#.half()
        
        gt = gt.cuda()#.half()
        if args.half_precision:
            input = input.half()
            gt = gt.half()

        orig_image = input.clone().cpu().detach()

        if args.attack and 'pgd' in args.method:
            input = init_linf(input, args.epsilon, half=args.half_precision)

        

        # Padding in case images are not multiples of 8
        h,w = input.shape[2], input.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input = F.pad(input, (0,padw,0,padh), 'reflect')

        input.requires_grad = True
        model_restoration.zero_grad()
        
        restored = model_restoration(input)        

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]
        restored = torch.clamp(restored, 0, 1)
        if args.attack:
            for itr in range(args.iterations):
                #model_restoration.cpu()
                for pred, target in zip(restored, gt):
                    loss += cri_pix(pred, target)
                            
                if args.method == 'cospgd':
                    loss = cospgd_scale(predictions=restored, labels=gt, loss=loss)
                #
                del restored
                gt = gt.cpu().detach()
                torch.cuda.empty_cache()
                #stx()
                grad = torch.autograd.grad(loss.mean(), input, retain_graph=False, create_graph=False)[0]                
                
                input = step_inf(perturbed_image=input.cpu().to(torch.float32), epsilon=args.epsilon, data_grad=grad.cpu(), orig_image=orig_image.to(torch.float32), alpha=args.alpha)
                gt = gt.cuda()#.half()
                input = input.cuda()#.half()
                
                model_restoration.cuda()#.half()
                if args.half_precision:
                    gt = gt.half()
                    input = input.half()
                    model_restoration.half()

                input.requires_grad = True
                restored = model_restoration(input)
                restored = restored[:,:,:h,:w]
                restored = torch.clamp(restored, 0, 1)  



        #restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        gt = gt.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        #for pred, target in zip(restored, gt):
        #psnr_value += PSNR(target, pred)
        #ssim_value += SSIM(target, pred)

        #stx()
        psnr_value += PSNR(gt, restored)        
        ssim_value += SSIM(gt, restored, channel_axis=2)
        free_mem = get_memory_free_GB(args.gpu_id)
        total_mem = (torch.cuda.get_device_properties(0).total_memory)//1024**3
        pbar.set_postfix_str("PSNR: {}  SSIM: {} mem: {}/{}".format(psnr_value/(i+1), ssim_value/(i+1), free_mem, total_mem))
        

        #stx()
        #for image, image_name in zip(restored, images_name):
        os.makedirs(os.path.join(result_dir, 'images'), exist_ok=True)
        #stx()
        utils.save_img((os.path.join(result_dir, 'images', images_name[0])), img_as_ubyte(restored.astype(np.float32)))

logger.info("PSNR: {}  SSIM: {}".format(psnr_value/len(data), ssim_value/len(data)))