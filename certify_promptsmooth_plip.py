
# evaluate a smoothed classifier on a dataset
import os
import argparse
import datetime
import numpy as np 
from PIL import Image
from tqdm import tqdm
from time import time
from data.plip_datasets_clsnames import *

from copy import deepcopy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torchvision.datasets as datasets
from clip_tpt.custom_plip import get_coop

from scipy.stats import binom_test, norm
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import data.augmix_ops as augmentations

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Example usage
set_seed(3) 

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix

def normalize(batch, mean= (0.48145466, 0.4578275, 0.40821073), std= (0.26862954, 0.26130258, 0.27577711)):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (batch - mean) / std

def denormalize(normalized_batch, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (normalized_batch * std) + mean

class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, sigma = 0.1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.sigma = sigma 
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
    
    #noisy copies for 
    def __call__(self,x):
        image = self.preprocess(self.base_transform(x))
        # noisy_copy = normalize(denormalize(image)+torch.randn_like(image)*self.sigma)
        # return [noisy_copy]
        batch = image.repeat((100, 1, 1, 1))
        noisy_copies = normalize(denormalize(batch)+torch.randn_like(batch)*self.sigma)
        return [image] + list(noisy_copies)

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def test_time_tuning(model, inputs, optimizer, scaler, args):    
    selected_idx = None
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            output = model(inputs) 

            # if selected_idx is not None:
            #     output = output[selected_idx]
            # else:
            #     output, selected_idx = select_confident_samples(output, args.selection_p)

            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    return

def _normalize_batch(batch, mean= (0.48145466, 0.4578275, 0.40821073), std= (0.26862954, 0.26130258, 0.27577711)):
    """
    Normalize a batch of images.

    Args:
    - batch (Tensor): A batch of images with shape [batch, channel, width, height].
    - mean (tuple): A tuple of means for each channel.
    - std (tuple): A tuple of standard deviations for each channel.

    Returns:
    - Tensor: The normalized batch of images.
    """
    mean = torch.tensor(mean).cuda().view(-1, 1, 1)
    std = torch.tensor(std).cuda().view(-1, 1, 1)
    return (batch - mean) / std

def _denormalize_batch(normalized_batch, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    """
    Denormalize a batch of normalized images.

    Args:
    - normalized_batch (Tensor): A batch of normalized images with shape [batch, channel, width, height].
    - mean (tuple): A tuple of means for each channel (used in the original normalization).
    - std (tuple): A tuple of standard deviations for each channel (used in the original normalization).

    Returns:
    - Tensor: The denormalized batch of images.
    """
    mean = torch.tensor(mean).cuda().view(-1, 1, 1)
    std = torch.tensor(std).cuda().view(-1, 1, 1)
    return (normalized_batch * std) + mean

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int):

        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)

        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        
        counts = np.zeros(self.num_classes, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noisybatch = _normalize_batch(_denormalize_batch(batch)+torch.randn_like(batch, device='cuda') * self.sigma)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = model(noisybatch)
            
            _, predictions = logits.topk(1)

            counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
        return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

parser = argparse.ArgumentParser(description='Certify many examples')


parser.add_argument('--dataset', type=str, default='kather', help='test dataset input any of the following: kather, PanNuke, SICAPv2, SkinCancer')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--arch', metavar='ARCH', default='ViT-B/32')
parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=100, help="batch size")
parser.add_argument("--n", type=int, default=500, help='number of test samples in the subset')
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
parser.add_argument('--zeroshot', default=False, type=bool, help='use zeroshot promptsmooth')
parser.add_argument('--fewshot', default=False, type=bool, help='use fewshot promptsmooth')
parser.add_argument('--load', type=str, help='path to fewshot learned weights')
# parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')

args = parser.parse_args()

# args.n=500
# args.dataset= 'kather' #['kather', 'PanNuke', 'SICAPv2', 'SkinCancer']
# args.zeroshot = True # or False
# args.fewshot = True # or False
# args.sigma = 1.0 # [0.1, 0.25, 0.5, 1.0]
# args.arch = 'ViT-B/32'
# args.outfile = "./certification_output/PromptSmooth/PLIP"
# args.load = "./PromptSmooth/pretrained_weights/fewshot_weights/kather_plip/FewshotPromptSmooth/vit_b32_ep50_16shots/nctx5_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"


#-----------------------------------------------------------------------------------------------------
#CODE STARTS HERE


# data transform 
# norm stats from plip.load()
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
if args.zeroshot:
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=63, sigma = args.sigma)
else:
    data_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        normalize,
    ])


print('certifying PromptSmooth PLIP for:', args.dataset)
classnames =  eval("{}_classes".format(args.dataset.lower()))
n_classes = len(classnames)

# Get dataset
# the 500 images subset can be obtained from script "sample_subset.py". Preferably name your subset the same name as the input args.dataset.
testdir = './subsets/{}_500subset/images/test'.format(args.dataset) 
testset = datasets.ImageFolder(testdir, transform=data_transform)

# rearrange classnames according to their idx assignment from ImageFolder
# for n_classes > 10
if n_classes > 10:
    ks = testset.class_to_idx.keys()
    new_classnames = [None]*n_classes
    for (i,j) in enumerate(ks):
        new_classnames[i] = classnames[int(j)]
        
    classnames = new_classnames

print(classnames)

#dataset laoder
loader = torch.utils.data.DataLoader(testset, batch_size = 1, num_workers=0) 

#ctx_init and n_ctx
if args.dataset in ["kather", "PanNuke"]:
    args.ctx_init = 'An_H&E_image_patch_of'
    args.n_ctx = 5
else:
    args.ctx_init = 'a_histopathology_slide_showing'
    args.n_ctx = 4
    
if args.fewshot:
    args.ctx_init = None

print("####", args.ctx_init)

#load zero-shot clip with tunable parameters
model = get_coop(args.arch, args.dataset, args.gpu, args.n_ctx, args.ctx_init)              
model_state = None
#load few-shot weights
if args.fewshot:
    print("Use pre-trained soft prompt (few-shot) as initialization")
    pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
    assert pretrained_ctx.size()[0] == args.n_ctx
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(pretrained_ctx)
        model.prompt_learner.ctx_init_state = pretrained_ctx

for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

assert args.gpu is not None
torch.cuda.set_device(args.gpu)
model = model.cuda(args.gpu)

# define optimizer
trainable_param = model.prompt_learner.parameters()
optimizer = torch.optim.AdamW(trainable_param, args.lr)
optim_state = deepcopy(optimizer.state_dict())

scaler = torch.cuda.amp.GradScaler(init_scale=1000)
cudnn.benchmark = True

model.reset_classnames(classnames, args.arch)

# prepare output file
#100 samples
outfile = os.path.join(args.outfile, '{}'.format(args.dataset.lower()), 'samples_{}'.format(args.n), 'sigma_{}'.format(args.sigma))
print(outfile)

if not os.path.exists(outfile.split('sigma')[0]):
    os.makedirs(outfile.split('sigma')[0])
    
f = open(outfile, 'w')
print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
f.close()

rad = []
corr = []
for i, (images, label) in enumerate(tqdm(loader)):
    
    if args.zeroshot:
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
        images = torch.cat(images)
        images = images.cuda()
        
        #separate images list into 100 noisy images that will be inout to prompt tunning 
        # and image which will be an input to certification with new base classifier
        noisy_copies = images[1:]
        image = images[0].unsqueeze(0)
        
        label = label[0].cpu().numpy()#.cuda()
    else:
        image = images.cuda()
        label = label[0].cpu().numpy()#.cuda()
    
    # breakpoint()
    # zero-shot promptsmooth
    if args.zeroshot:
        model.eval()
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)
        test_time_tuning(model, noisy_copies, optimizer, scaler, args)
    
    # certification
    smoothed_classifier = Smooth(model, n_classes, args.sigma)
    before_time = time()
    prediction, radius = smoothed_classifier.certify(image, args.N0, args.N, args.alpha, args.batch)
    after_time = time()
    
    
    correct = int(prediction == label)
    
    time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
    
    f = open(outfile, 'a')
    print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
    print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        i, label, prediction, radius, correct, time_elapsed), flush=True)
    
    rad.append(radius)
    corr.append(correct)

    f.close()

radi_values = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5]
n = args.n  #changes according to subset from args.n
# Iterate over each value of radi
for radi in radi_values:
    tot = 0
    for i in range(len(rad)):
        if rad[i] > radi and corr[i] == 1:
            tot += 1
    f = open(outfile, 'a')
    print(f"Total accuracy is {(tot/n)*100} against radii {radi}", file = f)
    f.close()