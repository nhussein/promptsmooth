# evaluate a smoothed classifier on a dataset
from data.plip_datasets_clsnames import *
from tqdm import tqdm
from time import time
import datetime
from PIL import Image
import numpy as np 
import argparse
import os
# from copy import deepcopy

from math import ceil
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from CLIP import clip
from CLIP.clip.model import build_model

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import random
def set_seed(seed):
    """
    Set the seed for reproducibility in PyTorch, NumPy, and Python's random module.

    Args:
    - seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(3) 

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk][0]

def normalize_batch(batch, mean= (0.48145466, 0.4578275, 0.40821073), std= (0.26862954, 0.26130258, 0.27577711)):
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

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
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

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                
                #print('this batch size',this_batch_size)
                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                #predictions = self.base_classifier(batch + noise).argmax(1)
                
                image_features = model.encode_image(self._normalize_batch(batch + noise))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights
        
                _,predictions = logits.topk(1)
                
                #predictions = self.base_classifier(normalize_batch(batch + noise)).argmax(1)
                
                
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _normalize_batch(self, batch, mean= (0.48145466, 0.4578275, 0.40821073), std= (0.26862954, 0.26130258, 0.27577711)):
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


parser = argparse.ArgumentParser(description='Certify many examples')

# parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='kather', help='test dataset input any of the following: kather, PanNuke, SICAPv2, SkinCancer')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument("--n", type=int, default=500, help='number of test samples in the subset')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=100, help="batch size")
# parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
# parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
# parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')

args = parser.parse_args()

# args.n = 500
# args.dataset= 'kather' #['kather', 'PanNuke', 'SICAPv2', 'SkinCancer']
# args.sigma = 1
# args.outfile = "/home/noor.hussein/certify_TPT/certification_output/test/N10000/"

#-----------------------------------------------------------------------------------------------------
#CODE STARTS HERE
print('certifying zero-shot PLIP for:',args.dataset)

classnames =  eval("{}_classes".format(args.dataset.lower()))
n_classes = len(classnames)

# the 500 images subset can be obtained from script "sample_subset.py". Preferably name your subset the same name as the input dataset_names.
testdir = '/l/users/noor.hussein/datasets/subsets/{}_500subset/images/test'.format(args.dataset) 
my_transforms = preprocess(224)
testset = datasets.ImageFolder(testdir, transform=my_transforms)

# rearrange classnames according to their idx assignment from ImageFolder
# for n_classes >10
if n_classes > 10:
    ks = testset.class_to_idx.keys()
    new_classnames = [None]*n_classes
    for (i,j) in enumerate(ks):
        new_classnames[i] = classnames[int(j)]
        
    classnames = new_classnames

print(classnames)

loader = torch.utils.data.DataLoader(testset, batch_size = 1, num_workers=0)  

#template
if args.dataset=="kather":
    template = ["An H&E image patch of {}."]
elif args.dataset=="PanNuke":
    template = ["An H&E image of {} tissue."]
else:
    template = ['a histopathology slide showing {}']

print(template)

#load converted plip model weights 
model_path = '/l/users/asif.hanif/pre-trained-models/med-adv-prompt/plip/plip_model_converted.pt'
state_dict = torch.load(model_path)
model = build_model(state_dict).cuda()

zeroshot_weights = zeroshot_classifier(classnames, template)

# prepare output file
outfile = os.path.join(args.outfile, '{}'.format(args.dataset.lower()), 'samples500', 'sigma_{}'.format(args.sigma))
print(outfile)

if not os.path.exists(outfile.split('sigma')[0]):
    os.makedirs(outfile.split('sigma')[0])
    
f = open(outfile, 'w')
print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
f.close()
    
smoothed_classifier = Smooth(model, n_classes, args.sigma)

rad = []
corr = []

for i, (images, label) in enumerate(tqdm(loader)):

    images = images.cuda()
    label = label[0].cpu().numpy()#.cuda()
    
    before_time = time()
    prediction, radius = smoothed_classifier.certify(images, args.N0, args.N, args.alpha, args.batch)
    after_time = time()
    
    #print(prediction,label)
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


radi_values = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
# n = 500
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