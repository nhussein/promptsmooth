import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import save_image


ID_to_DIRNAME={
    'I': 'imagenet/images',
    # 'I': 'blackbox_attacks/resnet50/autoattack/images',
    'S': 'imagenet/images/subset100',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    # if set_id == 'I':
    #     # ImageNet validation set
    #     testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
    #     testset = datasets.ImageFolder(testdir, transform=transform)
    if set_id in ['I']:
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        import torchvision.datasets as datasets
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


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


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views





# to generate noisy augmentations of the input image:

def generate_noisy_sample(image, sigma):

    # Convert the image to a PyTorch tensor
    image_tensor = to_tensor(image)
    # Get the dimensions of the image
    ch, col, row = image_tensor.size()
    # Generate Gaussian noise
    gaussian_noise = torch.normal(0, sigma, (ch, col, row))
    # Add Gaussian noise to the image tensor
    noisy_image_tensor = image_tensor + gaussian_noise
    noisy_image = to_pil_image(noisy_image_tensor)

    return noisy_image

# to generate noisy augmentations of the input image (noise sampled from different distributions):

def generate_noisy_sample_diffdist(image):

    # Convert the image to a PyTorch tensor
    image_tensor = to_tensor(image)
   
    # Get the dimensions of the image
    ch, col, row = image_tensor.size()
   
    # Generate Gaussian noise
    sigma = random.uniform(0.05, 0.1)
    gaussian_noise = torch.normal(0, sigma, (ch, col, row))
   
    # Add Gaussian noise to the image tensor
    noisy_image_tensor = image_tensor + gaussian_noise
    noisy_image = to_pil_image(noisy_image_tensor)

    return noisy_image

# only noise
def augnoise(image, preprocess, base_transform, sigma):

    x_gen = generate_noisy_sample(image, sigma)
    
    # #noise sampled from different distribution
    # x_gen = generate_noisy_sample_diffdist(image)
    
    x = base_transform(x_gen)
    x_processed = preprocess(x)
    
    return x_processed

#add gaussian noise to TPT augmented views
def augmixnoise(image, preprocess, sigma):
    
    preaugment = get_preaugment()
    crop = preaugment(image)
    
    noisycrop = generate_noisy_sample(crop, sigma)
    
    # #noise sampled from different distribution
    # noisycrop = generate_noisy_sample_diffdist(crop)
    
    x_processed = preprocess(noisycrop)
    
    return x_processed


class AugNoiseAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, sigma = 0.5):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.sigma = sigma
        
    def __call__(self, x):
        
        # noisy_image = generate_noisy_sample(x, self.sigma)
        # image = self.preprocess(self.base_transform(noisy_image))
        # views = [augnoise(x, self.preprocess, self.base_transform, self.sigma) for _ in range(self.n_views)]
        
        noisy_image = generate_noisy_sample(x, self.sigma)
        image = self.preprocess(self.base_transform(noisy_image))
        views = [augnoise(x, self.preprocess, self.base_transform, self.sigma) for _ in range(self.n_views)] #TPT*
        # views = [augmixnoise(x, self.preprocess, self.sigma) for _ in range(self.n_views)] #TPT**
        
        return [image] + views
    

def normalize(image, sigma):
    
    to_tensor = transforms.ToTensor()

    image_tensor = to_tensor(image)

    # Add Gaussian noise
    noise = torch.randn_like(image_tensor) * sigma
    noisy_image_tensor = image_tensor + noise

    # Ensure values are in the valid range [0, 1]
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 1)
    
    return noisy_image_tensor    

class AddNoiseToImage(object):
    def __init__(self, base_transform, sigma = 0.25):
        self.base_transform = base_transform
        self.sigma = sigma
        
    def __call__(self, x):
        noisy_image = normalize(x, self.sigma)
        image = self.base_transform(noisy_image)
        return image