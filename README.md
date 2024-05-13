# PromptSmooth
This is the code base for PromptSmooth: Certifying Robustness of Medical Vision-Language Models via Prompt Learning

## Installation
#### Setup conda environment.
```bash
# Create a conda environment
conda create -n promptsmooth python=3.8

# Activate the environment
conda activate promptsmooth

# Install torch (requires version >= 1.13) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
```
* #### Clone PromptSmooth code repository and install requirements
```bash
cd PromptSmooth/

# Install requirements
pip install -r requirements.txt
```

## Pre-trained Weights
### Medical Vision language model pretrained weights		
Please download PLIP wights from this [link](https://drive.google.com/file/d/1zwreSf0IYuTNJoLVymXJCEGWeEKUiWmi/view?usp=sharing). After downloading the zip file, have the folder in this structure PromptSmooth/pretrained_weights/..

## Data Preparation
In our paper we use a 500 images subset from each dataset. You can either use the script () to create your copy of the subset. However, we made it easier for you, you can just download the subset of kather dataset from this [link](https://drive.google.com/file/d/19BSLq5PHFhWhM90mU1-jHpVGId4HS6d3/view?usp=sharing). The folder structure then should be PromptSmooth/subsets/..

## Run Certification Scripts

