#  ***PromptSmooth***: Certifying Robustness of Medical Vision-Language Models via Prompt Learning [MICCAI 2024]

<p align="center">

  <a href="https://ae.linkedin.com/in/noor-hussein-67566a183/">Noor Hussein</a>,
 <a href="https://fahadshamshad.github.io/">Fahad Shamshad</a>,
 <a href="https://muzammal-naseer.com/">Muzammal Naseer</a>, and 
 <a href="https://scholar.google.com.pk/citations?user=2qx0RnEAAAAJ&hl=en">Karthik Nandakumar</a>
 <br>
    <span style="font-size:1em; "><strong> Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), UAE</strong>.</span>
</p>

  <a href="https://arxiv.org/abs/2408.16769" target='_blank'>
      <img src="https://img.shields.io/badge/arXiv-Paper-brown.svg">
  </a>

  <p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>

Official implementation of the paper "PromptSmooth: Certifying Robustness of Medical Vision-Language Models via Prompt Learning".

<hr>

## Updates
* QUILT and MedCLIP code. [will be released soon]
* Code for PromptSmooth is released. [August 29, 2024]
* Our paper is accepted at MICCAI 2024. [June 18, 2024]

## Highlights
![methodology](https://github.com/nhussein/promptsmooth/blob/main/PromptSmooth.png)

> **Abstract:** *Medical vision-language models (Med-VLMs) trained on large datasets of medical image-text pairs and later fine-tuned for specific tasks have emerged as a mainstream paradigm in medical image analysis. However, recent studies have highlighted the susceptibility of these Med-VLMs to adversarial attacks, raising concerns about their safety and robustness. Randomized smoothing is a well-known technique for turning any classifier into a model that is certifiably robust to adversarial perturbations. However, this approach requires retraining the Med-VLM-based classifier so that it classifies well under Gaussian noise, which is often infeasible in practice. In this paper, we propose a novel framework called PromptSmooth to achieve efficient certified robustness of Med-VLMs by leveraging the concept of prompt learning. Given any pre-trained Med-VLM, PromptSmooth adapts it to handle Gaussian noise by learning textual prompts in a zero-shot or few-shot manner, achieving a delicate balance between accuracy and robustness, while minimizing the computational overhead. Moreover, PromptSmooth requires only a single model to handle multiple noise levels, which substantially reduces the computational cost compared to traditional methods that rely on training a separate model for each noise level. Comprehensive experiments based on three Med-VLMs and across six downstream datasets of various imaging modalities demonstrate the efficacy of PromptSmooth.*
>
<hr>

## Installation
* #### Setup conda environment
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
cd promptsmooth/

# Install requirements
pip install -r requirements.txt
```

## Pre-trained Weights
* #### Medical Vision language model pretrained weights		
You can Download PLIP wights from this [link](https://drive.google.com/file/d/1zwreSf0IYuTNJoLVymXJCEGWeEKUiWmi/view?usp=sharing). After downloading the zip file, have the folder in this structure `PromptSmooth/pretrained_weights/..`.
* #### Few-Shot PromptSmooth weights
The same folder (pretrained_weights) contains the few-shot PromptSmooth learnet weights, can be found in the following folder `PromptSmooth/pretrained_weights/fewshot_weights`.

## Data Preparation
In our paper we use a 500 images subset from each dataset to do the randomized smoothing certification test. You can either use the script `./sample_subset` to create your copy of the subset. However, we made it easier for you, you can just download the available subsets from this [link](https://drive.google.com/file/d/19v3p2b06o67TNENES_o1RgD9e3tT9Fk3/view?usp=sharing). The folder structure then should be `PromptSmooth/subsets/..`.

## Run Certification Scripts
We provide two python scripts `./certify_zeroshot_plip` and `./certify_promptsmooth_plip`

* #### To certify Zero-shot PLIP:
```
python ./certify_zeroshot_plip.py 
```
* #### To certify PLIP with PromptSmooth:

Zero-Shot PromptSmooth (noise-level 0.25)
```
python ./certify_promptsmooth_plip.py --n 500 --dataset kather --zeroshot True --sigma 0.25 --arch ViT-B/32 --outfile ./certification_output/PromptSmooth/PLIP --load ./PromptSmooth/pretrained_weights/fewshot_weights/kather_plip/FewshotPromptSmooth/vit_b32_ep50_16shots/nctx5_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50
```
Few-Shot PromptSmooth (noise-level 0.25)
```
python ./certify_promptsmooth_plip.py --n 500 --dataset kather --fewhsot True --sigma 0.25 --arch ViT-B/32 --outfile ./certification_output/PromptSmooth/PLIP --load ./PromptSmooth/pretrained_weights/fewshot_weights/kather_plip/FewshotPromptSmooth/vit_b32_ep50_16shots/nctx5_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50
```
For certification of PLIP with PromptSmooth, run the same script with the same arguments. However, set both `--zerohsot True` and `--fewhsot True`.

## Citation
If you find our work and this repository useful, please consider giving our repo a star and citing our paper as follows:
```bibtex
@article{hussein2024promptsmooth,
  title={PromptSmooth: Certifying Robustness of Medical Vision-Language Models via Prompt Learning},
  author={Hussein, Noor and Shamshad, Fahad and Naseer, Muzammal and Nandakumar, Karthik},
  journal={arXiv preprint arXiv:2408.16769},
  year={2024}
}
```

