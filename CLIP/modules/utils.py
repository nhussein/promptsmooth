import random
import numpy as np
import logging
from pathlib import Path
import glob
from PIL import Image
from tqdm import tqdm
import os
from .model import CLIPModel

import torch
import torch.nn as nn
import clip


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    return logger


def save_model(config, model):
    if not os.path.exists("output"):
        os.mkdir("output")

    model_name = str(config.clip_model).replace("/","-")
    torch.save(model.state_dict(), f"output/CLIP_{config.dataset}_{model_name}_{config.seed}_Segmented.pt")


def save_embeddings(config, embeddings, type, type_emb, disease):
    model_name = str(config.clip_model).replace("/", "-")

    if not os.path.exists("output"):
        os.mkdir("output")

    if not os.path.exists(f"output/{model_name}"):
        os.mkdir(f"output/{model_name}")

    if not os.path.exists(f"output/{model_name}/{type}"):
        os.mkdir(f"output/{model_name}/{type}")

    np.save(f"output/{model_name}/{type}/{type_emb}_{disease}_{config.dataset}_{model_name}-Fine-Tuned_{config.seed}.npy", embeddings)

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set seed for general python operations
    random.seed(seed)
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    # Set the seed for Numpy operations
    np.random.seed(seed)
    # To use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # TO use deterministic benchmark
    torch.backends.cudnn.benchmark = False


def get_image_embeddings(config, test_dataloader, model_path):
    model = CLIPModel().to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    test_image_embeddings = dict()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            image_embeddings = model.image_projection(batch["image"].type(torch.float32).to(config.device))
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            test_image_embeddings[batch["image_id"][0]] = image_embeddings.detach().cpu().squeeze().numpy()

    return model, test_image_embeddings


def get_text_embeddings(config, caption, model_path):
    model = CLIPModel().to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    test_text_embeddings = []
    with torch.no_grad():
        text_embeddings = model.text_projection(torch.from_numpy(caption).type(torch.float32).to(config.device))
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        test_text_embeddings.append(text_embeddings)

    return model, torch.cat(test_text_embeddings).detach().cpu().numpy()


def extract_custom_text_embeddings(config, model, type_of_text):
    """

    :param config: configuration parameters
    :param model: loaded CLIP model
    :return: dictionary containing extracted text features per disease label

    {
        "disease_label": [feature]
    }
    """

    log = get_logger(__name__)  # init logger

    # Extract text features for each concept and save it to a numpy array
    log.info("Extracting text embeddings...")
    class_label_embeddings = dict()
    for disease_label in type_of_text.keys():
        text = clip.tokenize(type_of_text[disease_label]).to(config.device)

        with torch.no_grad():
            text_features = model.encode_text(text)

        text_features /= text_features.norm(dim=-1, keepdim=True)

        class_label_embeddings[disease_label] = text_features.detach().cpu().numpy()

        # np.save(f"extracted_text_embeddings/class_label_embeddings_{disease_label}_{MODEL}.npy", text_features.detach().cpu().numpy())

    return class_label_embeddings


def extract_text_embeddings(config, model):
    """

    :param config: configuration parameters
    :param model: loaded CLIP model
    :return: dictionary containing extracted text features per disease label

    {
        "disease_label": [feature]
    }
    """

    log = get_logger(__name__)  # init logger

    # Extract text features for each concept and save it to a numpy array
    log.info("Extracting text embeddings...")
    class_label_embeddings = dict()

    if config.dataset == "derm7pt":
        for disease_label in config.CLASS_LABELS_PROMPTS.keys():
            text = clip.tokenize(config.CLASS_LABELS_PROMPTS[disease_label]).to(config.device)

            with torch.no_grad():
                text_features = model.encode_text(text)

            text_features /= text_features.norm(dim=-1, keepdim=True)

            class_label_embeddings[disease_label] = text_features.detach().cpu().numpy()

            # np.save(f"extracted_text_embeddings/class_label_embeddings_{disease_label}_{MODEL}.npy", text_features.detach().cpu().numpy())
    else:
        for disease_label in config.CLASS_LABELS_PROMPTS_ISIC_2018.keys():
            text = clip.tokenize(config.CLASS_LABELS_PROMPTS_ISIC_2018[disease_label]).to(config.device)

            with torch.no_grad():
                text_features = model.encode_text(text)

            text_features /= text_features.norm(dim=-1, keepdim=True)

            class_label_embeddings[disease_label] = text_features.detach().cpu().numpy()

    return class_label_embeddings


def extract_reference_embeddings(config, model):
    """

    :param config: configuration parameters
    :param model: loaded CLIP model
    :return: numpy array containing extracted reference embeddings
    """

    log = get_logger(__name__)  # init logger

    # Extract text features for each concept and save it to a numpy array
    log.info("Extracting reference prompt embeddings...")

    if config.dataset == "derm7pt":
        text = clip.tokenize(config.REFERENCE_CONCEPT_PROMPTS).to(config.device)
        with torch.no_grad():
            text_features = model.encode_text(text)

        text_features /= text_features.norm(dim=-1, keepdim=True)
    else:
        text = clip.tokenize(config.REFERENCE_CONCEPT_PROMPTS_ISIC_2018).to(config.device)
        with torch.no_grad():
            text_features = model.encode_text(text)

        text_features /= text_features.norm(dim=-1, keepdim=True)

    # np.save(f"concept_embeddings/reference_embeddings_{MODEL}.npy", text_features.detach().cpu().numpy())
    return text_features.detach().cpu().numpy()


def extract_image_embeddings(config, model, preprocess):
    """

    :param config: configuration parameters
    :param model: loaded CLIP model
    :return: dictionary containing the image embeddings

    {
        "img_id": [feature]
    }
    """
    log = get_logger(__name__)  # init logger

    log.info("Extracting image embeddings...")

    if config.dataset == "derm7pt":
        # images = glob.glob(
        #     "/l/users/noor.hussein/datasets/derm7pt/images/*.png")
        images = glob.glob(os.path.join("/l/users/noor.hussein/datasets/derm7pt/images/", '**', '*.[jJ][pP][gG]'), recursive=True)
    elif config.dataset == "ISIC_2018":
        images = glob.glob("/home/cristianopatricio/Desktop/PhD/Datasets/Skin/HAM10000/Images_Segmented/*.png")
    else:
        raise Exception('Not a valid dataset. Available datasets: {derm7pt, ISIC_2018}')

    # Define a dictionary to save image embeddings
    img_embeddings = dict()
    # Iterate over all images
    for im in tqdm(images):
        # Load and Preprocess input
        image = preprocess(Image.open(im)).unsqueeze(0).to(config.device)

        # Get features
        with torch.no_grad():
            image_features = model.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Save image embedding to dictionary
        img_name = Path(im).stem
        img_embeddings[img_name] = image_features.cpu().detach().numpy().flatten()

    return img_embeddings
