import pandas as pd
from torch.utils.data import DataLoader


# Define a custom dataset
class image_title_dataset():
    def __init__(self, config, list_image_path, list_txt, class_embeddings, images):

        self.cfg = config
        self.image_path = list_image_path
        self.title = list_txt
        self.images = images
        self.class_embeddings = class_embeddings

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image_id = self.image_path[idx]
        image = self.images[self.image_path[idx]]

        if self.cfg.dataset == "ISIC_2018":
            if self.title[idx] == 0:
                title = self.class_embeddings['BKL'][0]
            elif self.title[idx] == 1:
                title = self.class_embeddings['NV'][0]
            elif self.title[idx] == 2:
                title = self.class_embeddings['DF'][0]
            elif self.title[idx] == 3:
                title = self.class_embeddings['MEL'][0]
            elif self.title[idx] == 4:
                title = self.class_embeddings['VASC'][0]
            elif self.title[idx] == 5:
                title = self.class_embeddings['BCC'][0]
            elif self.title[idx] == 6:
                title = self.class_embeddings['AKIEC'][0]
        elif self.cfg.dataset == "derm7pt":
            title = self.class_embeddings['Melanoma'] if self.title[idx] == 1 else self.class_embeddings['Nevus']

        label = self.title[idx]

        return {"image": image, "caption": title.squeeze(), "image_id": image_id, "label": label}


def create_train_dataloader(config, class_embeddings, images):
    """

    :param images: image embeddings
    :param class_embeddings: class label embeddings
    :param config: configuration parameters
    :return: train dataloader
    """

    if config.dataset == "derm7pt":
        input_data = "../data/derm7pt/derm7pt_train_seg.csv"

    elif config.dataset == "ISIC_2018":
        input_data = "../data/ISIC_2018/ISIC_2018_train.csv"

    else:
        raise Exception("Not a valid dataset. Available datasets: {derm7pt, ISIC_2018}")

    # Convert CSV into pandas DF
    data = pd.read_csv(input_data)

    list_image_path = []
    list_txt = []
    for img, label in zip(data['images'], data['labels']):
        img_path = img
        caption = label
        list_image_path.append(img_path)
        list_txt.append(caption)

    train_dataset = image_title_dataset(config, list_image_path, list_txt, class_embeddings, images)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    return train_dataloader


def create_val_dataloader(config, class_embeddings, images):
    """

    :param images: image embeddings
    :param class_embeddings: class label embeddings
    :param config: configuration parameters
    :return: val dataloader
    """

    if config.dataset == "derm7pt":
        input_data = "../data/derm7pt/derm7pt_validation_seg.csv"

    elif config.dataset == "ISIC_2018":
        input_data = "../data/ISIC_2018/ISIC_2018_validation.csv"

    else:
        raise Exception("Not a valid dataset. Available datasets: {derm7pt, ISIC_2018}")

    # Convert CSV into pandas DF
    data = pd.read_csv(input_data)

    list_image_path = []
    list_txt = []
    for img, label in zip(data['images'], data['labels']):
        img_path = img
        caption = label
        list_image_path.append(img_path)
        list_txt.append(caption)

    val_dataset = image_title_dataset(config, list_image_path, list_txt, class_embeddings, images)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return val_dataloader


def create_dataloader_inference(config, class_embeddings, images):
    if config.dataset == "derm7pt":
        input_data_train = "../data/derm7pt/derm7pt_train_seg.csv"
        input_data_val = "../data/derm7pt/derm7pt_validation_seg.csv"
        input_data_test = "../data/derm7pt/derm7pt_test_seg.csv"

    elif config.dataset == "ISIC_2018":
        input_data_train = "../data/ISIC_2018/ISIC_2018_train.csv"
        input_data_val = "../data/ISIC_2018/ISIC_2018_validation.csv"
        input_data_test = "../data/ISIC_2018/ISIC_2018_test.csv"
    else:
        raise Exception("Not a valid dataset. Available datasets: {derm7pt, ISIC_2018}")

    train = pd.read_csv(input_data_train)
    val = pd.read_csv(input_data_val)
    test = pd.read_csv(input_data_test)

    data = pd.concat([train, val, test], axis=0)

    list_image_path = []
    list_txt = []

    for img, label in zip(data['images'], data['labels']):
        img_path = img
        caption = label
        list_image_path.append(img_path)
        list_txt.append(caption)

    dataset = image_title_dataset(config=config, list_image_path=list_image_path, list_txt=list_txt, class_embeddings=class_embeddings, images=images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    return dataloader
