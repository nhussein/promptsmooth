import torch


class CFG:
    # ------------------------ Preprocessing data ------------------------------ #
    # MODELS = [ViT-B/32, ViT-B/16, RN50, RN101, ViT-L/14, RN50x16]
    clip_model = 'ViT-B/16'
    CLASS_LABELS_PROMPTS = {
        "Melanoma": ["This is dermatoscopy of melanoma"],
        "Nevus": ["This is dermatoscopy of nevus"]
    }
    CLASS_LABELS_PROMPTS_ISIC_2018 = {
        "BKL": ["This is dermatoscopy of pigmented benign keratosis",
                'This is dermoscopy of pigmented benign keratosis'],
        "NV": ["This is dermatoscopy of nevus", 'This is dermoscopy of nevus'],
        "DF": ['This is dermatoscopy of dermatofibroma', 'This is dermoscopy of dermatofibroma'],
        "MEL": ['This is dermatoscopy of melanoma', 'This is dermoscopy of melanoma'],
        "VASC": ['This is dermatoscopy of vascular lesion', 'This is dermoscopy of vascular lesion'],
        "BCC": ['This is dermatoscopy of basal cell carcinoma', 'This is dermoscopy of basal cell carcinoma'],
        "AKIEC": ['This is dermatoscopy of actinic keratosis', 'This is dermoscopy of actinic keratosis']
    }
    CONCEPT_PROMPTS = {
        "Asymmetry": ["This is dermatoscopy of an asymmetry"],
        "Irregular": ["This is dermatoscopy of irregular"],
        "Erosion": ["This is dermatoscopy of erosion"],
        "Black": ["This is dermatoscopy of black"],
        "Blue": ["This is dermatoscopy of blue"],
        "White": ["This is dermatoscopy of white"],
        "Brown": ["This is dermatoscopy of brown"],
        "Multiple Colors": ["This is dermatoscopy of multiple colors"],
        "Tiny": ["This is dermatoscopy of tiny"],
        "Regular": ["This is dermatoscopy of a regular"]
    }
    DETAILED_CONCEPT_PROMPTS = {
        "Asymmetry": ["This is dermatoscopy of an asymmetric shape with one half not mirroring the other half",
                      "This is dermatoscopy of asymmetrical distribution of pigmentation",
                      "This is dermatoscopy of irregular and non-symmetrical borders",
                      "This is dermatoscopy of significant asymmetry",
                      "This is dermatoscopy of asymmetry in the form of dissimilar features on opposite sides of the lesion"],
        "Irregular": ["This is dermatoscopy of irregular shapes or outlines",
                      "This is dermatoscopy of irregular distribution of pigmentation",
                      "This is dermatoscopy of poorly defined borders",
                      "This is dermatoscopy of irregular and atypical patterns",
                      "This is dermatoscopy of irregular features in the form of non-uniform characteristics"],
        "Erosion": ["This is dermatoscopy of surface ulceration or erosion",
                    "This is dermatoscopy of erosion as a crusted area on the skin",
                    "This is dermatoscopy of ulcerated appearance",
                    "This is dermatoscopy of erosion with exposed underlying tissue",
                    "This is dermatoscopy of erosion in the form of disrupted or absent epidermal structures"],
        "Black": ["This is dermatoscopy of dark or black pigmentation", "This is dermatoscopy of black coloration",
                  "This is dermatoscopy of dark brown to black areas",
                  "This is dermatoscopy of black structures or pigmentation",
                  "This is dermatoscopy of black coloration in the form of concentrated dark areas in the lesion"],
        "Blue": ["This is dermatoscopy of blue or blue-gray coloration", "This is dermatoscopy of blue coloration",
                 "This is dermatoscopy of bluish patches or areas of discoloration",
                 "This is dermatoscopy of blue structures or pigmentation",
                 "This is dermatoscopy of blue coloration in the form of bluish hues or tones in the lesion"],
        "White": ["This is dermatoscopy of white or hypopigmented coloration",
                  "This is dermatoscopy of white coloration",
                  "This is dermatoscopy of pale or depigmented patches or areas",
                  "This is dermatoscopy of white structures or depigmentation",
                  "This is dermatoscopy of white coloration in the form of reduced pigmentation in the lesion"],
        "Brown": ["This is dermatoscopy of brown or dark-brown pigmentation",
                  "This is dermatoscopy of brown coloration",
                  "This is dermatoscopy of brown patches or areas of discoloration",
                  "This is dermatoscopy of brown structures or pigmentation",
                  "This is dermatoscopy od brown coloration in the form of various shades of brown in the lesion"],
        "Multiple Colors": ["This is dermatoscopy of a combination of different colors",
                            "This is dermatoscopy of multiple colorations with a varied and complex appearance",
                            "This is dermatoscopy of a mix of different hues",
                            "This is dermatoscopy of diverse colors and pigmentation",
                            "This is dermatoscopy of multiple coloration in the form of different colored areas within the lesion"],
        "Tiny": ["This is dermatoscopy of small and minute structures or shapes",
                 "This is dermatoscopy of tiny shapes characterized by their small size",
                 "This is dermatoscopy of minuscule or small-sized patterns",
                 "This is dermatoscopy of tiny structures or shapes",
                 "This is dermatoscopy of tiny shape in the form of small and discrete features within the lesion"],
        "Regular": ["This is dermatoscopy of a regular and symmetrical pattern",
                    "This is dermatoscopy of regular and evenly spaced structures",
                    "This is dermatoscopy of uniform arrangement of patterns",
                    "This is dermatoscopy of regular pattern in the form of symmetrical and well-defined features within the lesion"]
    }
    REFERENCE_CONCEPT_PROMPTS = ["This is dermatoscopy"]
    REFERENCE_CONCEPT_PROMPTS_ISIC_2018 = ["This is dermatoscopy", "This is dermoscopy"]
    dataset = 'derm7pt' # {'derm7pt', 'ISIC_2018'}
    # -------------------------------------------------------------------------- #

    seed = 0
    batch_size = 32
    num_workers = 4
    head_lr = 1e-5
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_embedding = 512 #1024
    text_embedding = 512 #1024
    max_length = 200
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 512 #1024
    dropout = 0.2

    # model path best model
    model_name = str(clip_model).replace("/", "-")
    path_to_model = f"output/CLIP_{dataset}_{model_name}_{seed}_Segmented.pt"