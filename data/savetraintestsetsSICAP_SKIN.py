import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sicapdataset_loader import SicapDataset  # Replace 'your_module' with the actual module where SicapDataset is defined
from skincancer_loader import SkinDataset


# Set the root directory where your data is located
root_directory = '/l/users/noor.hussein/datasets/PLIP-External-Preprocessed-Datasets/SkinCancer/data'
csv_file_skin= "tiles-v2.csv"

# # Create instances of SicapDataset for train and test
# train_dataset = SicapDataset(root_directory, "images", transform=transforms.ToTensor(), train=True)
# test_dataset = SicapDataset(root_directory, "images", transform=transforms.ToTensor(), train=False)

# Create instances of SkinDataset for train and test
train_dataset = SkinDataset(root_directory, csv_file_skin, train=True)
test_dataset = SkinDataset(root_directory, csv_file_skin, train=False)

# Create directories to save images
train_save_dir = os.path.join(root_directory, 'train')
test_save_dir = os.path.join(root_directory, 'test_2')

os.makedirs(train_save_dir, exist_ok=True)
os.makedirs(test_save_dir, exist_ok=True)

# # Save train images with class subdirectories
# for i in range(len(train_dataset)):
#     image, label = train_dataset[i]
#     class_subdir = os.path.join(train_save_dir, str(label))
#     os.makedirs(class_subdir, exist_ok=True)
#     # Convert tensor to PIL Image
#     image = transforms.ToPILImage()(image)
    
#     image_save_path = os.path.join(class_subdir, f"{os.path.basename(train_dataset.image_paths[i])}.png")
#     image.save(image_save_path)

# Save test images with class subdirectories
for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    class_subdir = os.path.join(test_save_dir, str(label))
    os.makedirs(class_subdir, exist_ok=True)
    # Convert tensor to PIL Image
    # image = transforms.ToPILImage()(image)
    
    image_save_path = os.path.join(class_subdir, f"{os.path.basename(test_dataset.image_paths[i])}.png")
    image.save(image_save_path)