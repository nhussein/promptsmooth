import os
import random
from shutil import copyfile

def sample_images(input_dir, output_dir, total_samples, seed=3):
    # Set the seed for reproducibility
    random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all classes in the input directory
    classes = [class_name for class_name in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, class_name))]

    # List to keep track of selected images
    selected_images = set()

    # Continue until we reach the total_samples
    while total_samples > 0:
        # Iterate through each class
        for class_name in classes:
            class_path = os.path.join(input_dir, class_name)
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)

            # List all images in the current class
            images = [image for image in os.listdir(class_path) if image not in selected_images]

            # Check if there are images in the current class
            if images:
                # Randomly sample one image from the current class
                image = random.choice(images)

                # Copy the sampled image to the class-specific output directory
                src_path = os.path.join(class_path, image)
                dest_path = os.path.join(class_output_dir, image)
                copyfile(src_path, dest_path)

                # Add the selected image to the set
                selected_images.add(image)

                total_samples -= 1

                if total_samples == 0:
                    break

if __name__ == "__main__":
    input_directory = "./datasets/MedCLIP-datasets/COVID-19_Radiography_Dataset/images/test" #main full test set directory
    output_directory = "./datasets/subsets/COVID19_500subset/images/test" #location to save the 500 or 100 subset
    total_samples = 500
    seed = 3 #seed used in all experiments

    sample_images(input_directory, output_directory, total_samples, seed)