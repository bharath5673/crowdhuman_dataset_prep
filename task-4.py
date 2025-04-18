import yaml
import os

# Paths to your dataset
output_base_path = "custom_dataset"  # Change this to your actual output dataset path

# Absolute path to the custom dataset base folder
absolute_base_path = os.path.abspath(output_base_path)

# Classes
classes = ['person', 'head']  # List of class names for your dataset

# Create the YAML structure
data_yaml = {
    'train': os.path.join(absolute_base_path,'train', 'images'),  # Absolute path to training images
    'val': os.path.join(absolute_base_path, 'val', 'images'),      # Absolute path to validation images
    'test': os.path.join(absolute_base_path,'test', 'images'),    # Optional: Absolute path to test images
    'nc': len(classes),  # Number of classes
    'names': classes  # Class names
}

# Path to save the YAML file
yaml_path = os.path.join(absolute_base_path, 'crowdhuman_kaggle.yaml')

# Write the YAML file
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(data_yaml, yaml_file, default_flow_style=False)

print(f"crowdhuman_kaggle.yaml created at {yaml_path}")
