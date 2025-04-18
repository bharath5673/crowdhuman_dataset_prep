import kagglehub

# Download latest version
path = kagglehub.dataset_download("permanalwep/crowdhuman-crowd-detection")

print("Path to dataset files:", path)