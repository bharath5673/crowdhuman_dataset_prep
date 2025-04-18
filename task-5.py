from ultralytics import YOLO

# Load a model (use a pretrained model for training)
model = YOLO("yolov8n.pt")  # Load a pretrained model (recommended for training)

# Train the model on your custom dataset
results = model.train(
    data="./custom_dataset/crowdhuman_kaggle.yaml",  # Path to your custom dataset YAML file
    # data='cam18_22_.v2i.yolov8/data.yaml',
    epochs=100,               # Number of training epochs
    batch=8,                 # Batch size
    imgsz=640,                # Image size for training
    device="0",               # Use Cuda (set to "cpu" for CPU training)
    project="./runs/train",   # Directory where training results will be saved
    name="train_custom_dataset",  # Name of the training session for the saved results
    save=True,                # Save the model after training
)

# Optionally, load a partially trained model for further training
# model = YOLO("path/to/last.pt")  # Load a partially trained model
# results = model.train(resume=True)  # Resume training
