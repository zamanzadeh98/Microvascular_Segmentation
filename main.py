import argparse
import torch
import json
import torchvision.transforms as transforms
from PIL import Image
from src.modelArchitecture import UNet



# Load your PyTorch model
def load_model(model_path):
     # Initialize
    model = UNet(n_class=1)
    finalModel.load_state_dict(torch.load(loadPath))
    model.eval()  # Set the model to evaluation mode
    return model

# Define a function to preprocess the input data (e.g., for image data)
def preprocess_data(ImgPath):
    
    with open(configs, "r") as file:
        mean_std = json.load(file)
    mean, std =  mean_std[0], mean_std[0]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Replace with your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean, std=std)  # Modify for your model
    ])
    
    img = Image.open(ImgPath).convert("RGB")
    # Resize the image and mask to the desired size
    img = img.resize((256, 256), Image.ANTIALIAS)
    
    image = transform(img).unsqueeze(0)  # Add batch dimension
    return image

# Define a function to make predictions
def get_predictions(model, data):
    with torch.no_grad():
        predictions = model(data)
    return predictions

def main(args):
    # Load the PyTorch model
    model_path = args.model_path
    model = load_model(model_path)

    # Load and preprocess the data from the specified path
    data_path = args.data_path
    data = preprocess_data(data_path)

    # Get results
    results = get_predictions(model, data)

    # Display or save results as needed
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Deep Learning Inference Script')
    parser.add_argument('model_path', type=str, help='Path to the PyTorch model')
    parser.add_argument('data_path', type=str, help='Path to the data for inference')
    args = parser.parse_args()
    
    

    main(args)
