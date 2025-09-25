import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import warnings

warnings.filterwarnings('ignore')

# This helper function must be consistent with the one used for training
def get_model_and_transforms(model_name, num_classes, pretrained=False):
    """
    Creates the model and defines the appropriate transforms based on the model name.
    """
    img_size = 224
    if model_name == 'resnext50':
        model = models.resnext50_32x4d(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else: # Default to resnet50
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, val_transform

def load_model_and_predict(image_path, model_path, model_name):
    """
    Loads the trained model from a checkpoint and performs a prediction on a single image.
    The class names are loaded from the checkpoint, ensuring consistency.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint to get model details and class names
    checkpoint = torch.load(model_path, map_location=device)
    
    # CRITICAL: Get the number of classes and the class names from the checkpoint
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Recreate the model with the correct number of classes
    model, val_transform = get_model_and_transforms(model_name, num_classes, pretrained=False)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Perform prediction
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = val_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(outputs[0]).item()
            confidence = probabilities[predicted_idx].item()
        
        predicted_denomination = class_names[predicted_idx]
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

    return predicted_denomination, confidence
