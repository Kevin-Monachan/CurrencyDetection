import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the model architecture exactly as it was when trained
class IndianCurrencyModel(nn.Module):
    def __init__(self, num_classes, model_name):
        super().__init__()
        # Use the same architecture from your training script
        if model_name == 'resnext50':
            self.model = models.resnext50_32x4d(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(num_features, 1024), nn.ReLU(),
                nn.BatchNorm1d(1024), nn.Dropout(0.4), nn.Linear(1024, 512),
                nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        # Add other model definitions here if needed (e.g., resnet50)
        else: # Default to resnet50
            self.model = models.resnet50(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(num_features, 1024), nn.ReLU(),
                nn.BatchNorm1d(1024), nn.Dropout(0.4), nn.Linear(1024, 512),
                nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)

def load_model_and_predict(image_path, model_path, model_name, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    
    # Instantiate the model and load weights
    model = IndianCurrencyModel(num_classes=num_classes, model_name=model_name).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # CRUCIAL FIX: Correct the key mismatch by removing the 'model.' prefix
    state_dict_from_checkpoint = checkpoint['model_state_dict']
    new_state_dict = {
        k.replace('model.', ''): v for k, v in state_dict_from_checkpoint.items()
    }
    model.model.load_state_dict(new_state_dict)
    model.eval()

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(outputs[0]).item()
        confidence = probabilities[predicted_idx].item()

    predicted_denomination = class_names[predicted_idx]
    
    return predicted_denomination, confidence