import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score


def test_model_3d(model, test_loader, class_names, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            
            # Directly pass the inputs to the model without unsqueeze
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)  # Compute probabilities
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    
    # Calculate AUC score for each class
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        true_labels = [1 if label == i else 0 for label in all_labels]
        predicted_probabilities = [prob[i] for prob in all_probabilities]
        auc = roc_auc_score(true_labels, predicted_probabilities)
        auc_scores[class_name] = auc
        print(f"AUC for class {class_name}: {auc:.4f}")
    
    return auc_scores


# Modify the testing loop to move inputs to the GPU and calculate AUC
def test_model_2d(model, test_loader, class_names, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            
            # Remove the extra dimension for 2D CNN
            inputs = inputs.squeeze(2)  # Shape becomes [batch_size, channels, height, width]
            
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)  # Compute probabilities
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))
    
    # Calculate AUC score for each class
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        true_labels = [1 if label == i else 0 for label in all_labels]
        predicted_probabilities = [prob[i] for prob in all_probabilities]
        auc = roc_auc_score(true_labels, predicted_probabilities)
        auc_scores[class_name] = auc
        print(f"AUC for class {class_name}: {auc:.4f}")
    
    return auc_scores