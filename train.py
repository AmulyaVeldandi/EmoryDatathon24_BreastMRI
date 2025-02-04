import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Training loop with visualization
# Modify the training loop to move inputs and labels to the GPU
# def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
#     train_losses = []
#     val_losses = []
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            
#             optimizer.zero_grad()
            
#             outputs = model(inputs)  # Directly pass the inputs (expected shape: [batch_size, 1, depth, H, W])
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         train_loss = running_loss / len(train_loader)
#         train_losses.append(train_loss)
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train Accuracy: {100 * correct / total}%")
        
#         # Validation step
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
#                 outputs = model(inputs)  # Directly pass the inputs
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         val_loss = val_loss / len(val_loader)
#         val_losses.append(val_loss)
#         print(f"Validation Loss: {val_loss}, Validation Accuracy: {100 * correct / total}%")
    
#     # Plot the training and validation loss curves
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('loss_curve_3d.png')  # Save the loss curves
#     plt.show()

# Training loop with visualization and model saving based on validation loss
def train_model_3d(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25, patience=20):
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            
            optimizer.zero_grad()
            
            outputs = model(inputs)  # Directly pass the inputs (expected shape: [batch_size, 1, depth, H, W])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")
        
        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                outputs = model(inputs)  # Directly pass the inputs
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}")
        
        # Save the model if the validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_3d.pt')
            print(f"Model saved at epoch {epoch+1} with validation loss {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model_3d.pt'))
    
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve_3d.png')  # Save the loss curves
    plt.show()


# Training loop with visualization
# Modify the training loop to move inputs and labels to the GPU
# Training loop with early stopping
# Training loop with early stopping and visualization
def train_model_2d_mip(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            inputs = inputs.squeeze(2)  # For 2D model
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")
        
        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                inputs = inputs.squeeze(2)  # For 2D model
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0  # Reset patience counter
            # Save the best model
            torch.save(model.state_dict(), 'best_model_2d_mip.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model_2d_mip.pt'))
    
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve_2d_mip.png')  # Save the loss curves
    plt.show()


def train_model_2d_mid_slice(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            inputs = inputs.squeeze(2)  # For 2D model
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")
        
        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                inputs = inputs.squeeze(2)  # For 2D model
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0  # Reset patience counter
            # Save the best model
            torch.save(model.state_dict(), 'best_model_2d_mid_slice.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_model_2d_mid_slice.pt'))
    
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve_2d_mid_slice.png')  # Save the loss curves
    plt.show()
