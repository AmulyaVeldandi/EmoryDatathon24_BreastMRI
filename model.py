import torch
import torch.nn as nn
import torch.nn.functional as F

class Deeper3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Deeper3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        
        self.pool = nn.MaxPool3d(2, 2)
        self.dropout = nn.Dropout3d(0.5)

        # Calculate the flattened size
        self.flattened_size = self._get_flattened_size()
        
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
    
    def _get_flattened_size(self):
        # Create a dummy input tensor with the same dimensions as your input data
        dummy_input = torch.zeros(1, 1, 51, 128, 128)
        
        # Pass the dummy input through the conv and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Calculate the flattened size
        flattened_size = x.numel()  # Total number of elements in the flattened tensor
        return flattened_size

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Deeper2DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Deeper2DCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Reduced from 32 to 16 filters
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Reduced from 64 to 32 filters
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Reduced from 128 to 64 filters
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Reduced from 256 to 128 filters
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Reduced from 512 to 256 filters
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)

        # Calculate the flattened size
        self.flattened_size = self._get_flattened_size()
        
        self.fc1 = nn.Linear(self.flattened_size, 512)  # Reduced from 1024 to 512 neurons
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)  # Reduced from 256 to 128 neurons
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
    
    def _get_flattened_size(self):
        # Create a dummy input tensor with the same dimensions as your input data
        dummy_input = torch.zeros(1, 1, 128, 128)  # Adjust the size according to your data
        
        # Pass the dummy input through the conv and pooling layers
        x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        # Calculate the flattened size
        flattened_size = x.numel()  # Total number of elements in the flattened tensor
        return flattened_size

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# class Simpler2DCNN(nn.Module):
#     def __init__(self, num_classes=4):
#         super(Simpler2DCNN, self).__init__()
        
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
        
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
        
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
        
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout2d(0.5)

#         # Calculate the flattened size
#         self.flattened_size = self._get_flattened_size()
        
#         self.fc1 = nn.Linear(self.flattened_size, 256)  # Reduced fully connected layer
#         self.fc_bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc_bn2 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(64, num_classes)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
#         x = x.view(-1, self.num_flat_features(x))
        
#         x = F.relu(self.fc_bn1(self.fc1(x)))
#         x = self.dropout(x)
        
#         x = F.relu(self.fc_bn2(self.fc2(x)))
#         x = self.dropout(x)
        
#         x = self.fc3(x)
#         return x
    
#     def _get_flattened_size(self):
#         dummy_input = torch.zeros(1, 1, 128, 128)
#         x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         flattened_size = x.numel()
#         return flattened_size

#     def num_flat_features(self, x):
#         size = x.size()[1:]  
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


# class Simpler2DCNN(nn.Module):
#     def __init__(self, num_classes=4):
#         super(Simpler2DCNN, self).__init__()
        
#         # Reduce the number of convolutional layers and filters
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # From 16 to 8 filters
#         self.bn1 = nn.BatchNorm2d(8)
        
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)  # From 32 to 16 filters
#         self.bn2 = nn.BatchNorm2d(16)
        
#         self.pool = nn.MaxPool2d(2, 2)
        
#         # Reduce the number of neurons in the fully connected layers
#         self.flattened_size = self._get_flattened_size()
#         self.fc1 = nn.Linear(self.flattened_size, 64)  # From 512 to 64 neurons
#         self.fc_bn1 = nn.BatchNorm1d(64)
#         self.fc2 = nn.Linear(64, num_classes)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
#         x = x.view(-1, self.num_flat_features(x))
        
#         x = F.relu(self.fc_bn1(self.fc1(x)))
#         x = self.fc2(x)
#         return x
    
#     def _get_flattened_size(self):
#         # Create a dummy input tensor with the same dimensions as your input data
#         dummy_input = torch.zeros(1, 1, 128, 128)  # Adjust the size according to your data
        
#         # Pass the dummy input through the conv and pooling layers
#         x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
#         # Calculate the flattened size
#         flattened_size = x.numel()  # Total number of elements in the flattened tensor
#         return flattened_size

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # All dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features