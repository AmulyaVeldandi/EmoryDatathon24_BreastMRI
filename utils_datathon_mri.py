import os
import gdcm
import imageio
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from random import sample
import pydicom as pydicom
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import rotate
from IPython.display import HTML, Video
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def load_dicom_image(filepath):
    dicom_file = pydicom.dcmread(filepath)
    px_arr = apply_voi_lut(apply_modality_lut(dicom_file.pixel_array, dicom_file),dicom_file)
    return px_arr, dicom_file

# Function to load a DICOM image
def load_dicom_image_px_arr(dicom_path):
    dicom_file = pydicom.dcmread(dicom_path)
    mri_slice = dicom_file.pixel_array
    return mri_slice

def save_image_as_png(image, dicom_file, output_path):
    # Convert pixel array to PNG as a 16-bit greyscale
    image_to_save = image.astype(np.double)
    
    # Rescale grey scale between 0-65535
    image_to_save = (np.maximum(image_to_save, 0) / image_to_save.max()) * 65535.0
    
    # Convert to uint16
    image_to_save = np.uint16(image_to_save)
    
    # Save the image as PNG
    imsave(output_path, image_to_save, check_contrast=False)

def process_study(group, output_root):
    # Group by SeriesNumber and sort by InstanceNumber within each series
    series_grouped = group.groupby('SeriesNumber')
    
    pre_contrast_slices = None
    post_contrast_slices = []

    for series_number, series_group in series_grouped:
        series_group = series_group.sort_values(by='InstanceNumber')
        
        if pre_contrast_slices is None:
            # Load the pre-contrast series (first encountered series)
            pre_contrast_slices = [load_dicom_image(row['sampled_anon_dicom_path_pacs'])[0] for _, row in series_group.iterrows()]
        else:
            # Load each post-contrast series
            post_contrast_slices.append([
                {
                    'file_path': row['sampled_anon_dicom_path_pacs'],
                    'image': load_dicom_image(row['sampled_anon_dicom_path_pacs'])[0],
                    'dicom_file': load_dicom_image(row['sampled_anon_dicom_path_pacs'])[1]
                }
                for _, row in series_group.iterrows()
            ])
    
    # Perform subtraction between pre-contrast and each post-contrast series
    if pre_contrast_slices is not None:
        for post_series in post_contrast_slices:
            for i, post_slice in enumerate(post_series):
                subtracted_image = post_slice['image'] - pre_contrast_slices[i]
                
                # Create corresponding output path
                original_path = post_slice['file_path']
                # output_path = os.path.join(output_root, original_path[53:]) # sample
                output_path = os.path.join(output_root, original_path[53:])
                output_path = output_path.replace('.dcm', '.png')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the image as PNG
                save_image_as_png(subtracted_image, post_slice['dicom_file'], output_path)

def process_and_save_subtracted_series(df, output_root, max_workers=16):
    # Sort the DataFrame by study_instance_uid and series_number first
    df = df.sort_values(by=['StudyInstanceUID_anon', 'SeriesNumber', 'InstanceNumber'])
    
    # Group by study_instance_uid
    grouped = df.groupby(['StudyInstanceUID_anon'])

    # Parallel processing of the groups
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_study, [group for _, group in grouped], [output_root] * len(grouped)), total=len(grouped)))


def load_dicom_series(df, series_number):
    # Filter the DataFrame for the specific series number and sort by InstanceNumber
    series_df = df[df['SeriesInstanceUID_anon'] == series_number].sort_values(by='InstanceNumber')
    images = []
    for _, row in series_df.iterrows():
        dicom_path = row['sampled_anon_dicom_path_pacs']
        dicom_file = pydicom.dcmread(dicom_path)
        images.append(dicom_file.pixel_array)
    return np.array(images), series_df

def load_png_series(df, series_number, output_root):
    # Filter the DataFrame for the specific series number and sort by InstanceNumber
    series_df = df[df['SeriesInstanceUID_anon'] == series_number].sort_values(by='InstanceNumber')
    images = []
    for _, row in series_df.iterrows():
        dicom_path = row['sampled_anon_dicom_path_pacs']
        png_path = os.path.join(output_root, dicom_path[53:]).replace('.dcm', '.png')
        images.append(imread(png_path))
    return np.array(images)

def animate_images(images, title):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.imshow(images[frame], cmap='gray')
        ax.set_title(f'{title} - Slice {frame+1}')
    
    anim = FuncAnimation(fig, update, frames=len(images), interval=100)
    plt.close(fig)  # Prevent double display of static figure
    return HTML(anim.to_jshtml())  # Display the animation inline in the notebook

# Function to read a DICOM file
def read_dicom(dicom_file):
    return pydicom.dcmread(dicom_file)

# Function to decompress pixel data if compressed
def decompress_pixel_data(ds):
    if 'TransferSyntaxUID' in ds.file_meta and ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress(handler_name='gdcm')
    return ds

# Function to adjust the orientation of the DICOM image
def adjust_orientation(ds):
    pixel_array = ds.pixel_array

    # Flip the image horizontally and vertically
    adjusted_image = np.flipud(np.fliplr(pixel_array))

    # Update the DICOM dataset with the adjusted image
    ds.PixelData = adjusted_image.tobytes()
    ds.Rows, ds.Columns = adjusted_image.shape

    # Update the ImageOrientationPatient tag
    orientation = ds.ImageOrientationPatient
    # Flip horizontally and vertically
    orientation[0] = -orientation[0]  # Flip first row direction cosine X
    orientation[1] = -orientation[1]  # Flip first row direction cosine Y
    orientation[3] = -orientation[3]  # Flip first column direction cosine X
    orientation[4] = -orientation[4]  # Flip first column direction cosine Y
    ds.ImageOrientationPatient = orientation

    return ds

# Function to process each DICOM file
def process_dicom_file(dicom_file):
    try:
        # Read the DICOM file
        ds = read_dicom(dicom_file)

        # Decompress pixel data if needed
        ds = decompress_pixel_data(ds)

        # Adjust the orientation
        adjusted_ds = adjust_orientation(ds)

        # Recompress pixel data if original was compressed
        # if 'TransferSyntaxUID' in ds.file_meta and ds.file_meta.TransferSyntaxUID.is_compressed:
            # frames = generate_pixel_data_frame(ds.PixelData, ds.Rows, ds.Columns, 1, len(ds.PixelData))
            # ds.PixelData = encapsulate(frames)

        # Save the adjusted DICOM file back to the original file
        adjusted_ds.save_as(dicom_file)

        return None  # Return None if the process is successful
    except Exception as e:
        return f"Error processing {dicom_file}: {e}"

# # Define train transform function that includes a consistent random rotation
# def apply_transforms_to_stack(image_stack, rotation_angle=None, is_train=False):
#     transformed_stack = []
#     # Define the transformations without the rotation for easier reuse
#     resize_transform = transforms.Resize((128, 128))
#     normalize_transform = transforms.Normalize(mean=[0], std=[1])
    
#     for img in image_stack:
#         img = resize_transform(img)
        
#         if is_train and rotation_angle is not None:
#             img = transforms.functional.rotate(img, rotation_angle)  # Apply consistent rotation
        
#         img = transforms.ToTensor()(img)
#         # img = normalize_transform(img)
#         # img = (img - (img.mean()))/(img.std())
#         img = (img - img.min()) / (img.max() - img.min())
#         transformed_stack.append(img)
    
#     return torch.stack(transformed_stack)  # Shape: [num_slices, C, H, W]

# Define train transform function that includes a consistent random rotation
def apply_transforms_to_stack(image_stack, rotation_angle=None, is_train=False):
    transformed_stack = []
    resize_transform = transforms.Resize((128, 128))
    
    # Convert to Tensor and resize each image
    tensor_stack = torch.stack([resize_transform(transforms.ToTensor()(img)) for img in image_stack])
    
    # Calculate the mean and std across the entire stack
    stack_mean = tensor_stack.mean()
    stack_std = tensor_stack.std()
    
    for img in tensor_stack:
        if is_train and rotation_angle is not None:
            img = transforms.functional.rotate(img, rotation_angle)  # Apply consistent rotation
        
        # Normalize across the entire stack
        img = (img - stack_mean) / stack_std
        transformed_stack.append(img)
    
    return torch.stack(transformed_stack)  # Shape: [num_slices, C, H, W]

def load_image(file_path):
    # Load the image
    img = Image.open(file_path).convert('L')  # Convert to grayscale if needed
    return img

def process_and_save_study(df, study_instance, save_dir, is_train=False):
    study_df = df[df['StudyInstanceUID_anon'] == study_instance]
    label = study_df['BPE_type'].iloc[0]  # Assuming the label is the same for the entire study
    
    all_series = []
    
    for series_instance in study_df['SeriesInstanceUID_anon'].unique():
        series_df = study_df[study_df['SeriesInstanceUID_anon'] == series_instance]
        series_df = series_df.sort_values(by='InstanceNumber').reset_index(drop=True)  # Sort by InstanceNumber
        
        image_stack = [load_image(fp) for fp in series_df['FilePath']]
        
        # Generate a random rotation angle for the entire series if training
        rotation_angle = transforms.RandomRotation.get_params((-10, 10)) if is_train else None
        
        # Apply the transformations to the entire stack with consistent rotation if is_train
        transformed_stack = apply_transforms_to_stack(image_stack, rotation_angle, is_train)
        
        all_series.append(transformed_stack)
    
    # Combine the transformed series in the correct order
    combined_series = torch.cat(all_series, dim=0)  # Shape: [4*num_slices, C, H, W]
    
    # Save the combined tensor in the appropriate directory based on label
    save_path = os.path.join(save_dir, label, f"{study_instance}.pt")
    torch.save(combined_series, save_path)
    # print(f"Saved transformed data for {study_instance} at {save_path}")

# Select slices per series using Pandas groupby and apply
def select_slices_per_series(df):
    def select_middle_slices(group):
        group = group.sort_values(by='InstanceNumber').reset_index(drop=True)
        num_slices = len(group)
        middle_idx = num_slices // 2
        start_idx = max(0, middle_idx - 25)
        end_idx = min(num_slices, middle_idx + 26)
        return group.iloc[start_idx:end_idx]
    
    filtered_df = df.groupby(['StudyInstanceUID_anon', 'SeriesInstanceUID_anon']).apply(select_middle_slices)
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df

# Combine series within each study
def combine_series(filtered_df):
    combined_df = filtered_df.sort_values(by=['StudyInstanceUID_anon', 'SeriesInstanceUID_anon', 'InstanceNumber'])
    return combined_df

# Function to load a .pth file and convert it to a NumPy array
def load_prcsd_pth_series(pth_file_path):
    tensor = torch.load(pth_file_path)  # Load the .pth file
    images = tensor.numpy()  # Convert to NumPy array
    # If the tensor is in shape (N, C, H, W) where N is the number of slices,
    # and C is the number of channels, we need to handle it correctly.
    if images.shape[1] == 1:  # If it's grayscale, remove the channel dimensions
        images = images.squeeze(1)  # Convert from (N, 1, H, W) to (N, H, W)
    return images

# Function to animate a series of images
def animate_prcd_images(images, title, vmin=0, vmax=1):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.imshow(images[frame], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title} - Slice {frame+1}')
    
    anim = FuncAnimation(fig, update, frames=len(images), interval=100)
    plt.close(fig)  # Prevent double display of static figure
    return HTML(anim.to_jshtml())  # Display the animation inline in the notebook

# Function to animate a series of images
def animate_images_3d(images, title, vmin=None, vmax=None):
    # Set vmin and vmax if not provided
    if vmin is None:
        vmin = images.min()
    if vmax is None:
        vmax = images.max()
    
    fig, ax = plt.subplots()
    ax.axis('off')
    
    def update(frame):
        ax.clear()
        ax.imshow(images[frame], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{title} - Slice {frame+1}')
    
    anim = FuncAnimation(fig, update, frames=len(images), interval=100)
    plt.close(fig)  # Prevent double display of static figure
    return HTML(anim.to_jshtml())  # Display the animation inline in the notebook


# Function to visualize and predict using .pt file
def visualize_and_predict_pt_3d(file_path, loaded_model, combined_df, class_names, device):
    # Load the image from the .pt file
    image = torch.load(file_path)
    
    # Extract the filename without the extension to query the dataframe
    file_name = file_path.split('/')[-1].replace('.pt', '')
    
    # Query the label from the dataframe using the file name
    true_label_name = combined_df[combined_df.StudyInstanceUID_anon == file_name].BPE_type.unique()[0]
    
    image = image.permute(1, 0, 2, 3).unsqueeze(0)
    # Model prediction
    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(image.to(device))
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
        predicted_label_name = class_names[predicted_label]

    # Display predicted label
    print(f"True Label: {true_label_name}, Predicted Label: {predicted_label_name}")

    # Display the animation with both true and predicted labels
    display(animate_images_3d(image.squeeze(), f"True: {true_label_name}, Predicted: {predicted_label_name}"))


# Function to visualize a single 2D image and predict the label
def visualize_and_predict_2d_mip(file_path, model, combined_df, class_names, device):
    # Load the image from the .pt file
    image = torch.load(file_path)  # Assuming the image is in 3D [depth, H, W]
    
    # Extract the filename without the extension to query the dataframe
    file_name = file_path.split('/')[-1].replace('.pt', '')
    
    # Query the label from the dataframe using the file name
    true_label_name = combined_df[combined_df.StudyInstanceUID_anon == file_name].BPE_type.unique()[0]
    
    # Apply Maximum Intensity Projection (MIP) to get a 2D image
    mip_image = torch.max(image, dim=0)[0]  # MIP along the depth dimension

    # Prepare the image for the 2D model
    mip_image = mip_image.unsqueeze(0)  # Add channel dimension [1, H, W]

    # Remove the unnecessary extra dimension to make it [1, 128, 128]
    # mip_image = mip_image.unsqueeze(0)  # Add batch dimension [1, 1, H, W]
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(mip_image.to(device))
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
        predicted_label_name = class_names[predicted_label]

    # Display predicted label
    print(f"True Label: {true_label_name}, Predicted Label: {predicted_label_name}")

    # Display the image with both true and predicted labels
    plt.figure(figsize=(4, 4))
    plt.imshow(mip_image.squeeze().cpu(), cmap='gray')
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.axis('off')
    plt.show()


def visualize_mip_from_dataset(dataset, index):
    """
    Visualizes the MIP image from the dataset at the specified index.
    
    Args:
        dataset (MRIDataset2D): The dataset instance.
        index (int): The index of the item to visualize.
    """
    mip_image, label = dataset[index]  # Retrieve the MIP image and label

    # Convert the tensor to a numpy array and squeeze to remove the channel dimension
    mip_image_np = mip_image.squeeze().cpu().numpy()
    
    # Display the MIP image
    plt.figure(figsize=(5, 5))
    plt.imshow(mip_image_np, cmap='gray')
    plt.title(f'MIP Image - Label: {label}')
    plt.axis('off')
    plt.show()

def visualize_middle_slice(dataset, index):
    """
    Visualizes the middle slice from the dataset at the specified index.
    
    Args:
        dataset (MRIDataset2D): The dataset instance.
        index (int): The index of the item to visualize.
    """
    middle_slice, label = dataset[index]  # Retrieve the middle slice and label

    # Convert the tensor to a numpy array and squeeze to remove the channel dimension
    middle_slice_np = middle_slice.squeeze().cpu().numpy()
    
    # Display the middle slice
    plt.figure(figsize=(5, 5))
    plt.imshow(middle_slice_np, cmap='gray')
    plt.title(f'Middle Slice - Label: {label}')
    plt.axis('off')
    plt.show()

# Function to visualize a single 2D image and predict the label
def visualize_and_predict_2d_mid_slice(file_path, model, combined_df, class_names, device):
    # Load the image from the .pt file
    image = torch.load(file_path)  # Assuming the image is in 3D [depth, H, W]
    
    # Extract the filename without the extension to query the dataframe
    file_name = file_path.split('/')[-1].replace('.pt', '')
    
    # Query the label from the dataframe using the file name
    true_label_name = combined_df[combined_df.StudyInstanceUID_anon == file_name].BPE_type.unique()[0]
    
    # Apply Maximum Intensity Projection (MIP) to get a 2D image
    mip_image = torch.max(image, dim=0)[0]  # MIP along the depth dimension

    # Prepare the image for the 2D model
    mip_image = mip_image.unsqueeze(0)  # Add channel dimension [1, H, W]

    # Remove the unnecessary extra dimension to make it [1, 128, 128]
    # mip_image = mip_image.unsqueeze(0)  # Add batch dimension [1, 1, H, W]
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(mip_image.to(device))
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        predicted_label = predicted.item()
        predicted_label_name = class_names[predicted_label]

    # Display predicted label
    print(f"True Label: {true_label_name}, Predicted Label: {predicted_label_name}")

    # Display the image with both true and predicted labels
    plt.figure(figsize=(4, 4))
    plt.imshow(mip_image.squeeze().cpu(), cmap='gray')
    plt.title(f"True: {true_label_name}, Predicted: {predicted_label_name}")
    plt.axis('off')
    plt.show()