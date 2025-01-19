import numpy as np
np.float=np.float64
import os
import pydicom #for conversion
import dicom2nifti# conversion

import nibabel as nib
import matplotlib.pyplot as plt
import torch
from functools import partial
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.networks.nets import SwinUNETR

conversion_tasks = [
    {'dicom_directory': '/content/d1/T1', 'output_file': '/content'},
    {'dicom_directory': '/content/d1/T1ce', 'output_file': '/content'},
    {'dicom_directory': '/content/d1/T2', 'output_file': '/content'},
    {'dicom_directory': '/content/d1/flair', 'output_file': '/content'},
    {'dicom_directory': '/content/d2/T1', 'output_file': '/content/Untitled Folder'},
    {'dicom_directory': '/content/d2/T1ce', 'output_file': '/content/Untitled Folder'},
    {'dicom_directory': '/content/d2/T2', 'output_file': '/content/Untitled Folder'},
    {'dicom_directory': '/content/d2/flair', 'output_file': '/content/Untitled Folder'}
]

# Perform the conversions
for task in conversion_tasks:
    dicom_directory = task['dicom_directory']
    output_file = task['output_file']
    
    # Create the output directory if it does not exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert DICOM to NIFTI
    dicom2nifti.convert_directory(dicom_directory, output_file)

    print(f"Converted DICOM from {dicom_directory} to NIFTI file {output_file}")

print("All conversions are complete.")

# The NIfTI file will be saved in the specified output directory

# Define the model inference function
def perform_inference(model, data_loader, roi_size, sw_batch_size, overlap):

    model.eval()
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap
    )
    results = []
    with torch.no_grad():
        for batch_data in data_loader:
            image = batch_data["image"].cuda()
            prob = torch.sigmoid(model_inferer(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            results.append(seg_out)
    return results

# Function to create data loader
def create_data_loader(test_files, roi_size):
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            #transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    test_ds = Dataset(data=test_files, transform=test_transform)
    return DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

# Load model
def load_model(model_path, device):
    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)
    model.load_state_dict(torch.load(model_path)["state_dict"])
    return model

# Function to calculate the volume for a given unique value
def calculate_volume(seg_slice, unique_value):
    return np.sum(seg_slice == unique_value)

# Function to find the slice with the largest tumor volume
def find_largest_volume_slice(seg_out, unique_values):
    max_volume = 0
    best_slice = 0
    for i in range(seg_out.shape[2]):  # Iterate over slices
        current_volume = sum(calculate_volume(seg_out[:, :, i], val) for val in unique_values)
        if current_volume > max_volume:
            max_volume = current_volume
            best_slice = i
    return best_slice

# Inference for two sets of input files
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "model\model.pt"
    roi_size = (128, 128, 128)
    sw_batch_size = 1
    overlap = 0.6
    # Define file paths
    input_1 = [
        {
            "image": [
                "/content/Untitled Folder/11_t1post.nii.gz",
                "/content/Untitled Folder/29901_dt1.nii.gz",
                "/content/Untitled Folder/35742_flair_reg.nii.gz",
                "/content/Untitled Folder/11_t1post.nii.gz",
            ],
        }
    ]
    input_2 = [
        {
            "image": [
                "//content/11_t1post.nii.gz",
                "//content/23708_dt1.nii.gz",
                "//content/36500_flair_reg.nii.gz",
                "//content/37906_t2_reg.nii.gz",
            ],
        }
    ]
    model = load_model(model_path, device)
    test_loader_1 = create_data_loader(input_1, roi_size)
    test_loader_2 = create_data_loader(input_2, roi_size)
    # Perform inference
    results_1 = perform_inference(model, test_loader_1, roi_size, sw_batch_size, overlap)
    results_2 = perform_inference(model, test_loader_2, roi_size, sw_batch_size, overlap)
    # Find the slice with the largest tumor volume
    unique_values = [1, 2, 4]  # Unique segmentation values
    classes=['Class 1','Class 1','Class 2','Class 1','Class 4']
    seg_out_1 = results_1[0] if results_1 else np.zeros(roi_size)
    seg_out_2 = results_2[0] if results_2 else np.zeros(roi_size)
    slice_num = find_largest_volume_slice(seg_out_1, unique_values)
    print(f"Slice with largest tumor volume: {slice_num}")
    def visualize_and_save(seg_out_1, seg_out_2, slice_num, unique_values):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Slice {slice_num} - Difference", fontsize=16)
        # Extract slice data
        seg_slice_1 = seg_out_1[:, :, slice_num]
        seg_slice_2 = seg_out_2[:, :, slice_num]
        difference = seg_slice_2 - seg_slice_1
        # Display segmentation mask 1
        axes[0, 0].imshow(seg_slice_1, cmap="gray")
        axes[0, 0].set_title("Segmentation Mask 1")
        axes[0, 0].axis("off")
        # Display segmentation mask 2
        axes[0, 1].imshow(seg_slice_2, cmap="gray")
        axes[0, 1].set_title("Segmentation Mask 2")
        axes[0, 1].axis("off")
        # Display difference between masks
        axes[0, 2].imshow(difference, cmap="bwr", vmin=-1, vmax=1)
        axes[0, 2].set_title("Difference between Masks")
        axes[0, 2].axis("off")
        # Highlight areas where the unique value is present in Mask 1 but not in Mask 2
        highlight = np.logical_and(seg_slice_1 == 1, seg_slice_2 != 1)
        axes[0, 3].imshow(highlight, cmap="gray")
        axes[0, 3].set_title("Unique in Mask 1 but not in Mask 2")
        axes[0, 3].axis("off")
        # Loop through all unique values to generate images
        for i, unique_value in enumerate(unique_values):
            mask_1 = (seg_slice_1 == unique_value)
            mask_2 = (seg_slice_2 == unique_value)
            diff_mask = np.logical_xor(mask_2, mask_1)
            # Display the mask for the specific value in Mask 1
            axes[1, 0].imshow(mask_1, cmap="gray")
            axes[1, 0].set_title(f"Mask 1 - Value {unique_value}")
            axes[1, 0].axis("off")
            # Display the mask for the specific value in Mask 2
            axes[1, 1].imshow(mask_2, cmap="gray")
            axes[1, 1].set_title(f"Mask 2 -  {classes[unique_value]}")
            axes[1, 1].axis("off")
            # Display difference for the specific value between masks
            axes[1, 2].imshow(diff_mask, cmap="bwr", vmin=-1, vmax=1)
            axes[1, 2].set_title(f"Difference -  {classes[unique_value]}")
            axes[1, 2].axis("off")
            # Highlight where the unique value is present in Mask 1 but not in Mask 2
            highlight_unique = np.logical_and(mask_1, np.logical_not(mask_2))
            axes[1, 3].imshow(highlight_unique, cmap="gray")
            axes[1, 3].set_title(f"Unique in Mask 1 -  {classes[unique_value]}")
            axes[1, 3].axis("off")
            # Calculate and print volumes
            volume_1 = calculate_volume(seg_out_1[:, :, slice_num], unique_value)
            volume_2 = calculate_volume(seg_out_2[:, :, slice_num], unique_value)
            volume_difference = volume_2 - volume_1
            print(f"Volume for  {classes[unique_value]} in Mask 1: {volume_1}")
            print(f"Volume for  {classes[unique_value]} in Mask 2: {volume_2}")
            print(f"Volume difference for class {classes[unique_value]}: {volume_difference}\n")
            # Check if the tumor has grown or shrunk
            if volume_difference > 0:
                print(f"Tumor has grown for  {classes[unique_value]}")
            elif volume_difference < 0:
                print(f"Tumor has shrunk for  {classes[unique_value]}")
            else:
                print(f"No change in tumor volume for  {classes[unique_value]}")
        # Save images
        output_directory = "./output_images"
        os.makedirs(output_directory, exist_ok=True)
        image_path = os.path.join(output_directory, f"comparison_slice_{slice_num}.png")
        plt.savefig(image_path, bbox_inches="tight")
        plt.close()
        print(f"Image saved at {image_path}")
        
    # Visualize and save images
    visualize_and_save(seg_out_1, seg_out_2, slice_num, unique_values)
    plt.show()
    

if __name__ == "__main__":
    main()

