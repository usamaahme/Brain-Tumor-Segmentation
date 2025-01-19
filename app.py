from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import os
import shutil
import zipfile
import glob
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import shutil
import zipfile
import numpy as np
np.float=np.float64
import pydicom
import dicom2nifti

import nibabel as nib
import matplotlib.pyplot as plt
import torch
from functools import partial
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.networks.nets import SwinUNETR
import cv2
import io
import json

app = FastAPI()

# Mount a static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define the global model variable
model = None

# Define functions for inference
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

def create_data_loader(test_files, roi_size):
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
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

def calculate_volume(seg_slice, unique_value):
    return np.sum(seg_slice == unique_value)

def find_largest_volume_slice(seg_out, unique_values):
    max_volume = 0
    best_slice = 0
    for i in range(seg_out.shape[2]):  # Iterate over slices
        current_volume = sum(calculate_volume(seg_out[:, :, i], val) for val in unique_values)
        if current_volume > max_volume:
            max_volume = current_volume
            best_slice = i
    return best_slice

def generate_output(volume_1, volume_2, unique_value):
    volume_difference = volume_2 - volume_1
    status = "Tumor has shrunk" if volume_difference < 0 else "Tumor has grown" if volume_difference > 0 else "Tumor remains the same"
    return volume_difference, status

def visualize_and_save(seg_out_1, seg_out_2, slice_num, unique_values):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Slice {slice_num} - Difference", fontsize=16)
    seg_slice_1 = seg_out_1[:, :, slice_num]
    seg_slice_2 = seg_out_2[:, :, slice_num]
    difference = seg_slice_2 - seg_slice_1
    axes[0, 0].imshow(seg_slice_1, cmap="gray")
    axes[0, 0].set_title("Segmentation Mask 1")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(seg_slice_2, cmap="gray")
    axes[0, 1].set_title("Segmentation Mask 2")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(difference, cmap="bwr", vmin=-1, vmax=1)
    axes[0, 2].set_title("Difference between Masks")
    axes[0, 2].axis("off")
    highlight = np.logical_and(seg_slice_1 == 1, seg_slice_2 != 1)
    axes[0, 3].imshow(highlight, cmap="gray")
    axes[0, 3].set_title("Unique in Mask 1 but not in Mask 2")
    axes[0, 3].axis("off")
    for i, unique_value in enumerate(unique_values):
        mask_1 = (seg_slice_1 == unique_value)
        mask_2 = (seg_slice_2 == unique_value)
        diff_mask = np.logical_xor(mask_2, mask_1)
        axes[1, i].imshow(diff_mask, cmap="bwr", vmin=-1, vmax=1)
        axes[1, i].set_title(f"Difference - Value {unique_value}")
        axes[1, i].axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

# # Define functions for file handling
# def save_and_extract(file: UploadFile, extract_path: str):
#     os.makedirs(extract_path, exist_ok=True)
#     zip_path = os.path.join(extract_path, file.filename)
#     with open(zip_path, "wb") as f:
#         shutil.copyfileobj(file.file, f)
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)
#     os.remove(zip_path)

# def convert_dicom_to_nifti(dicom_dir: str, output_dir: str):
#     os.makedirs(output_dir, exist_ok=True)
#     dicom2nifti.convert_directory(dicom_dir, output_dir)

# Load the model on startup
@app.on_event("startup")
async def startup_event():
    global model
    model_path = "model/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)


@app.get("/", response_class=HTMLResponse)
async def get_html_tester():
    with open("static/index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)


def save_and_process_zip(file: UploadFile, person_id: int):
    extract_path = f"extracted{person_id}"
    convert_path = f"converted{person_id}"
    
    # Ensure directories are clean
    shutil.rmtree(extract_path, ignore_errors=True)
    shutil.rmtree(convert_path, ignore_errors=True)
    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(convert_path, exist_ok=True)
    
    # Save and extract zip file
    zip_path = os.path.join(extract_path, file.filename)
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    os.remove(zip_path)
    
    # Convert DICOM to NIfTI for each subfolder
    nifti_files = []
    for folder in os.listdir(extract_path):
        subfolder_path = os.path.join(extract_path, folder)
        if os.path.isdir(subfolder_path):
            output_nifti = os.path.join(convert_path)
            try:
                print(f"Converting {subfolder_path} to {output_nifti}","#######")
                dicom2nifti.convert_directory(subfolder_path, output_nifti, compression=True, reorient=True)
                nifti_files.append(os.listdir(output_nifti))
                nifti_files=set(nifti_files)
            except Exception as e:
                print(f"Error converting {subfolder_path}: {str(e)}")
    
    if len(nifti_files) != 4:
        raise ValueError(f"Expected 4 NIfTI files, but found {len(nifti_files)} in {convert_path}")
    
    return nifti_files

@app.post("/analyze/")
async def analyze_brain_scans(
    person1_files: UploadFile = File(...),
    person2_files: UploadFile = File(...)
):
    # Process files
    nifti_files = []
    print(nifti_files)
    for i, file in enumerate([person1_files, person2_files], 1):
        person_nifti_files = save_and_process_zip(file, i)
        nifti_files.append(person_nifti_files)
    



    input_1 = [
        {
            "image": [
                "converted1/11_t1post.nii.gz",
                "converted1/11_t1post.nii.gz",
                "converted1/11_t1post.nii.gz",
                "converted1/11_t1post.nii.gz",
            ],
        }
    ]
    input_2 = [
        {
            "image": [
                "converted2/11_t1post.nii.gz",
                "converted2/11_t1post.nii.gz",
                "converted2/11_t1post.nii.gz",
                "converted2/11_t1post.nii.gz",
            ],
        }
    ]

    # Prepare data for inference
    test_files = [input_1, input_2]
    roi_size = (128, 128, 128)
    sw_batch_size = 1
    overlap = 0.5

    data_loaders = [create_data_loader(files, roi_size) for files in test_files]

    # Perform inference
    seg_outs = [perform_inference(model, loader, roi_size, sw_batch_size, overlap)[0] for loader in data_loaders]

    unique_values = [1, 2, 4]
    slice_num = find_largest_volume_slice(seg_outs[0], unique_values)
    print(f"Slice with largest tumor volume: {slice_num}")

    # Calculate volumes and generate results
    volume_results = []
    for unique_value in unique_values:
        volume_1 = calculate_volume(seg_outs[0][:, :, slice_num], unique_value)
        volume_2 = calculate_volume(seg_outs[1][:, :, slice_num], unique_value)
        volume_diff, status = generate_output(volume_1, volume_2, unique_value)
        volume_results.append({
            "class": unique_value,
            "volume_mask_1": float(volume_1),
            "volume_mask_2": float(volume_2),
            "volume_difference": float(volume_diff),
            "status": status
        })
    print(volume_results)
    # Generate visualization
    buf = visualize_and_save(seg_outs[0], seg_outs[1], slice_num, unique_values)
    
    # Convert the image buffer to base64
    buf.seek(0)
    img = Image.open(buf)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # Prepare the response
    response_data = {
        "volume_results": volume_results,
        "slice_num": int(slice_num),
        "image": img_base64
    }

    return JSONResponse(content=response_data)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)