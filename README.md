# BraTS21 Tumor Segmentation

This project implements a tumor segmentation model using the DICOM (Brain Tumor) dataset that was converted to NIFTI. It utilizes the SwinUNETR model for segmentation and provides functionality for inference, visualization, and volume calculation.

## Features

- Tumor segmentation using SwinUNETR model
- Sliding window inference for processing large 3D volumes
- Visualization of segmentation results and differences between two sets of inputs
- Volume calculation for different tumor regions

## Requirements

- Python 
- PyTorch
- MONAI
- NumPy
- Nibabel
- Matplotlib

## Converssion Requirements
dicom2nifti
pydicom

exact versions are given in requirements.txt file
## Installation

1. orignal repository:
   ```
   git clone https://github.com/yourusername/brats21-tumor-segmentation.git
 
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the pre-trained model and place it in the project root directory as `model.pt`.

## Usage

1. Prepare your  DICOM dataset files. The code expects the following file structure:
   ```
   /content/zip/
   ├── T1
   ├── T2
   ├── Flair
   └── tice
   ```   
   for comparison we need two sets of inputs like these

2. Update the `input_1` and `input_2` variables in the `main()` function with the paths to your input data.

3. Run the script:
   ```
   python main.py
   ```

4. The script will perform inference on both sets of inputs, visualize the results, and save a comparison image as `slice_67_comparison.png` 

5. Volume calculations and tumor growth/shrinkage information will be printed to the console.

## Customization

- Modify the `unique_values` list to focus on specific tumor regions (1: necrotic and non-enhancing tumor core, 2: peritumoral edema, 4: enhancing tumor).
- Change the `roi_size`, `sw_batch_size`, and `overlap` parameters in the `main()` function to adjust the sliding window inference behavior.

## Output

The script generates:
1. that which Slice has the highest Volume e.g 6.
2. Console output with volume calculations and tumor growth/shrinkage information.
3. A PNG image (`output_comparison.png`) showing the segmentation results, differences, and highlighted regions for the specified slice.



## Acknowledgments

This project uses the DICOM Brain Tumor dataset and the SwinUNETR model architecture. 
