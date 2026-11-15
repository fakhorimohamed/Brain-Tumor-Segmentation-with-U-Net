# Brain Tumor Segmentation using 2D and 3D U-Net

A Deep Learning project exploring and comparing **2D and 3D U-Net architectures** for multi-class brain tumor segmentation using multimodal MRI scans from the **BraTS 2019 dataset**.

The project investigates how spatial context influences segmentation performance, comparing slice-based processing with volumetric learning using PyTorch.

## Project Overview

Brain tumor segmentation aims to identify and delineate tumor regions from MRI scans at the pixel or voxel level.

This project implements two approaches:

- **2D U-Net:** Segments individual axial MRI slices independently.
- **3D U-Net:** Processes volumetric MRI patches to capture spatial relationships across adjacent slices.

Both architectures use four MRI modalities (T1, T1ce, T2, and FLAIR) to predict three tumor subregions and the background.

## Dataset

The project uses the **BraTS 2019 training dataset**, containing multimodal brain MRI volumes from patients diagnosed with gliomas.

**Dataset source:** [Brain Tumor Segmentation – BraTS 2019 (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)

### MRI Modalities

| Modality | Description |
|---|---|
| T1 | T1-weighted MRI |
| T1ce | Contrast-enhanced T1-weighted MRI |
| T2 | T2-weighted MRI |
| FLAIR | Fluid-Attenuated Inversion Recovery |

### Segmentation Labels

The original BraTS segmentation masks contain the following labels:

| Original Label | Model Label | Region |
|---:|---:|---|
| 0 | 0 | Background |
| 1 | 1 | Necrotic / Non-enhancing Tumor Core (NCR/NET) |
| 2 | 2 | Peritumoral Edema (ED) |
| 4 | 3 | Enhancing Tumor (ET) |

Original label `4` is remapped to `3` to obtain contiguous class indices for model training.

## Project Pipeline

The implementation follows a common workflow for both architectures:

1. **Data Exploration:** Inspect MRI modalities, volume dimensions, and segmentation masks.
2. **Preprocessing:** Load NIfTI volumes, normalize MRI intensities, and remap segmentation labels.
3. **Dataset Splitting:** Separate patients into training, validation, and test sets to prevent data leakage.
4. **Model Development:** Implement 2D and 3D U-Net architectures.
5. **Training:** Optimize both models using a hybrid Dice + Cross-Entropy loss.
6. **Evaluation:** Measure segmentation quality using the Dice Similarity Coefficient.
7. **Visualization:** Compare predictions with ground-truth tumor regions.

## Model Architectures

### 1. 2D U-Net

<details>
<summary>Architecture details</summary>

The 2D U-Net processes MRI volumes slice by slice using axial 2D images.

Each input slice contains four channels corresponding to the MRI modalities.

**Input:** `(Batch, 4, Height, Width)`

The architecture consists of:

- **Encoder:** Extracts hierarchical spatial features through convolutional blocks and downsampling.
- **Bottleneck:** Captures deeper feature representations.
- **Decoder:** Reconstructs spatial resolution through upsampling.
- **Skip Connections:** Combine encoder and decoder features to preserve spatial details.
- **Output Layer:** Produces four class logits for each pixel.

This approach is computationally efficient but does not explicitly model spatial relationships between adjacent MRI slices.

</details>

### 2. 3D U-Net

<details>
<summary>Architecture details</summary>

The 3D U-Net extends the conventional U-Net architecture using 3D convolutions, allowing the model to learn spatial features across depth, height, and width.

Instead of processing individual slices, it operates on volumetric MRI patches.

**Input:** `(Batch, 4, Depth, Height, Width)`

To reduce GPU memory consumption, a patch-based training strategy is adopted:

- **Patch size:** `64 × 64 × 64`
- **Training:** Random 3D patch extraction.
- **Validation:** Center cropping.
- **Testing:** Center cropping.

This approach enables volumetric feature learning while keeping memory requirements manageable.

</details>

## Training Configuration

| Parameter | 2D U-Net | 3D U-Net |
|---|---:|---:|
| Framework | PyTorch | PyTorch |
| Input Channels | 4 | 4 |
| Output Classes | 4 | 4 |
| Optimizer | AdamW | AdamW |
| Learning Rate | 1e-4 | 1e-4 |
| Weight Decay | 1e-5 | 1e-5 |
| Loss Function | Dice + CE | Dice + CE |
| Batch Size | 4 | 2 |
| Epochs | 1 | 15 |

### Loss Function

Both models use a hybrid Dice + Cross-Entropy loss to address class imbalance between background and tumor regions.

\[
\mathcal{L}_{total}
=
\lambda\mathcal{L}_{CE}
+
(1-\lambda)\mathcal{L}_{Dice}
\]

With equal weighting:

\[
\mathcal{L}_{total}
=
0.5\mathcal{L}_{CE}
+
0.5\mathcal{L}_{Dice}
\]

Cross-Entropy provides pixel-wise or voxel-wise classification supervision, while Dice loss encourages better overlap between predicted and ground-truth segmentation regions.

## Evaluation

The primary evaluation metric is the **Dice Similarity Coefficient (DSC)**.

\[
DSC(P,G)=
\frac{2|P\cap G|}{|P|+|G|}
\]

Where:
- \(P\): Predicted segmentation mask.
- \(G\): Ground-truth segmentation mask.

Evaluation focuses on the three tumor regions, excluding background:

- NCR/NET
- ED
- ET

The 2D model produces slice-level predictions that can be reconstructed into 3D volumes for volumetric evaluation.

The 3D model produces volumetric predictions from input patches. For a fair comparison, predictions should be reconstructed over the complete volume before calculating patient-level Dice scores.

## Technologies

| Category | Tools |
|---|---|
| Deep Learning | PyTorch |
| Medical Imaging | NiBabel |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Data Processing | Scikit-learn |
| Environment | Google Colab |

## Repository Structure

```text
Brain-Tumor-Segmentation/
│
├── Brain_Tumor_Segmentation_with_U-Net.ipynb
├── README.md
│
└── models/
    ├── best_model_2d.pth
    └── best_model_3d.pth
```

*Model checkpoints are generated during training and are not necessarily included in the repository.*

## Future Improvements

- Increase the number of training patients.
- Extend training and optimize hyperparameters.
- Implement complete-volume inference for the 3D U-Net.
- Compare per-class and per-patient Dice scores.
- Investigate tumor-aware patch sampling strategies.
- Explore attention mechanisms and uncertainty estimation.

## Project Objective

**To investigate the impact of spatial context on brain tumor segmentation by comparing slice-based 2D learning with volumetric 3D learning using multimodal MRI data.**
