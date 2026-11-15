# Brain Tumor Segmentation with 2D U-Net

A PyTorch-based medical image segmentation project using the **BraTS 2019** MRI dataset. The notebook explores multi-modal brain MRI data and implements a **2D U-Net** to perform pixel-wise segmentation of brain tumor regions from axial slices.

> **Project status:** Experimental notebook / baseline implementation. The current notebook implements and trains a 2D U-Net. A 3D U-Net, quantitative Dice evaluation, and benchmark results are not yet included in this version.

## Overview

Brain tumor segmentation consists of identifying tumor-related regions in MRI scans at the pixel level. This project uses four complementary MRI modalities as input to a U-Net model and predicts one of four segmentation classes for every pixel.

The work is organized around:

- Exploring and visualizing the BraTS 2019 dataset.
- Loading multi-modal MRI volumes and their ground-truth masks.
- Selecting a manageable patient subset for experimentation.
- Preparing 2D axial-slice samples.
- Implementing a multi-class 2D U-Net in PyTorch.
- Training with a combined Dice and Cross-Entropy loss.
- Monitoring validation loss and saving the best checkpoint.

## Dataset

The project uses the **BraTS 2019 training dataset**, obtained from the Kaggle distribution:

- [Brain Tumor Segmentation – BraTS 2019 (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)

The notebook expects the following directory structure:

```text
MICCAI_BraTS_2019_Data_Training/
├── HGG/
│   ├── patient_001/
│   │   ├── *_flair.nii
│   │   ├── *_t1.nii
│   │   ├── *_t1ce.nii
│   │   ├── *_t2.nii
│   │   └── *_seg.nii
│   └── ...
└── LGG/
    ├── patient_002/
    │   ├── *_flair.nii
    │   ├── *_t1.nii
    │   ├── *_t1ce.nii
    │   ├── *_t2.nii
    │   └── *_seg.nii
    └── ...
```

The dataset contains 335 training patients in total:

- 259 High-Grade Glioma (HGG) cases.
- 76 Low-Grade Glioma (LGG) cases.

Each patient provides four MRI modalities and a corresponding segmentation mask.

### MRI modalities

| Modality | Description |
|---|---|
| T1 | T1-weighted MRI |
| T1ce | T1-weighted contrast-enhanced MRI |
| T2 | T2-weighted MRI |
| FLAIR | Fluid-Attenuated Inversion Recovery |

The four modalities provide complementary information about brain anatomy and tumor appearance.

### Segmentation labels

The original BraTS labels are `0`, `1`, `2`, and `4`. The notebook remaps label `4` to `3` so that the target classes are contiguous and compatible with a four-class model.

| Original label | Model label | Region |
|---:|---:|---|
| 0 | 0 | Background |
| 1 | 1 | Necrotic / non-enhancing tumor core |
| 2 | 2 | Peritumoral edema |
| 4 | 3 | Enhancing tumor |

## Methodology

### 1. Data exploration

The notebook:

- Checks the HGG and LGG patient folders.
- Inspects the available MRI modalities and segmentation masks.
- Loads NIfTI volumes with **NiBabel**.
- Visualizes representative axial slices.
- Displays ground-truth masks over MRI slices, including interactive slice exploration.

### 2. Patient subset and data splitting

To keep experimentation manageable, the implementation selects a subset of patients separately from the HGG and LGG folders.

In the current code, `subset_fraction = 0.1` is used. This selects approximately:

- 25 HGG patients.
- 7 LGG patients.
- 32 patients in total.

The selected patients are split into training, validation, and held-out test sets using patient-level splitting and grade stratification. The code uses a 75% / 12.5% / 12.5% split (approximately 24 / 4 / 4 patients).

Splitting at the patient level is important: slices from the same patient should not be distributed across different subsets, as this could lead to data leakage.

**Note:** These are internal experimental splits, not official BraTS benchmark splits. The subset size and split proportions above describe the executable code in the notebook.

### 3. Preprocessing and 2D slice construction

Each 3D MRI volume is loaded and normalized independently using its mean and standard deviation:

\[
x_{\text{normalized}} = \frac{x-\mu}{\sigma}
\]

A zero-standard-deviation check prevents division by zero.

The volume is then converted into axial 2D slices. For each slice:

- The corresponding T1, T1ce, T2, and FLAIR slices are stacked as four input channels.
- The corresponding segmentation slice is used as the target mask.
- The segmentation labels are remapped from `{0, 1, 2, 4}` to `{0, 1, 2, 3}`.

The resulting input has the shape:

```text
(C, H, W) = (4, H, W)
```

The target mask has the shape:

```text
(H, W)
```

All axial slices are retained, including slices without visible tumor regions.

### 4. 2D U-Net architecture

The model is implemented from scratch using PyTorch.

**Input:** 4 channels (T1, T1ce, T2, FLAIR)  
**Output:** 4 classes (background and three tumor regions)

The architecture follows the encoder–bottleneck–decoder design:

- **Encoder:** Four convolutional blocks extract increasingly abstract image features. Each block uses two `3×3` convolutions, Batch Normalization, and ReLU activation, followed by max pooling for downsampling.
- **Bottleneck:** A deeper convolutional block captures high-level feature representations.
- **Decoder:** Transposed convolutions upsample the feature maps.
- **Skip connections:** Encoder features are concatenated with decoder features at corresponding resolutions to preserve spatial information.
- **Output layer:** A `1×1` convolution produces four class logits for each pixel.

### 5. Loss function

The notebook implements a hybrid **Dice + Cross-Entropy loss** to address the imbalance between the large background region and smaller tumor regions.

\[
L_{\text{DiceCE}} = 0.5L_{\text{CE}} + 0.5L_{\text{Dice}}
\]

- **Cross-Entropy Loss:** Provides pixel-level multi-class classification supervision.
- **Dice Loss:** Encourages overlap between predicted regions and ground-truth masks.

The Dice component is computed from the softmax probabilities and one-hot encoded target masks.

### 6. Training and validation

The training loop includes:

- Forward propagation.
- Loss computation.
- Backpropagation.
- Parameter updates using AdamW.
- Validation under `torch.no_grad()`.
- Saving the model checkpoint when validation loss improves.

Current notebook settings:

| Parameter | Value |
|---|---|
| Model | 2D U-Net |
| Input channels | 4 |
| Number of classes | 4 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Batch size | `4` |
| Dice / CE weighting | `0.5 / 0.5` |
| Number of epochs in current run configuration | `1` |
| Random seed | `42` |

The notebook selects CUDA when a GPU is available; otherwise, it uses the CPU.

## Tech Stack

- **Python**
- **PyTorch** – model implementation and training
- **NiBabel** – NIfTI medical image loading
- **NumPy** – numerical processing
- **Pandas** – data handling
- **Matplotlib** – MRI and segmentation visualization
- **scikit-learn** – patient-level dataset splitting
- **tqdm** – training progress bars
- **Google Colab / Google Drive** – notebook execution and data storage
- **Kaggle API** – dataset download

## Getting Started

### Recommended environment

The notebook is designed to run in **Google Colab** and uses Google Drive to store the dataset and model checkpoint.

1. Open the notebook in Google Colab.
2. Enable GPU acceleration if available (`Runtime` → `Change runtime type` → `T4 GPU`, or another available GPU).
3. Mount Google Drive when prompted.
4. Add your Kaggle API credentials to Colab Secrets using the names:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
5. Run the notebook cells in order.

The notebook downloads the dataset using the Kaggle CLI and extracts it into the configured Google Drive directory. Adjust `project_path` if you want to use a different location.

### Main notebook

```text
Segmentations.ipynb
```

## Outputs

The training code saves the best 2D U-Net weights (based on validation loss) to:

```text
/content/drive/MyDrive/segmentation_project/best_model_2d.pth
```

Change `project_seg` or `checkpoint_path_2d` in the notebook to use another output directory.

## Current Scope and Next Steps

The current notebook is a first 2D baseline. Potential next steps include:

- Train for more epochs and monitor convergence.
- Implement quantitative Dice evaluation per tumor class on the held-out test set.
- Visualize model predictions alongside the MRI slices and ground-truth masks.
- Compare alternative sampling strategies for tumor-containing and empty slices.
- Add a 3D U-Net to exploit volumetric context across adjacent slices.
- Compare 2D and 3D approaches in terms of segmentation quality, memory use, and training time.
- Investigate uncertainty estimation to understand where model predictions are less reliable.

