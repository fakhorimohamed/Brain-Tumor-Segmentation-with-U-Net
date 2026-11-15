# Brain Tumor Segmentation using 2D and 3D U-Net

## 1. Project Overview

This project focuses on **brain tumor segmentation using Deep Learning**, based on the BraTS 2019 MRI dataset.

The objective is to automatically identify and segment different brain tumor regions from MRI scans using two architectures:

- **2D U-Net:** Processes MRI scans slice by slice.
- **3D U-Net:** Processes 3D MRI volumes using spatial information across multiple slices.

Both models use four MRI modalities (T1, T1ce, T2, and FLAIR) to predict the tumor regions.

## 2. Dataset

We use the **BraTS 2019 dataset**, which contains MRI scans from patients diagnosed with brain gliomas.

Each patient includes four MRI modalities and a ground-truth segmentation mask.

| MRI Modality | Description |
|---|---|
| T1 | T1-weighted MRI |
| T1ce | T1-weighted contrast-enhanced MRI |
| T2 | T2-weighted MRI |
| FLAIR | Fluid-Attenuated Inversion Recovery |

### Tumor Segmentation Classes

The original segmentation masks contain four labels:

| Label | Tumor Region |
|---|---|
| 0 | Background |
| 1 | Necrotic / Non-enhancing Tumor Core (NCR/NET) |
| 2 | Peritumoral Edema (ED) |
| 4 | Enhancing Tumor (ET) |

For model training, the original labels are remapped from `{0, 1, 2, 4}` to `{0, 1, 2, 3}`.

**Dataset source:** [BraTS 2019 – Kaggle](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)

## 3. Project Workflow

The project follows these main steps:

1. **Data Exploration:** Explore the MRI modalities, patient structure, and segmentation masks.
2. **Preprocessing:** Load NIfTI volumes, normalize MRI intensities, and remap segmentation labels.
3. **Dataset Splitting:** Divide patients into training, validation, and testing sets.
4. **Model Development:** Implement two different U-Net architectures.
5. **Training:** Train both models using a hybrid Dice + Cross Entropy loss.
6. **Evaluation:** Measure segmentation performance using the Dice Similarity Coefficient (DSC).
7. **Visualization:** Compare predicted tumor regions with the ground-truth masks.

## 4. 2D U-Net Implementation

The first approach uses a conventional 2D U-Net architecture.

Instead of processing an entire MRI volume, we extract individual axial slices. Each slice contains four input channels corresponding to the four MRI modalities.

**Input shape:** `(Batch, 4, Height, Width)`

The architecture consists of:
- An encoder to extract spatial features.
- A bottleneck to capture deeper representations.
- A decoder to reconstruct the segmentation map.
- Skip connections to preserve spatial information.

The model predicts four classes for every pixel: background and three tumor regions.

## 5. 3D U-Net Implementation

The second approach uses a 3D U-Net to capture spatial information across multiple MRI slices.

Unlike the 2D architecture, 3D U-Net uses 3D convolutions to learn volumetric features.

Since processing complete MRI volumes requires significant GPU memory, we adopt a **patch-based training strategy**.

- **Patch size:** `64 × 64 × 64`
- **Training:** Random 3D cropping.
- **Validation and Testing:** Center cropping.

**Input shape:** `(Batch, 4, Depth, Height, Width)`

This approach allows the model to learn volumetric tumor structures while reducing GPU memory requirements.

## 6. Training Configuration

Both architectures use similar optimization strategies.

| Configuration | 2D U-Net | 3D U-Net |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning Rate | 0.0001 | 0.0001 |
| Weight Decay | 0.00001 | 0.00001 |
| Loss Function | Dice + CE | Dice + CE |
| Input Channels | 4 | 4 |
| Output Classes | 4 | 4 |
| Batch Size | 4 | 2 |
| Epochs | 1 | 15 |

### Loss Function

We use a hybrid **Dice + Cross Entropy Loss** to address the class imbalance problem in medical image segmentation.

- **Cross Entropy:** Measures pixel-wise or voxel-wise classification errors.
- **Dice Loss:** Measures the overlap between predicted segmentation regions and ground-truth masks.

The combination helps the models learn both accurate class predictions and better tumor-region segmentation.

## 7. Evaluation Metrics

We evaluate both models using the **Dice Similarity Coefficient (DSC)**.

The Dice score measures the overlap between the predicted segmentation and the ground truth.

\[
Dice = \frac{2|P \cap G|}{|P|+|G|}
\]

Where:
- P represents the predicted segmentation.
- G represents the ground-truth segmentation.

The evaluation focuses on the three tumor regions, excluding the background class.

For the 2D model, we evaluate individual slices and reconstruct predictions for volume-level evaluation.

For the 3D model, we evaluate the predicted 3D patches.

## 8. Technologies Used

- Python
- PyTorch
- NumPy
- NiBabel
- Matplotlib
- Scikit-learn
- Google Colab

## 9. Project Structure

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

*Note: The model checkpoint files are generated after training and can be added to the repository or shared separately.*

## 10. Future Improvements

- Increase the training dataset size.
- Experiment with different U-Net configurations.
- Improve tumor-region sampling strategies.
- Compare 2D and 3D segmentation performance.
- Explore attention mechanisms and other segmentation architectures.

---

**Project Goal:** Explore and compare 2D and 3D deep learning approaches for automated brain tumor segmentation using multimodal MRI data.