# SPSD-nnUNet: Self-Paced Self-Distillation for Medical Image Segmentation

This repository contains the implementation of our proposed Self-Paced Self-Distillation (SPSD) framework, built as an extension on top of [nnUNet V2](https://github.com/MIC-DKFZ/nnUNet). 

By integrating self-distillation mechanisms dynamically across the decoder architecture, SPSD enhances feature representations and mitigates knowledge transfer inconsistencies.

## Features

- **Dynamic Consensus Masking (DCG Block):** Provides flexible masking constraints using Top-K, fixed-threshold, or dynamically computed threshold modes to guarantee high-quality self-distillation.
- **Deep Decoder Supervision:** Distills knowledge simultaneously through decoder (`use_aux_outputs`) streams.
- **Stable Distillation Measurement:** Implements symmetric Binary KL / JS Divergences and SSIM constraints for fine-grained alignment between main logits and auxiliary heads.

## Installation & Integration

SPSD is designed to be easily injected into an existing nnUNet V2 workspace workspace. 

1. Install the official `nnUNetv2` following their [official guide](https://github.com/MIC-DKFZ/nnUNet).
2. Clone or download this repository.
3. Transfer all Python files from this repository directly into the nnUNetv2 repository.

## Usage

### 1. Training

Run the training process natively using nnUNet commands by specifying our custom trainer `SPSDTrainer`.

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD_ID -tr SPSDTrainer
```
*(Replace `DATASET_ID` with your designated task ID and `FOLD_ID` with your cross-validation fold (e.g., 0).)*

### 2. Inference

Inference operates exactly as it does in standard nnUNet, because the exported model wraps the checkpoint natively. 

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -tr SPSDTrainer -f FOLD_ID
```

## 📄 License & Acknowledgment

This code adheres to the Apache 2.0 license, consistent with normal nnUNet extensions. If our code assists your research, please consider citing our associated paper.

Special thanks to the authors of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for their excellent and highly extensible medical imaging framework.

