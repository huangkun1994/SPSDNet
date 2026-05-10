# SPSD-nnUNet: Self-Paced Self-Distillation for Medical Image Segmentation

This repository contains the implementation of our proposed Self-Paced Self-Distillation (SPSD) framework, built as an extension on top of [nnUNet V2](https://github.com/MIC-DKFZ/nnUNet). 

By integrating self-distillation mechanisms dynamically across both the encoder and decoder architectures, SPSD enhances feature representations and mitigates knowledge transfer inconsistencies.

## 🚀 Features

- **Dynamic Consensus Masking (CLG Block):** Provides flexible masking constraints using Top-K, fixed-threshold, or dynamically computed threshold modes to guarantee high-quality self-distillation.
- **Deep Encoder-Decoder Supervision:** Distills knowledge simultaneously through both encoder (`use_encoder_outputs`) and decoder (`use_aux_outputs`) streams.
- **Stable Distillation Measurement:** Implements symmetric Binary KL / JS Divergences and SSIM constraints for fine-grained alignment between main logits and auxiliary heads.
- **Plug-and-Play Integration:** Implemented natively as an `nnUNetTrainer` variant, allowing for zero-hassle testing in the nnUNet standard pipeline.

## 🛠️ Installation & Integration

SPSD is designed to be easily injected into an existing nnUNet V2 workspace workspace. 

1. Install the official `nnUNetv2` following their [official guide](https://github.com/MIC-DKFZ/nnUNet).
2. Clone or download this repository.
3. Copy all Python files from this repository directly into the `variants` directory of your nnUNet installation. Specifically:
   ```bash
   cp -r *.py /path/to/nnUNet/nnunetv2/training/nnUNetTrainer/variants/spsd/
   ```
   *(If the `spsd` folder does not exist, simply create it.)*

## 📚 Usage

### 1. Training

Run the training process natively using nnUNet commands by specifying our custom trainer `SPSDTrainer`.

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD_ID -tr SPSDTrainer
```
*(Replace `DATASET_ID` with your designated task ID and `FOLD_ID` with your cross-validation fold (e.g., 0).)*

### 2. Ablation & Configurations

SPSD provides multiple variants via environment variables or distinct trainer classes. You can modify these parameters on the fly via shell execution:

- `SPSD_DECODER_KL_WEIGHT`: Weight for decoder self-distillation (default: `1.0`)
- `SPSD_ENCODER_KL_WEIGHT`: Weight for encoder self-distillation (default: `1.0`)
- `SPSD_TEMP`: Distillation temperature (default: `1.5`)
- `SPSD_K_START`: Initial Kappa for Top-K masking (default: `0.5`)

Example ablation run mapping configuration into the command:
```bash
SPSD_TEMP=2.0 SPSD_K_START=0.3 nnUNetv2_train 123 3d_fullres 0 -tr SPSDTrainer
```

Alternate specialized trainers included:
- `SPSDTrainer_EncoderOnly`
- `SPSDTrainer_DecoderOnly`
- `SPSDTrainer_Threshold`
- `SPSDTrainer_Dynamic`

### 3. Inference

Inference operates exactly as it does in standard nnUNet, because the exported model wraps the checkpoint natively. 

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -tr SPSDTrainer -f FOLD_ID
```

## 📄 License & Acknowledgment

This code adheres to the Apache 2.0 license, consistent with normal nnUNet extensions. If our code assists your research, please consider citing our associated paper.

Special thanks to the authors of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for their excellent and highly extensible medical imaging framework.
