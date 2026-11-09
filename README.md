# NSLN
the prediction of NSLN metastasis in patients with SLN-positive breast cancer
Overview
This repository contains the core code for the manuscript:
TITLE

Environment Setup
See environment.yml
bash
# Create conda environment
conda env create -f environment.yml


Data Structure
../data_incohort or data_excohort{No.}/
├── MRMG
│   ├── Positive/case/
│   │   ├── MG
│   ├── Negative/case/
│   │   ├── MG
└── output_data_{split_type}.pkl or output_excohort{No.}_data_{split_type}.pkl
└── CP.xlsx

Usage

# get_SAmask.py
Unified pre-computation of adaptive decay masks for DCE (Gaussian) and MLO (Linear) tumor target volumes.
This script converts binary tumor segmentations (`GTV` and `GTVnd1`) into spatial masks where the core region equals `1` and values smoothly decay to `0` outward. The output can be used as an additional input channel for multi-modal models (DCE-MRI, MG-MLO).
| Modality | Decay Type | Key Parameter | Output File |
|----------|------------|---------------|-------------|
| DCE | Adaptive Gaussian | `sigma_base` (default `3.0`) | `{ID}/{ID}_DCE_{suffix}.nii.gz` |
| MLO | Adaptive Linear | `distance_factor` (default `3.0`) | `{ID}/MG/{ID}_*MLO_{suffix}.nii.gz` |
Shared rules for both modalities:
- The union `GTV ∪ GTVnd1` forms the core region (value = `1`);
- Values decay outward from the core;
- The decay scale adapts to the equivalent radius `r = sqrt(area / π)` of the mask.

Basic Commands
bash
# Train SAMT full multi-modal model (2 images + 2 features)
python train.py --task pathology_pretrain --mode train --fold 1 --bs 4 --epoch 100 --seed 42 --date all --dropout 0
# test
python train.py --task pathology_pretrain --mode test --fold 1 --bs 4 --epoch 100 --seed 42 --date all --dropout 0
