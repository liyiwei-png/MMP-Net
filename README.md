# MMP-Net

Enhancing Medical Image Segmentation with the Modification of U-Shaped Network

## Datasets

Tested on 5 medical image segmentation datasets:
- BUSI (Breast Ultrasound)
- COVID-19 (Lung CT) 
- ISIC2018 (Skin Lesion)
- ISIC2017 (Skin Lesion)
- PH2 (Dermoscopy)

## Setup

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn matplotlib
```

## Dataset Structure

```
datasets/
├── BUSI_exp1/
│   ├── Train_Folder/
│   │   ├── img/
│   │   └── labelcol/
│   ├── Val_Folder/
│   │   ├── img/
│   │   └── labelcol/
│   └── Test_Folder/
│       ├── img/
│       └── labelcol/
└── ... (other datasets)
```

## Usage

1. **Configure**: Modify dataset paths and parameters in `Config.py`
2. **Train**: `python train_model.py`
3. **Test**: `python test_model.py`

## Key Parameters

- Learning rate: 1e-3
- Batch size: 16
- Early stopping patience: 130
- Dropout rate: 0.1
- Weight decay: 1e-5

## Model Features

- **MKC**: Multi-scale feature extraction with different kernel sizes
- **PGLC**: Phase-rotation and Laplacian convolution for edge enhancement  
- **MLCA**: Multi-Layer-Guided Channel Attention Module
