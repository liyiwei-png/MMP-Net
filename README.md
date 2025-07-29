# MMP-Net
Medical image segmentation is a key technology in computer-aided diagnosis systems, and the accuracy of the existing segmentation models cannot meet the need in practical applications. To improve the accuracy of the model,this paper proposes a novel U-shaped network called MMP-Net, meticulously crafted for medical image segmentation tasks. The MMP-Net incorporates three core modules: the multi-kernel convolution (MKC) module, which enhances the multi-receptive-field representation of the model; the multi-layer-guided channel attention (MLCA) module, which combines the features from different encoder layers and the combined features are used to guide the channel attention;the phase-guided Laplacian convolution (PGLC) module, which leverages the boundary sensitivity of Laplacian convolution kernels to effectively capture edge gradient changes and local detail textures in images. The proposed MMP-Net has been validated with two metrics (Dice and HD95) in five public medical image datasets (ISIC2017, ISIC2018, BUSI, COVID, PH2). Experimental results show that the proposed MMP-Net outperforms other popular models in all five datasets with minimal model parameters and computational cost, which are 2.03M and 5.3G respectively. This achievement offers an efficient and accurate solution for medical image segmentation tasks, making it particularly suitable for mobile healthcare and edge computing scenarios. 

## Datasets

Tested on 5 medical image segmentation datasets:
- BUSI (Breast Ultrasound)
- COVID-19 (Lung CT) 
- ISIC2018 (Skin Lesion)
- ISIC2017 (Skin Lesion)
- PH2 (Dermoscopy)

## Setup

```bash
conda create -n mmpnet python=3.8
conda activate mmpnet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install numpy opencv-python pillow scikit-learn matplotlib
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

2. **Train**: 
```bash
python Experiment/train_model.py

3. **Test**: 
```bash
python Experiment/test_model.py


## Key Parameters

- Learning rate: 1e-3
- Batch size: 16
- Early stopping patience: 100
- Dropout rate: 0.1
- Weight decay: 1e-5

## Model Features

- **MKC**: Multi-scale feature extraction with different kernel sizes
- **PGLC**: Phase-rotation and Laplacian convolution for edge enhancement  
- **MLCA**: Multi-Layer-Guided Channel Attention Module

## Citation

If you find our repo useful for your research, please consider citing our article. This article has been submitted for peer-review in the journal "The Visual Computer".

```bibtex
@article{li2025enhancing,
  author = {Li, Yiwei and Li, Shiren and Davydov, Maksim and Viriri, Serestina and Talib, Irsa and Yuan, Zhihao and Yang, Guangguang},
  journal = {The Visual Computer},
  title = {Enhancing Medical Image Segmentation with the Modification of U-Shaped Network},
  year = {2025},
}
