import os
import torch
import time
import ml_collections

save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)
n_filts = 32
cosineLR = True

min_lr = 1.0e-6
warmup_epochs = 5
n_channels = 3
n_labels = 1
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 100
weight_decay = 1e-5
dropout_rate = 0.1
pretrain = False

# task_name = 'Covid_exp1'
task_name = 'BUSI_exp1'
# task_name = 'ISIC2018'
# task_name = 'isic2017'
# task_name = 'PH2_exp1'


learning_rate = 1e-3
# learning_rate = 1.00e-04
# batch_size = 1
batch_size = 16
# model_name = 'ACC_UNet'
# model_name = 'SwinUnet'
# model_name = 'SMESwinUnet'
# model_name = 'UCTransNet'
# model_name = 'UNet_base'
# model_name = 'MultiResUnet1_32_1.67'
model_name = 'MMP_Net'
test_session = "session"

# train_dataset = '/home/ta/yy/ACC-UNet-main/datasets/isic2017/Train_Folder'
# val_dataset = '/home/ta/yy/ACC-UNet-main/datasets/isic2017/Val_Folder'
# test_dataset = '/home/ta/yy/ACC-UNet-main/datasets/isic2017/Test_Folder'

# train_dataset = '/home/ta/yy/ACC-UNet-main/datasets/Covid_exp1/Train_Folder'
# val_dataset = '/home/ta/yy/ACC-UNet-main/datasets/Covid_exp1/Val_Folder'
# test_dataset = '/home/ta/yy/ACC-UNet-main/datasets/Covid_exp1/Test_Folder'

train_dataset = '/home/ta/yy/ACC-UNet-main/datasets/BUSI-combined/Train_Folder'
val_dataset = '/home/ta/yy/ACC-UNet-main/datasets/BUSI-combined/Val_Folder'
test_dataset = '/home/ta/yy/ACC-UNet-main/datasets/BUSI-combined/Test_Folder'

# train_dataset = '/home/ta/yy/ACC-UNet-main/datasets/ph2_dataset_split/train'
# val_dataset = '/home/ta/yy/ACC-UNet-main/datasets/ph2_dataset_split/val'
# test_dataset = '/home/ta/yy/ACC-UNet-main/datasets/ph2_dataset_split/test'

# train_dataset = '/home/ta/yy/ACC-UNet-main/datasets/ISIC18_exp1/Train_Folder'
# val_dataset = '/home/ta/yy/ACC-UNet-main/datasets/ISIC18_exp1/Val_Folder'
# test_dataset = '/home/ta/yy/ACC-UNet-main/datasets/ISIC18_exp1/Test_Folder'

session_name = 'session'
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


def get_dataset_loader(dataset_path, transform=None, image_size=224):
    print(f"Detecting dataset structure: {dataset_path}")

    if (os.path.exists(os.path.join(dataset_path, 'images')) and
            os.path.exists(os.path.join(dataset_path, 'labels'))):
        print("Detected images/labels dataset structure, using ImagesLabelsDataset loader")
        return ImagesLabelsDataset(dataset_path, transform, image_size)

    elif (os.path.exists(os.path.join(dataset_path, 'images')) and
          os.path.exists(os.path.join(dataset_path, 'masks'))):
        print("Detected images/masks dataset structure (FIVES after splitting), using ImagesLabelsDataset loader")
        return ImagesLabelsDataset(dataset_path, transform, image_size, mask_folder='masks')

    elif (os.path.exists(os.path.join(dataset_path, 'train', 'Original')) and
          os.path.exists(os.path.join(dataset_path, 'train', 'Ground truth'))):
        print("Detected FIVES dataset (train/Original and train/Ground truth structure), using FIVESDataset loader")
        return FIVESDataset(dataset_path, transform, image_size)

    elif (os.path.exists(os.path.join(dataset_path, 'trainx')) and
          os.path.exists(os.path.join(dataset_path, 'trainy'))):
        print("Detected PH2 dataset (trainx/trainy structure), using PH2Dataset loader")
        return PH2Dataset(dataset_path, transform, image_size)

    elif any(f.endswith('.h5') for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))):
        print("Detected ACDC dataset (.h5 files), using ACDCDataset loader")
        return ACDCDataset(dataset_path, transform, image_size)

    elif any(f.endswith('.npz') for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))):
        print("Detected Synapse dataset (.npz files), using SynapseDataset loader")
        return SynapseDataset(dataset_path, transform, image_size)

    elif (os.path.exists(os.path.join(dataset_path, 'img')) and
          os.path.exists(os.path.join(dataset_path, 'labelcol'))):
        print("Detected traditional image dataset, using ImageToImage2D loader")
        return ImageToImage2D(dataset_path, transform, image_size)

    else:
        available_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        available_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

        error_msg = f"Unsupported dataset format: {dataset_path}\n"
        error_msg += f"Available subdirectories: {available_dirs}\n"
        error_msg += f"Available files: {available_files[:5]}{'...' if len(available_files) > 5 else ''}\n"
        error_msg += "Supported formats:\n"
        error_msg += "- General: images/labels directories\n"
        error_msg += "- FIVES after splitting: images/masks directories\n"
        error_msg += "- FIVES original: train/Original and train/Ground truth directories\n"
        error_msg += "- PH2: trainx/trainy directories\n"
        error_msg += "- ACDC: .h5 files\n"
        error_msg += "- Traditional: img/labelcol directories"

        raise ValueError(error_msg)


def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config