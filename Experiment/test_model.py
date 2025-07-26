import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
import pickle
import glob
from scipy.spatial.distance import directed_hausdorff
from scipy import ndimage
import cv2

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
from torch import nn
from nets.ACC_UNet import ACC_UNet
from nets.UCTransNet import UCTransNet
from nets.UNet_base import UNet_base
from nets.SMESwinUnet import SMESwinUnet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.MMP_Net import MMP_Net
from utils import *
import numpy as np
from sklearn.metrics import jaccard_score


def check_dataset_path(dataset_path):
    print(f"Checking dataset path: {dataset_path}")

    if os.path.exists(dataset_path):
        print(f"✓ Path exists: {dataset_path}")
        subdirs = []
        try:
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    subdirs.append(item)
            print(f"Found subdirectories: {subdirs}")
        except:
            pass

        try:
            files = os.listdir(dataset_path)
            has_synapse_files = any(f.endswith(('.npz', '.npy', '.h5')) for f in files)
            has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
                             for f in files if os.path.isfile(os.path.join(dataset_path, f)))

            if has_synapse_files or has_images:
                print(f"✓ Found data files: {dataset_path}")
                return dataset_path
        except:
            pass

        possible_img_dirs = [
            "img", "images", "image", "imgs", "test_images",
            "data", "x", "trainx", "input", "test_vol_h5", "vol_h5"
        ]

        for img_dir in possible_img_dirs:
            img_path = os.path.join(dataset_path, img_dir)
            if os.path.exists(img_path):
                print(f"✓ Found data directory: {img_path}")
                try:
                    files = os.listdir(img_path)
                    has_synapse_files = any(f.endswith(('.npz', '.npy', '.h5')) for f in files)
                    has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
                                     for f in files if os.path.isfile(os.path.join(img_path, f)))

                    if has_synapse_files or has_images:
                        print(f"✓ Data directory contains data files: {img_path}")
                        return dataset_path
                except:
                    continue

        print(f"No data files found in {dataset_path}, trying to search subdirectories...")

    print(f"✗ Path does not exist: {dataset_path}")

    if "Synapse" in dataset_path:
        print("This is Synapse dataset path, prioritizing Synapse related paths")
        base_dirs = [
            "/home/ta/yy/ACC-UNet-main/datasets",
            "/home/ta/yy/ACC-UNet-main/Experiments/datasets",
            "./datasets",
            "../datasets"
        ]

        synapse_patterns = [
            "Synapse/split_data/test",
            "Synapse/test_vol_h5",
            "Synapse/test",
            "Synapse_split/test",
            "synapse/split_data/test",
            "synapse/test_vol_h5",
            "synapse/test",
            "Synapse/split_data/test_vol_h5",
            "synapse/split_data/test_vol_h5",
            "Synapse/train_npz",
            "Synapse/test_npz",
            "synapse/train_npz",
            "synapse/test_npz",
            "Synapse/Abdomen/RawData/Testing",
            "Synapse/Abdomen/RawData/Training",
            "synapse/Abdomen/RawData/Testing",
            "synapse/Abdomen/RawData/Training",
        ]

        print("Searching for Synapse dataset paths...")
        for base_dir in base_dirs:
            for pattern in synapse_patterns:
                potential_path = os.path.join(base_dir, pattern)
                if os.path.exists(potential_path):
                    print(f"✓ Found Synapse path: {potential_path}")
                    try:
                        files = os.listdir(potential_path)
                        has_synapse_files = any(f.endswith(('.npz', '.npy', '.h5')) for f in files)
                        has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
                                         for f in files if os.path.isfile(os.path.join(potential_path, f)))

                        if has_synapse_files or has_images:
                            print(f"✓ Synapse path contains data files: {potential_path}")
                            return potential_path
                        else:
                            print(f"- Synapse path exists but does not contain data files: {potential_path}")
                    except:
                        continue

        print("No valid Synapse dataset path found")
        return None


    base_dirs = [
        "/home/ta/yy/ACC-UNet-main/datasets",
        "/home/ta/yy/ACC-UNet-main/Experiments/datasets",
        "./datasets",
        "../datasets"
    ]

    dataset_patterns = [
        "FIVES*/test/img",
        "FIVES*/test/images",
        "FIVES*/test",
        "FIVES*/split/test/img",
        "FIVES*/split/test/images",
        "FIVES*/split/test",
        "FIVES A Fundus Image Dataset*/test/img",
        "FIVES A Fundus Image Dataset*/test/images",
        "FIVES A Fundus Image Dataset*/test",
        "FIVES A Fundus Image Dataset*/split/test/img",
        "FIVES A Fundus Image Dataset*/split/test/images",
        "FIVES A Fundus Image Dataset*/split/test",
        "*FIVES*/test/img",
        "*FIVES*/test/images",
        "*FIVES*/test",
        "*FIVES*/split/test/img",
        "*FIVES*/split/test/images",
        "*FIVES*/split/test",
        "ACDC_split/test/slices/img",
        "ACDC/test/slices/img",
        "ACDC_split/test/img",
        "ACDC/test/img",
        "ACDC_split/test/slices",
        "ACDC/test/slices",
        "ACDC_split/test",
        "ACDC/test",
        "ACDC_exp1/test/slices/img",
        "ACDC_exp1/test/img",
        "ACDC_exp1/test",
        "ph2_dataset_split/test/img",
        "ph2_dataset_split/test",
        "PH2_split/test/img",
        "PH2_split/test",
        "PH2/test/img",
        "PH2/test",
        "ph2_dataset/test/img",
        "ph2_dataset/test",
        "GlaS_split/test/img",
        "GlaS_split/test",
        "GlaS/test/img",
        "GlaS/test",
        "ISIC18_split/test/img",
        "ISIC18_split/test",
        "ISIC2018/test/img",
        "ISIC2018/test",
        "*/test/img",
        "*/test/images",
        "*/test",
        "*_split/test/img",
        "*_split/test/images",
        "*_split/test"
    ]

    print("Searching for possible dataset paths...")
    for base_dir in base_dirs:
        for pattern in dataset_patterns:
            if '*' in pattern:
                import glob
                full_pattern = os.path.join(base_dir, pattern)
                matching_paths = glob.glob(full_pattern)
                for potential_path in matching_paths:
                    if os.path.exists(potential_path):
                        print(f"✓ Found possible path: {potential_path}")
                        try:
                            files = os.listdir(potential_path)
                            has_images = any(
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.h5', '.bmp'))
                                for f in files if os.path.isfile(os.path.join(potential_path, f)))
                            if has_images:
                                print(f"✓ Path contains data files: {potential_path}")
                                if potential_path.endswith(('/img', '/images')):
                                    return os.path.dirname(potential_path)
                                else:
                                    return potential_path
                            else:
                                print(f"- Path exists but does not contain data files: {potential_path}")
                        except:
                            continue
            else:
                potential_path = os.path.join(base_dir, pattern)
                if os.path.exists(potential_path):
                    print(f"✓ Found possible path: {potential_path}")
                    try:
                        files = os.listdir(potential_path)
                        has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.h5', '.bmp'))
                                         for f in files if os.path.isfile(os.path.join(potential_path, f)))
                        if has_images:
                            print(f"✓ Path contains data files: {potential_path}")
                            if potential_path.endswith(('/img', '/images')):
                                return os.path.dirname(potential_path)
                            else:
                                return potential_path
                        else:
                            print(f"- Path exists but does not contain data files: {potential_path}")
                    except:
                        continue

    print("\nNo matching dataset path found. Listing available datasets...")
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            print(f"\nFound in {base_dir}:")
            try:
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  📁 {item}")
                        # Try to list subdirectories
                        try:
                            subitems = os.listdir(item_path)
                            for subitem in subitems[:5]:  # Show only first 5
                                subitem_path = os.path.join(item_path, subitem)
                                if os.path.isdir(subitem_path):
                                    print(f"    📁 {subitem}")
                                else:
                                    print(f"    📄 {subitem}")
                            if len(subitems) > 5:
                                print(f"    ... {len(subitems) - 5} more items")
                        except:
                            pass
            except Exception as e:
                print(f"  Cannot list directory contents: {e}")

    return None


def find_latest_model(task_name, model_type):
    pattern = f'{task_name}/{model_type}/*/models/best_model-{model_type}.pth.tar'
    model_files = glob.glob(pattern)

    if not model_files:
        alternative_patterns = [
            f'{task_name}/{model_type}/session/models/best_model-{model_type}.pth.tar',
            f'{task_name}/{model_type}/models/best_model-{model_type}.pth.tar',
            f'{task_name}/{model_type}/**/best_model-{model_type}.pth.tar',
        ]

        for alt_pattern in alternative_patterns:
            model_files = glob.glob(alt_pattern, recursive=True)
            if model_files:
                break

        if not model_files:
            raise FileNotFoundError(f"No {model_type} model files found, search pattern: {pattern}")
    latest_model = max(model_files, key=os.path.getmtime)
    save_path = os.path.dirname(os.path.dirname(latest_model)) + '/'
    print(f"Found model file: {latest_model}")
    print(f"Corresponding save path: {save_path}")
    return latest_model, save_path


def calculate_recall(prediction, ground_truth):
    pred_binary = (prediction > 0.5).astype(np.float32)
    gt_binary = (ground_truth > 0.5).astype(np.float32)
    tp = np.sum(pred_binary * gt_binary)
    fn = np.sum(gt_binary * (1 - pred_binary))
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 1.0 if np.sum(gt_binary) == 0 else 0.0
    return recall


def calculate_hd95(prediction, ground_truth):
    pred_binary = (prediction > 0.5).astype(np.uint8)
    gt_binary = (ground_truth > 0.5).astype(np.uint8)
    if len(pred_binary.shape) > 2:
        pred_binary = pred_binary.squeeze()
    if len(gt_binary.shape) > 2:
        gt_binary = gt_binary.squeeze()
    pred_binary = np.ascontiguousarray(pred_binary, dtype=np.uint8)
    gt_binary = np.ascontiguousarray(gt_binary, dtype=np.uint8)
    if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
        return 0.0
    if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
        return 100.0
    try:
        pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except Exception as e:
        print(f"Contour detection error: {e}")
        return 100.0
    if len(pred_contours) == 0 or len(gt_contours) == 0:
        return 100.0
    pred_points = []
    for contour in pred_contours:
        pred_points.extend(contour.reshape(-1, 2))
    gt_points = []
    for contour in gt_contours:
        gt_points.extend(contour.reshape(-1, 2))
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 100.0
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)
    try:
        distances_pred_to_gt = []
        for pred_point in pred_points:
            min_dist = np.min(np.sqrt(np.sum((gt_points - pred_point) ** 2, axis=1)))
            distances_pred_to_gt.append(min_dist)
        distances_gt_to_pred = []
        for gt_point in gt_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - gt_point) ** 2, axis=1)))
            distances_gt_to_pred.append(min_dist)
        all_distances = distances_pred_to_gt + distances_gt_to_pred
        hd95 = np.percentile(all_distances, 95)
    except Exception as e:
        print(f"HD95 calculation error: {e}")
        hd95 = 100.0
    return hd95


def calculate_rad(prediction, ground_truth):
    pred_binary = (prediction > 0.5).astype(np.float32)
    gt_binary = (ground_truth > 0.5).astype(np.float32)
    pred_area = np.sum(pred_binary)
    gt_area = np.sum(gt_binary)
    if gt_area > 0:
        rad = abs(pred_area - gt_area) / gt_area
    else:
        rad = 1.0 if pred_area > 0 else 0.0

    return rad


def show_image_with_all_metrics(predict_save, labs, save_path):
    if len(labs.shape) > 2:
        labs = labs.squeeze()
    if len(predict_save.shape) > 2:
        predict_save = predict_save.squeeze()

    tmp_lbl = (labs > 0.5).astype(np.float32)
    tmp_pred = (predict_save > 0.5).astype(np.float32)
    intersection = np.sum(tmp_lbl * tmp_pred)
    union = np.sum(tmp_lbl) + np.sum(tmp_pred)
    dice_pred = (2.0 * intersection + 1e-5) / (union + 1e-5)
    dice_pred = min(max(dice_pred, 0.0), 1.0)
    union_pixels = np.sum((tmp_lbl + tmp_pred) > 0)
    if union_pixels > 0:
        iou_pred = intersection / (union_pixels + 1e-5)
    else:
        iou_pred = 1.0 if np.sum(tmp_lbl) == 0 else 0.0
    recall = calculate_recall(predict_save, labs)
    hd95 = calculate_hd95(predict_save, labs)
    rad = calculate_rad(predict_save, labs)
    return {
        'dice': dice_pred,
        'iou': iou_pred,
        'recall': recall,
        'hd95': hd95,
        'rad': rad
    }


def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens,
                         save_visualization=True, save_pickle=True):
    model.eval()
    with torch.no_grad():
        output = model(input_img.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    metrics = show_image_with_all_metrics(predict_save, labs,
                                          save_path=vis_save_path + '_predict' + model_type + '.png')
    input_img_vis = input_img[0].transpose(0, -1).cpu().detach().numpy()
    labs_vis = labs[0].squeeze()
    output_vis = output[0, 0, :, :].cpu().detach().numpy()
    if input_img_vis.shape[-1] == 3:
        input_img_display = input_img_vis
    else:
        input_img_display = input_img_vis.squeeze()

    input_img_display = (input_img_display - input_img_display.min()) / (
                input_img_display.max() - input_img_display.min() + 1e-8)
    if save_pickle:
        pickle.dump({
            'input': input_img_vis,
            'output': (output_vis >= 0.5) * 1.0,
            'ground_truth': labs_vis,
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'recall': metrics['recall'],
            'hd95': metrics['hd95'],
            'rad': metrics['rad'],
            'output_prob': output_vis
        },
            open(vis_save_path + '.p', 'wb'))

    if save_visualization:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        if len(input_img_display.shape) == 3:
            axes[0].imshow(input_img_display)
        else:
            axes[0].imshow(input_img_display, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(labs_vis, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        pred_binary = (output_vis >= 0.5) * 1.0
        axes[2].imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        im = axes[3].imshow(output_vis, cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title('Probability Heatmap', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        metrics_text = (f'Dice: {metrics["dice"]:.4f} | IoU: {metrics["iou"]:.4f} | '
                        f'Recall: {metrics["recall"]:.4f} | HD95: {metrics["hd95"]:.2f} | RAD: {metrics["rad"]:.4f}')
        fig.suptitle(metrics_text, fontsize=12, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        plt.savefig(vis_save_path + '.png', dpi=300, bbox_inches='tight')
        plt.close()
    return metrics


if __name__ == '__main__':
    model_type = config.model_name
    test_session = config.test_session
    SAVE_VISUALIZATION = True
    SAVE_PICKLE = True
    SAVE_EVERY_N = 1
    print("=== Checking Dataset Path ===")
    original_test_path = config.test_dataset
    corrected_test_path = check_dataset_path(original_test_path)

    if corrected_test_path is None:
        print("\n❌ Error: Unable to find test dataset!")
        print("Please check the following:")
        print("1. Ensure ACDC dataset has been correctly downloaded and extracted")
        print("2. Check the test_dataset path setting in Config.py file")
        print("3. Ensure dataset directory structure is correct")
        print("\nSuggested directory structure:")
        print("datasets/")
        print("└── ACDC_split/")
        print("    └── test/")
        print("        └── slices/")
        print("            ├── img/")
        print("            └── mask/")
        exit(1)

    if corrected_test_path != original_test_path:
        print(f"\n⚠️  Note: Using corrected dataset path")
        print(f"Original path: {original_test_path}")
        print(f"New path: {corrected_test_path}")
        config.test_dataset = corrected_test_path

    if config.task_name.startswith("GlaS_exp"):
        expected_test_num = 80
    elif config.task_name.startswith("ISIC18_exp") or config.task_name == "ISIC2018":
        expected_test_num = 518
    elif config.task_name.startswith("Clinic_exp"):
        expected_test_num = 122
    elif config.task_name.startswith("BUSI_exp"):
        expected_test_num = 130
    elif config.task_name.startswith("Covid_exp"):
        expected_test_num = 20
    elif config.task_name == "Kvasir-SEG" or config.task_name.startswith("Kvasir"):
        expected_test_num = 200
    elif config.task_name.startswith("ACDC"):
        expected_test_num = 50
    elif config.task_name.startswith("PH2_exp"):
        expected_test_num = 40
    elif config.task_name.startswith("FIVES_exp") or config.task_name.startswith("FIVES"):
        expected_test_num = 160
    elif config.task_name.startswith("Synapse") or config.task_name == "Synapse":
        expected_test_num = 30
    else:
        print(f"Warning: Unknown task name '{config.task_name}', will dynamically determine test set size")
        expected_test_num = None

    try:
        model_path, save_path = find_latest_model(config.task_name, model_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure model training is completed and best model file is saved")
        exit(1)
    print(f"Using model path: {model_path}")
    vis_path = save_path + 'visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    pickle_path = os.path.join(vis_path, 'pickle_data')
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    checkpoint = torch.load(model_path, map_location='cuda')
    fp = open(save_path + 'test.result', 'a')
    fp.write(str(datetime.now()) + '\n')

    if model_type == 'ACC_UNet':
        config_vit = config.get_CTranS_config()
        model = ACC_UNet(n_channels=config.n_channels, n_classes=config.n_labels, n_filts=config.n_filts)
    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    elif model_type == 'UNet_base':
        config_vit = config.get_CTranS_config()
        model = UNet_base(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'SwinUnet':
        model = SwinUnet()
        model.load_from()
    elif model_type == 'SMESwinUnet':
        model = SMESwinUnet(n_channels=config.n_channels, n_classes=config.n_labels)
        model.load_from()
    elif model_type.split('_')[0] == 'MultiResUnet1':
        model = MultiResUnet(n_channels=config.n_channels, n_classes=config.n_labels,
                             nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))
    elif model_type == 'MMP_Net':
        model = MMP_Net(n_channels=config.n_channels, n_classes=config.n_labels, n_filts=config.n_filts)
    else:
        raise TypeError('Please enter a valid model type name')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    state_dict = checkpoint['state_dict']
    keys_to_remove = [key for key in state_dict.keys() if 'total_ops' in key or 'total_params' in key]
    for key in keys_to_remove:
        del state_dict[key]

    if model_type == 'SwinACC_UNet':
        corrected_state_dict = {}
        for key, value in state_dict.items():
            if 'phase_gen' in key:
                new_key = key.replace('phase_gen', 'phase')
                corrected_state_dict[new_key] = value
                print(f"Key mapping: {key} -> {new_key}")
            else:
                corrected_state_dict[key] = value
        try:
            model.load_state_dict(corrected_state_dict, strict=False)
            print('Model loaded successfully! (using corrected key names)')
        except Exception as e:
            print(f"Still failed after trying to correct key names: {e}")
            model.load_state_dict(state_dict, strict=False)
            print('Model loaded successfully! (ignoring mismatched keys)')
    else:
        model.load_state_dict(state_dict)
    print('Model loaded successfully!')

    print("=== Setting up test data loader ===")
    try:
        tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
        test_path = config.test_dataset
        possible_structures = [
            {'img_dir': 'images', 'mask_dir': 'masks'},
            {'img_dir': 'images', 'mask_dir': 'labels'},
            {'img_dir': 'images', 'mask_dir': 'labelcol'},
            {'img_dir': 'img', 'mask_dir': 'mask'},
            {'img_dir': 'img', 'mask_dir': 'label'},
            {'img_dir': 'img', 'mask_dir': 'labelcol'},     
            {'img_dir': 'trainx', 'mask_dir': 'trainy'},
            {'img_dir': 'trainx', 'mask_dir': 'labelcol'}, 
        ]
        dataset_structure = None
        for structure in possible_structures:
            img_path = os.path.join(test_path, structure['img_dir'])
            mask_path = os.path.join(test_path, structure['mask_dir'])
            if os.path.exists(img_path) and os.path.exists(mask_path):
                dataset_structure = structure
                print(f"Detected dataset structure: image_dir={structure['img_dir']}, mask_dir={structure['mask_dir']}")
                break
        if dataset_structure is None:
            if any(f.endswith('.npz') for f in os.listdir(test_path)):
                print("Detected Synapse dataset (.npz files), using SynapseDataset loader")


                class SynapseDataset:
                    def __init__(self, root_path, transform, image_size=224):
                        self.root_path = root_path
                        self.transform = transform
                        self.image_size = image_size
                        all_files = [f for f in os.listdir(root_path) if f.endswith('.npz')]
                        self.npz_files = sorted(all_files)
                        print(f"Synapse dataset loaded: found {len(self.npz_files)} test files")
                        print("First 5 files:")
                        for i, fname in enumerate(self.npz_files[:5]):
                            print(f"  {i + 1}. {fname}")

                    def __len__(self):
                        return len(self.npz_files)

                    def __getitem__(self, idx):
                        npz_file = self.npz_files[idx]
                        npz_path = os.path.join(self.root_path, npz_file)
                        try:
                            data = np.load(npz_path)
                            image = data['image']
                            label = data['label']
                            if len(np.unique(label)) > 2:
                                label = (label > 0).astype(np.float32)
                            else:
                                label = label.astype(np.float32)
                        except Exception as e:
                            print(f"Error loading file {npz_file}: {e}")
                            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
                            label = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                        if len(image.shape) == 2:
                            image = np.stack([image, image, image], axis=-1)
                        elif len(image.shape) == 3 and image.shape[2] == 1:
                            image = np.repeat(image, 3, axis=2)
                        if len(label.shape) == 3:
                            if label.shape[2] == 1:
                                label = label[:, :, 0]
                            else:
                                label = (np.sum(label, axis=2) > 0).astype(np.float32)
                        image = cv2.resize(image, (self.image_size, self.image_size))
                        label = cv2.resize(label, (self.image_size, self.image_size))
                        if image.max() > 1.0:
                            image = image.astype(np.float32) / 255.0
                        else:
                            image = image.astype(np.float32)
                        if label.max() > 1.0:
                            label = (label > 0).astype(np.float32)
                        else:
                            label = (label > 0.5).astype(np.float32)
                        image = image.transpose(2, 0, 1)
                        label = np.expand_dims(label, axis=0)

                        return {
                            'image': image,
                            'label': label
                        }, [npz_file]


                test_dataset = SynapseDataset(test_path, tf_test, image_size=config.img_size)
            elif os.path.exists(os.path.join(test_path, 'images')) and os.path.exists(
                    os.path.join(test_path, 'labels')):
                print("Detected images and labels directories, using PH2Dataset loader")
                from Load_Dataset import PH2Dataset

                test_dataset = PH2Dataset(test_path, tf_test, image_size=config.img_size)
            elif any(f.endswith('.h5') for f in os.listdir(test_path)):
                print("Detected ACDC dataset (.h5 files), using ACDCDataset loader")
                from Load_Dataset import ACDCDataset

                test_dataset = ACDCDataset(test_path, tf_test, image_size=config.img_size)
            else:
                print("Using traditional image dataset loader")
                test_dataset = ImageToImage2D(test_path, tf_test, image_size=config.img_size)
        else:
            if dataset_structure['img_dir'] == 'images' and dataset_structure['mask_dir'] == 'masks':
                print("Detected FIVES dataset structure, using custom loader directly")


                class FivesDataset:
                    def __init__(self, root_path, transform, image_size=256):
                        self.root_path = root_path
                        self.transform = transform
                        self.image_size = image_size
                        self.img_dir = os.path.join(root_path, 'images')
                        self.mask_dir = os.path.join(root_path, 'masks')
                        self.img_names = sorted([f for f in os.listdir(self.img_dir)
                                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                        print(f"FIVES dataset loaded: found {len(self.img_names)} images")

                    def __len__(self):
                        return len(self.img_names)

                    def __getitem__(self, idx):
                        img_name = self.img_names[idx]
                        img_path = os.path.join(self.img_dir, img_name)
                        mask_path = os.path.join(self.mask_dir, img_name)
                        import cv2
                        image = cv2.imread(img_path)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if image is None or mask is None:
                            raise ValueError(f"Cannot read image or label: {img_name}")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if self.transform:
                            import torch
                            from torchvision import transforms
                            image = cv2.resize(image, (self.image_size, self.image_size))
                            mask = cv2.resize(mask, (self.image_size, self.image_size))

                            if hasattr(self.transform, '__call__'):
                                try:
                                    transformed = self.transform(image, mask)
                                    if isinstance(transformed, tuple):
                                        image, mask = transformed
                                except:
                                    pass
                        else:
                            image = cv2.resize(image, (self.image_size, self.image_size))
                            mask = cv2.resize(mask, (self.image_size, self.image_size))
                        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
                        mask = mask.astype(np.float32) / 255.0
                        mask = np.expand_dims(mask, axis=0)

                        return {
                            'image': image,
                            'label': mask
                        }, [img_name]


                test_dataset = FivesDataset(test_path, tf_test, image_size=config.img_size)

            elif dataset_structure['img_dir'] == 'images' and dataset_structure['mask_dir'] == 'labels':
                print("Detected PH2 dataset structure (images/labels), using PH2Dataset loader")
                from Load_Dataset import PH2Dataset

                test_dataset = PH2Dataset(test_path, tf_test, image_size=config.img_size)
            elif dataset_structure['img_dir'] == 'trainx' and dataset_structure['mask_dir'] == 'trainy':
                print("Detected PH2 dataset structure (trainx/trainy), using PH2Dataset loader")
                from Load_Dataset import PH2Dataset

                test_dataset = PH2Dataset(test_path, tf_test, image_size=config.img_size)
            else:
                print("Using traditional image dataset loader")
                test_dataset = ImageToImage2D(test_path, tf_test, image_size=config.img_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"✓ Data loader created successfully")
    except Exception as e:
        print(f"❌ Error creating data loader: {e}")

        print(f"Test path: {config.test_dataset}")
        if os.path.exists(config.test_dataset):
            print("Directory contents:")
            for item in os.listdir(config.test_dataset):
                item_path = os.path.join(config.test_dataset, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    try:
                        sub_items = os.listdir(item_path)[:5]
                        for sub_item in sub_items:
                            print(f"    📄 {sub_item}")
                        if len(os.listdir(item_path)) > 5:
                            print(f"    ... {len(os.listdir(item_path)) - 5} more files")
                    except:
                        pass
                else:
                    print(f"  📄 {item}")
        exit(1)
    test_num = len(test_loader)
    if expected_test_num is not None:
        print(f"Expected test samples: {expected_test_num}")
    print(f"Actual test samples in data loader: {test_num}")
    if test_num == 0:
        print("❌ Error: Test dataset is empty! Please check dataset path and file format")
        exit(1)
    dice_scores = []
    iou_scores = []
    recall_scores = []
    hd95_scores = []
    rad_scores = []
    print(f"Starting testing and visualization, save path: {vis_path}")
    print(f"Computing metrics: Dice, IoU, Recall, HD95, RAD")
    with tqdm(total=test_num, desc='Testing visualization', unit='img', ncols=120, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            input_img = torch.from_numpy(arr)
            metrics = vis_and_save_heatmap(
                model, input_img, None, lab,
                os.path.join(vis_path, f'sample_{i:04d}'),
                dice_pred=0, dice_ens=0,
                save_visualization=SAVE_VISUALIZATION,
                save_pickle=SAVE_PICKLE
            )
            dice_scores.append(metrics['dice'])
            iou_scores.append(metrics['iou'])
            recall_scores.append(metrics['recall'])
            hd95_scores.append(metrics['hd95'])
            rad_scores.append(metrics['rad'])
            pbar.set_postfix({
                'Dice': f'{metrics["dice"]:.3f}',
                'IoU': f'{metrics["iou"]:.3f}',
                'Recall': f'{metrics["recall"]:.3f}',
                'HD95': f'{metrics["hd95"]:.1f}',
                'RAD': f'{metrics["rad"]:.3f}',
                'AvgD': f'{np.mean(dice_scores):.3f}',
                'AvgR': f'{np.mean(recall_scores):.3f}'
            })
            torch.cuda.empty_cache()
            pbar.update()
    metrics_stats = {
        'dice': {'scores': dice_scores, 'mean': np.mean(dice_scores), 'std': np.std(dice_scores)},
        'iou': {'scores': iou_scores, 'mean': np.mean(iou_scores), 'std': np.std(iou_scores)},
        'recall': {'scores': recall_scores, 'mean': np.mean(recall_scores), 'std': np.std(recall_scores)},
        'hd95': {'scores': hd95_scores, 'mean': np.mean(hd95_scores), 'std': np.std(hd95_scores)},
        'rad': {'scores': rad_scores, 'mean': np.mean(rad_scores), 'std': np.std(rad_scores)}
    }
    print(f"\n=== Test Results Statistics ===")
    print(f"Test samples: {len(dice_scores)}")
    print(f"Average Dice score: {metrics_stats['dice']['mean']:.4f} ± {metrics_stats['dice']['std']:.4f}")
    print(f"Average IoU score: {metrics_stats['iou']['mean']:.4f} ± {metrics_stats['iou']['std']:.4f}")
    print(f"Average Recall score: {metrics_stats['recall']['mean']:.4f} ± {metrics_stats['recall']['std']:.4f}")
    print(f"Average HD95 distance: {metrics_stats['hd95']['mean']:.2f} ± {metrics_stats['hd95']['std']:.2f} (pixels)")
    print(f"Average RAD: {metrics_stats['rad']['mean']:.4f} ± {metrics_stats['rad']['std']:.4f}")
    with open(os.path.join(vis_path, 'detailed_statistics.txt'), 'w') as f:
        f.write(f"Test date: {datetime.now()}\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Task name: {config.task_name}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Dataset path: {config.test_dataset}\n")
        f.write(f"Test samples: {len(dice_scores)}\n\n")
        for metric_name, stats in metrics_stats.items():
            f.write(f"=== {metric_name.upper()} ===\n")
            f.write(f"Mean: {stats['mean']:.6f}\n")
            f.write(f"Std: {stats['std']:.6f}\n")
            f.write(f"Min: {np.min(stats['scores']):.6f}\n")
            f.write(f"Max: {np.max(stats['scores']):.6f}\n")
            f.write(f"Median: {np.median(stats['scores']):.6f}\n\n")
        f.write(f"Detailed score records:\n")
        f.write(f"{'Sample':<8} {'Dice':<8} {'IoU':<8} {'Recall':<8} {'HD95':<8} {'RAD':<8}\n")
        f.write("-" * 60 + "\n")
        for idx in range(len(dice_scores)):
            f.write(f"{idx + 1:04d}     {dice_scores[idx]:.4f}   {iou_scores[idx]:.4f}   "
                    f"{recall_scores[idx]:.4f}   {hd95_scores[idx]:.2f}    {rad_scores[idx]:.4f}\n")
    plt.figure(figsize=(15, 10))
    metrics_to_plot = [
        ('Dice Score', dice_scores, 'blue'),
        ('IoU Score', iou_scores, 'green'),
        ('Recall', recall_scores, 'red'),
        ('HD95 Distance', hd95_scores, 'purple'),
        ('RAD', rad_scores, 'orange')
    ]
    for i, (name, scores, color) in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        plt.hist(scores, bins=20, alpha=0.7, color=color, edgecolor='black')
        mean_val = np.mean(scores)
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_val:.3f}')
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.title(f'{name} Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 6)
    plt.axis('off')
    table_data = []
    for metric_name, stats in metrics_stats.items():
        table_data.append([metric_name.upper(), f"{stats['mean']:.4f}", f"{stats['std']:.4f}"])

    table = plt.table(cellText=table_data,
                      colLabels=['Metric', 'Mean', 'Std'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Metrics Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_path, 'comprehensive_metrics_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    fp.write(f"\n=== Comprehensive Evaluation Metrics Results ===\n")
    fp.write(f"Average Dice score: {metrics_stats['dice']['mean']:.4f} ± {metrics_stats['dice']['std']:.4f}\n")
    fp.write(f"Average IoU score: {metrics_stats['iou']['mean']:.4f} ± {metrics_stats['iou']['std']:.4f}\n")
    fp.write(f"Average Recall score: {metrics_stats['recall']['mean']:.4f} ± {metrics_stats['recall']['std']:.4f}\n")
    fp.write(f"Average HD95 distance: {metrics_stats['hd95']['mean']:.2f} ± {metrics_stats['hd95']['std']:.2f}\n")
    fp.write(f"Average RAD: {metrics_stats['rad']['mean']:.4f} ± {metrics_stats['rad']['std']:.4f}\n")
    fp.write(f"Visualization results saved in: {vis_path}\n")
    fp.close()
    print(f"\nVisualization results saved to: {vis_path}")
    print(f"Detailed statistics saved to: {os.path.join(vis_path, 'detailed_statistics.txt')}")
    print(
        f"Comprehensive metrics analysis plot saved to: {os.path.join(vis_path, 'comprehensive_metrics_analysis.png')}")
    print(f"All images are in PNG format with complete evaluation metric information")