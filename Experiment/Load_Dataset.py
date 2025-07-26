import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
import PIL.Image as Image
import h5py
def random_rot_flip(image, label):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(label, Image.Image):
        label = np.array(label)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label
def random_rotate(image, label):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(label, Image.Image):
        label = np.array(label)
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.shape[0] == 3:  # CHW -> HWC
                image = image.transpose(1, 2, 0)
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)
        if image.shape[0] != self.output_size[0] or image.shape[1] != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1], 1) if len(
                image.shape) == 3 else (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1]),
                         order=3)
            label = zoom(label, (self.output_size[0] / label.shape[0], self.output_size[1] / label.shape[1]), order=0)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        label = torch.from_numpy(label).long()
        sample = {'image': image, 'label': label}
        return sample
class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.shape[0] == 3:  # CHW -> HWC
                image = image.transpose(1, 2, 0)
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        if image.shape[0] != self.output_size[0] or image.shape[1] != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1], 1) if len(
                image.shape) == 3 else (self.output_size[0] / image.shape[0], self.output_size[1] / image.shape[1]),
                         order=3)
            label = zoom(label, (self.output_size[0] / label.shape[0], self.output_size[1] / label.shape[1]), order=0)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        label = torch.from_numpy(label).long()
        sample = {'image': image, 'label': label}
        return sample
def to_long_tensor(pic):
    if isinstance(pic, Image.Image):
        pic = np.array(pic)
    img = torch.from_numpy(np.array(pic, np.uint8))
    return img.long()
def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images
class PH2Dataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_size=224):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_size = image_size        
        print(f"PH2 Dataset path: {dataset_path}")
        print(f"Dataset path exists: {os.path.exists(dataset_path)}")        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        self.image_path = os.path.join(dataset_path, 'trainx')
        self.label_path = os.path.join(dataset_path, 'trainy')        
        print(f"Image path: {self.image_path}")
        print(f"Label path: {self.label_path}")
        print(f"Image path exists: {os.path.exists(self.image_path)}")
        print(f"Label path exists: {os.path.exists(self.label_path)}")       
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label directory not found: {self.label_path}")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.image_files = []
        for ext in image_extensions:
            import glob
            self.image_files.extend(glob.glob(os.path.join(self.image_path, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.image_path, ext.upper())))
        self.label_files = []
        for ext in image_extensions:
            self.label_files.extend(glob.glob(os.path.join(self.label_path, ext)))
            self.label_files.extend(glob.glob(os.path.join(self.label_path, ext.upper())))        
        print(f"Found {len(self.image_files)} image files")
        print(f"Found {len(self.label_files)} label files")
        self.matched_pairs = []
        image_dict = {}
        label_dict = {}
        for img_file in self.image_files:
            name = os.path.splitext(os.path.basename(img_file))[0]
            image_dict[name] = img_file
        for lbl_file in self.label_files:
            original_name = os.path.splitext(os.path.basename(lbl_file))[0]
            if original_name.endswith('_lesion'):
                clean_name = original_name.replace('_lesion', '')
            else:
                clean_name = original_name
            label_dict[clean_name] = lbl_file
        for name in image_dict.keys():
            if name in label_dict:
                self.matched_pairs.append((image_dict[name], label_dict[name]))        
        print(f"Found {len(self.matched_pairs)} matched image-label pairs")        
        if len(self.matched_pairs) == 0:
            raise ValueError("No matched image-label pairs found")
        if len(self.matched_pairs) > 0:
            print("First few matched pairs:")
            for i, (img_path, lbl_path) in enumerate(self.matched_pairs[:3]):
                print(f"  {i+1}. Image: {os.path.basename(img_path)} -> Label: {os.path.basename(lbl_path)}")    
    def __len__(self):
        return len(self.matched_pairs)    
    def __getitem__(self, idx):
        image_path, label_path = self.matched_pairs[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to load label: {label_path}")
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        label = (label > 127).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)        
        return sample, os.path.basename(image_path)
class ACDCDataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_size=224):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_size = image_size       
        print(f"ACDC Dataset path: {dataset_path}")
        print(f"Dataset path exists: {os.path.exists(dataset_path)}")        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        self.h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
        self.h5_files.sort()       
        print(f"Found {len(self.h5_files)} .h5 files")
        if len(self.h5_files) > 0:
            print(f"First few files: {self.h5_files[:5]}")        
        if len(self.h5_files) == 0:
            raise ValueError(f"No .h5 files found in {dataset_path}")   
    def __len__(self):
        return len(self.h5_files)    
    def __getitem__(self, idx):
        h5_filename = self.h5_files[idx]
        h5_path = os.path.join(self.dataset_path, h5_filename)        
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'image' in f.keys():
                    image = f['image'][:]
                elif 'img' in f.keys():
                    image = f['img'][:]
                else:
                    keys = list(f.keys())
                    print(f"Available keys in {h5_filename}: {keys}")
                    image = f[keys[0]][:]              
                if 'label' in f.keys():
                    label = f['label'][:]
                elif 'mask' in f.keys():
                    label = f['mask'][:]
                elif 'gt' in f.keys():
                    label = f['gt'][:]
                else:
                    keys = list(f.keys())
                    if len(keys) > 1:
                        label = f[keys[1]][:]
                    else:
                        print(f"Warning: No label found in {h5_filename}, creating dummy label")
                        label = np.zeros_like(image)        
        except Exception as e:
            print(f"Error loading {h5_path}: {e}")
            raise e
        if len(image.shape) > 2:
            image = image[:, :, 0] if image.shape[2] == 1 else image
        if len(label.shape) > 2:
            label = label[:, :, 0] if label.shape[2] == 1 else label
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        label = label.astype(np.uint8)
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)        
        return sample, h5_filename
class FIVESDataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_size=224):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_size = image_size        
        print(f"FIVES Dataset path: {dataset_path}")
        print(f"Dataset path exists: {os.path.exists(dataset_path)}")        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        self.image_path = os.path.join(dataset_path, 'train', 'Original')
        self.label_path = os.path.join(dataset_path, 'train', 'Ground truth')       
        print(f"Image path: {self.image_path}")
        print(f"Label path: {self.label_path}")
        print(f"Image path exists: {os.path.exists(self.image_path)}")
        print(f"Label path exists: {os.path.exists(self.label_path)}")        
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label directory not found: {self.label_path}")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.image_files = []
        for ext in image_extensions:
            import glob
            self.image_files.extend(glob.glob(os.path.join(self.image_path, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.image_path, ext.upper())))
        self.label_files = []
        for ext in image_extensions:
            self.label_files.extend(glob.glob(os.path.join(self.label_path, ext)))
            self.label_files.extend(glob.glob(os.path.join(self.label_path, ext.upper())))        
        print(f"Found {len(self.image_files)} image files")
        print(f"Found {len(self.label_files)} label files")
        self.matched_pairs = []
        image_dict = {}
        label_dict = {}
        for img_file in self.image_files:
            name = os.path.splitext(os.path.basename(img_file))[0]
            image_dict[name] = img_file
        for lbl_file in self.label_files:
            name = os.path.splitext(os.path.basename(lbl_file))[0]
            label_dict[name] = lbl_file
        for name in image_dict.keys():
            if name in label_dict:
                self.matched_pairs.append((image_dict[name], label_dict[name]))        
        print(f"Found {len(self.matched_pairs)} matched image-label pairs")        
        if len(self.matched_pairs) == 0:
            print("No matched pairs found. Let's debug:")
            print("Image files (first 5):")
            for i, img_file in enumerate(self.image_files[:5]):
                print(f"  {os.path.basename(img_file)}")
            print("Label files (first 5):")
            for i, lbl_file in enumerate(self.label_files[:5]):
                print(f"  {os.path.basename(lbl_file)}")
            print("Image dict keys (first 5):")
            for i, key in enumerate(list(image_dict.keys())[:5]):
                print(f"  {key}")
            print("Label dict keys (first 5):")
            for i, key in enumerate(list(label_dict.keys())[:5]):
                print(f"  {key}")
            raise ValueError("No matched image-label pairs found")
        if len(self.matched_pairs) > 0:
            print("First few matched pairs:")
            for i, (img_path, lbl_path) in enumerate(self.matched_pairs[:3]):
                print(f"  {i+1}. Image: {os.path.basename(img_path)} -> Label: {os.path.basename(lbl_path)}")    
    def __len__(self):
        return len(self.matched_pairs)    
    def __getitem__(self, idx):
        image_path, label_path = self.matched_pairs[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to load label: {label_path}")
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        label = (label > 127).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)        
        return sample, os.path.basename(image_path)
class ImageToImage2D(Dataset):
    def __init__(self, dataset_path, transform=None, image_size=224):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        print(f"Image directory: {self.input_path}")
        print(f"Image directory exists: {os.path.exists(self.input_path)}")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Image directory not found: {self.input_path}")       
        self.mask_path = os.path.join(dataset_path, 'labelcol')
        print(f"Mask directory: {self.mask_path}")
        print(f"Mask directory exists: {os.path.exists(self.mask_path)}")
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_path}")
        self.images_list = os.listdir(self.input_path)
        print(f"Number of images found: {len(self.images_list)}")
        if len(self.images_list) > 0:
            print(f"First few images: {self.images_list[:5]}")
        mask_files = os.listdir(self.mask_path)
        print(f"Number of mask files found: {len(mask_files)}")
        if len(mask_files) > 0:
            print(f"First few masks: {mask_files[:5]}")
        self.image_to_mask = {}
        self.valid_images = []    
        for image_file in self.images_list:
            base_name = os.path.splitext(image_file)[0]
            mask_file_found = None
            possible_mask_names = [
                base_name + ".png",
                base_name + ".jpg", 
                base_name + ".jpeg",
                base_name + ".tif",
                base_name + ".tiff",
                base_name + ".bmp",
                base_name + "_lesion.bmp",
                base_name + "_lesion.png",
                base_name + "_lesion.jpg",
                base_name + "_lesion.jpeg",
                base_name + "_lesion.tif",
                base_name + "_lesion.tiff",
                base_name + "_mask.png",
                base_name + "_mask.jpg",
                base_name + "_label.png",
                base_name + "_label.jpg",
                base_name + "_gt.png",
                base_name + "_gt.jpg"
            ]            
            for possible_name in possible_mask_names:
                if possible_name in mask_files:
                    mask_file_found = possible_name
                    break           
            if mask_file_found:
                self.image_to_mask[image_file] = mask_file_found
                self.valid_images.append(image_file)
            else:
                print(f"Warning: No mask found for image {image_file}")
        self.images_list = self.valid_images
        print(f"Number of valid image-mask pairs: {len(self.images_list)}")        
        if len(self.images_list) == 0:
            print("Available mask files:")
            for mask_file in mask_files[:10]:
                print(f"  {mask_file}")
            raise ValueError("No valid image-mask pairs found")
    def __len__(self):
        return len(self.images_list)    
    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image_path = os.path.join(self.input_path, image_filename)      
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")      
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask_filename = self.image_to_mask[image_filename]
        mask_path = os.path.join(self.mask_path, mask_filename)
        mask = cv2.imread(mask_path, 0)  
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        sample = {'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)       
        return sample, image_filename
class SynapseDataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_size=224):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_size = image_size
        
        print(f"Synapse Dataset path: {dataset_path}")
        print(f"Dataset path exists: {os.path.exists(dataset_path)}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        self.npz_files = []
        for file in os.listdir(dataset_path):
            if file.endswith('.npz'):
                self.npz_files.append(file)
        
        self.npz_files.sort()
        
        print(f"Found {len(self.npz_files)} .npz files")
        if len(self.npz_files) > 0:
            print(f"First few files: {self.npz_files[:5]}")
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No .npz files found in {dataset_path}")    
    def __len__(self):
        return len(self.npz_files)
    def __getitem__(self, idx):
        npz_filename = self.npz_files[idx]
        npz_path = os.path.join(self.dataset_path, npz_filename)
        
        try:
            data = np.load(npz_path)
            if 'image' in data.files:
                image = data['image']
            else:
                raise ValueError(f"No 'image' key found in {npz_filename}")
            
            if 'label' in data.files:
                label = data['label']
            else:
                raise ValueError(f"No 'label' key found in {npz_filename}")
            
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            raise e
        if len(image.shape) > 2:
            image = image[:, :, 0] if image.shape[2] == 1 else image
        if len(label.shape) > 2:
            label = label[:, :, 0] if label.shape[2] == 1 else label
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        label = label.astype(np.uint8)

        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        
        return sample, npz_filename
class ImagesLabelsDataset(Dataset):
    def __init__(self, dataset_path, transform=None, image_size=224):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_size = image_size        
        print(f"Images/Labels Dataset path: {dataset_path}")
        print(f"Dataset path exists: {os.path.exists(dataset_path)}")        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        self.image_path = os.path.join(dataset_path, 'images')
        self.label_path = os.path.join(dataset_path, 'labels')        
        print(f"Image path: {self.image_path}")
        print(f"Label path: {self.label_path}")
        print(f"Image path exists: {os.path.exists(self.image_path)}")
        print(f"Label path exists: {os.path.exists(self.label_path)}")        
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image directory not found: {self.image_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label directory not found: {self.label_path}")
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.image_files = []
        for ext in image_extensions:
            import glob
            self.image_files.extend(glob.glob(os.path.join(self.image_path, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.image_path, ext.upper())))
        self.label_files = []
        for ext in image_extensions:
            self.label_files.extend(glob.glob(os.path.join(self.label_path, ext)))
            self.label_files.extend(glob.glob(os.path.join(self.label_path, ext.upper())))        
        print(f"Found {len(self.image_files)} image files")
        print(f"Found {len(self.label_files)} label files")
        self.matched_pairs = []
        image_dict = {}
        label_dict = {}
        for img_file in self.image_files:
            name = os.path.splitext(os.path.basename(img_file))[0]
            image_dict[name] = img_file
        for lbl_file in self.label_files:
            name = os.path.splitext(os.path.basename(lbl_file))[0]
            clean_name = name
            suffixes_to_remove = ['_mask', '_label', '_gt', '_lesion', '_segmentation']
            for suffix in suffixes_to_remove:
                if clean_name.endswith(suffix):
                    clean_name = clean_name.replace(suffix, '')
                    break
            label_dict[clean_name] = lbl_file
        for name in image_dict.keys():
            if name in label_dict:
                self.matched_pairs.append((image_dict[name], label_dict[name]))
            else:
                clean_name = name
                suffixes_to_remove = ['_img', '_image', '_photo']
                for suffix in suffixes_to_remove:
                    if clean_name.endswith(suffix):
                        clean_name = clean_name.replace(suffix, '')
                        break
                if clean_name in label_dict:
                    self.matched_pairs.append((image_dict[name], label_dict[clean_name]))        
        print(f"Found {len(self.matched_pairs)} matched image-label pairs")        
        if len(self.matched_pairs) == 0:
            print("No matched pairs found. Let's debug:")
            print("Image files (first 5):")
            for i, img_file in enumerate(self.image_files[:5]):
                print(f"  {os.path.basename(img_file)}")
            print("Label files (first 5):")
            for i, lbl_file in enumerate(self.label_files[:5]):
                print(f"  {os.path.basename(lbl_file)}")
            print("Image dict keys (first 5):")
            for i, key in enumerate(list(image_dict.keys())[:5]):
                print(f"  {key}")
            print("Label dict keys (first 5):")
            for i, key in enumerate(list(label_dict.keys())[:5]):
                print(f"  {key}")
            raise ValueError("No matched image-label pairs found")
        if len(self.matched_pairs) > 0:
            print("First few matched pairs:")
            for i, (img_path, lbl_path) in enumerate(self.matched_pairs[:3]):
                print(f"  {i+1}. Image: {os.path.basename(img_path)} -> Label: {os.path.basename(lbl_path)}")    
    def __len__(self):
        return len(self.matched_pairs)    
    def __getitem__(self, idx):
        image_path, label_path = self.matched_pairs[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to load label: {label_path}")
        image = cv2.resize(image, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if label.max() > 1:
            label = (label > 127).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)        
        return sample, os.path.basename(image_path)
def get_dataset_loader(dataset_path, transform=None, image_size=224):
    print(f"Detecting dataset structure: {dataset_path}")
    if (os.path.exists(os.path.join(dataset_path, 'img')) and
        os.path.exists(os.path.join(dataset_path, 'mask'))):
        print("Detected img/mask dataset structure, using ImgMaskDataset loader")
        class ImgMaskDataset(Dataset):
            def __init__(self, dataset_path, transform=None, image_size=224):
                self.dataset_path = dataset_path
                self.transform = transform
                self.image_size = image_size
                self.image_path = os.path.join(dataset_path, 'img')
                self.mask_path = os.path.join(dataset_path, 'mask')
                print(f"Image path: {self.image_path}")
                print(f"Mask path: {self.mask_path}")
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
                self.image_files = []
                for ext in image_extensions:
                    import glob
                    self.image_files.extend(glob.glob(os.path.join(self.image_path, ext)))
                    self.image_files.extend(glob.glob(os.path.join(self.image_path, ext.upper())))
                self.mask_files = []
                for ext in image_extensions:
                    self.mask_files.extend(glob.glob(os.path.join(self.mask_path, ext)))
                    self.mask_files.extend(glob.glob(os.path.join(self.mask_path, ext.upper())))
                print(f"Found {len(self.image_files)} image files")
                print(f"Found {len(self.mask_files)} mask files")
                self.matched_pairs = []
                image_dict = {}
                mask_dict = {}
                for img_file in self.image_files:
                    name = os.path.splitext(os.path.basename(img_file))[0]
                    image_dict[name] = img_file
                for mask_file in self.mask_files:
                    name = os.path.splitext(os.path.basename(mask_file))[0]
                    mask_dict[name] = mask_file
                for name in image_dict.keys():
                    if name in mask_dict:
                        self.matched_pairs.append((image_dict[name], mask_dict[name]))
                print(f"Found {len(self.matched_pairs)} matched image-mask pairs")
                if len(self.matched_pairs) == 0:
                    raise ValueError("No matched image-mask pairs found")
            def __len__(self):
                return len(self.matched_pairs)
            def __getitem__(self, idx):
                image_path, mask_path = self.matched_pairs[idx]
                import cv2
                import numpy as np
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to load mask: {mask_path}")
                image = cv2.resize(image, (self.image_size, self.image_size))
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.uint8)
                sample = {'image': image, 'label': mask}
                if self.transform:
                    sample = self.transform(sample)
                return sample, os.path.basename(image_path)
        return ImgMaskDataset(dataset_path, transform, image_size)
    elif (os.path.exists(os.path.join(dataset_path, 'images')) and
          os.path.exists(os.path.join(dataset_path, 'masks'))):
        print("Detected images/masks dataset structure, using ImagesMasksDataset loader")
        class ImagesMasksDataset(Dataset):
            def __init__(self, dataset_path, transform=None, image_size=224):
                self.dataset_path = dataset_path
                self.transform = transform
                self.image_size = image_size
                self.image_path = os.path.join(dataset_path, 'images')
                self.mask_path = os.path.join(dataset_path, 'masks')
                print(f"Image path: {self.image_path}")
                print(f"Mask path: {self.mask_path}")
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
                self.image_files = []
                for ext in image_extensions:
                    import glob
                    self.image_files.extend(glob.glob(os.path.join(self.image_path, ext)))
                    self.image_files.extend(glob.glob(os.path.join(self.image_path, ext.upper())))
                self.mask_files = []
                for ext in image_extensions:
                    self.mask_files.extend(glob.glob(os.path.join(self.mask_path, ext)))
                    self.mask_files.extend(glob.glob(os.path.join(self.mask_path, ext.upper())))
                print(f"Found {len(self.image_files)} image files")
                print(f"Found {len(self.mask_files)} mask files")
                self.matched_pairs = []
                image_dict = {}
                mask_dict = {}
                for img_file in self.image_files:
                    name = os.path.splitext(os.path.basename(img_file))[0]
                    image_dict[name] = img_file
                for mask_file in self.mask_files:
                    name = os.path.splitext(os.path.basename(mask_file))[0]
                    mask_dict[name] = mask_file
                for name in image_dict.keys():
                    if name in mask_dict:
                        self.matched_pairs.append((image_dict[name], mask_dict[name]))
                print(f"Found {len(self.matched_pairs)} matched image-mask pairs")
                if len(self.matched_pairs) == 0:
                    raise ValueError("No matched image-mask pairs found")
            def __len__(self):
                return len(self.matched_pairs)
            def __getitem__(self, idx):
                image_path, mask_path = self.matched_pairs[idx]
                import cv2
                import numpy as np
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to load mask: {mask_path}")
                image = cv2.resize(image, (self.image_size, self.image_size))
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.uint8)
                sample = {'image': image, 'label': mask}
                if self.transform:
                    sample = self.transform(sample)
                return sample, os.path.basename(image_path)
        return ImagesMasksDataset(dataset_path, transform, image_size)
    elif (os.path.exists(os.path.join(dataset_path, 'images')) and
          os.path.exists(os.path.join(dataset_path, 'labels'))):
        print("Detected images/labels dataset structure, using ImagesLabelsDataset loader")
        return ImagesLabelsDataset(dataset_path, transform, image_size)
    elif (os.path.exists(os.path.join(dataset_path, 'img')) and
          os.path.exists(os.path.join(dataset_path, 'labelcol'))):
        print("Detected traditional image dataset, using ImageToImage2D loader")
        return ImageToImage2D(dataset_path, transform, image_size)
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
    else:
        available_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        available_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
        error_msg = f"Unsupported dataset format: {dataset_path}\n"
        error_msg += f"Available subdirectories: {available_dirs}\n"
        error_msg += f"Available files: {available_files[:5]}{'...' if len(available_files) > 5 else ''}\n"
        error_msg += "Supported formats:\n"
        error_msg += "- Test set: img/mask directories\n"
        error_msg += "- FIVES after splitting: images/masks directories\n"
        error_msg += "- General: images/labels directories\n"
        error_msg += "- Traditional: img/labelcol directories\n"
        error_msg += "- FIVES original: train/Original and train/Ground truth directories\n"
        error_msg += "- PH2: trainx/trainy directories\n"
        error_msg += "- ACDC: .h5 files"
        raise ValueError(error_msg)
