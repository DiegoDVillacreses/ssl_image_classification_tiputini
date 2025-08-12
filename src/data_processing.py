import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import pandas as pd
import numpy as np
from PIL import Image
from utils import cropp
import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage, ToTensor
from src.config import TARGET_SIZE, PATH_UNLABELED_METADATA
from src.datasets import LatentFeatureDataset

import json

def torchdataset_to_dataframe(dataset):
    features, labels = [], []
    for x, y in dataset:
        features.append(x.numpy())  # assuming x is a tensor
        labels.append(y.numpy())    # assuming y is a tensor
    X = pd.DataFrame(features)
    y = pd.Series(labels, name='target')
    df = pd.concat([X, y], axis=1)
    return df

def latent_simclr_vit_torch_dataset(simclr_vit_model, datamodule, device):
    simclr_vit_model.eval()
    simclr_vit_model = simclr_vit_model.to(device)
    simclr_vit_model.vit.classifier = nn.Identity() 

    feats, labels = [], []
    for img, lab in datamodule:
        img = img.to(device) # send information to gpu for faster computing of simclr output
        #img_latent = simclr_vit_model.vit(img).logits
        img_latent = simclr_vit_model.vit(img).last_hidden_state[:, 0]
        img_latent = img_latent.detach().cpu() # detach from gpu to avoid collapsing vram
        labels.append(lab)
        feats.append(img_latent)

    labels = torch.cat(labels)
    feats = torch.cat(feats)

    dataset_labeled_latent = LatentFeatureDataset(
        features = feats,
        labels=labels
    )
    torch.cuda.empty_cache()
    return dataset_labeled_latent


def latent_dino_torch_dataset(model, datamodule, device):
    model.eval()
    model = model.to(device)

    feats, labels = [], []
    for img, lab in datamodule:
        img = img.to(device)
        with torch.no_grad():
            img_latent = model(img)
        img_latent = img_latent.detach().cpu() # detach from gpu to avoid collapsing vram
        labels.append(lab)
        feats.append(img_latent)

    labels = torch.cat(labels)
    feats = torch.cat(feats)

    dataset_labeled_latent = LatentFeatureDataset(
        features = feats,
        labels=labels
    )
    torch.cuda.empty_cache()
    return dataset_labeled_latent

def load_labeled_data(dir_data):

    # Get image and label file name
    train_images = os.listdir(dir_data + "/train/images")
    valid_images = os.listdir(dir_data + "/valid/images")
    test_images = os.listdir(dir_data + "/test/images")

    train_labels = os.listdir(dir_data + "/train/labels")
    valid_labels = os.listdir(dir_data + "/valid/labels")
    test_labels = os.listdir(dir_data + "/test/labels")

    train = ['train' for i in train_images]
    valid = ['valid' for i in valid_images]
    test = ['test' for i in test_images]

    df1 = pd.DataFrame({"image_file": train_images + valid_images + test_images,
                        "subset": train + valid + test})
    df2 = pd.DataFrame({"image_label": train_labels + valid_labels + test_labels,
                        "subset": train + valid + test})

    df1['name'] = df1['image_file'].str.replace('.jpg','')
    df2['name'] = df2['image_label'].str.replace('.txt','')

    df = pd.merge(left = df1,
            right=df2,
            on = ['name','subset'],
            how = 'inner')
    del df1, df2

    # Load image and crop it
    l = []
    for i in range(df.shape[0]):
        image_file, subset, name, image_label = df.iloc[i, :]
        dir_x = dir_data + "/" + subset + "/" + '/labels/' + image_label
        with open(dir_x) as f:
            lines = f.readlines()
            if len(lines)>1:
                lines = [i.replace('\n','') for i in lines]
        image = Image.open(dir_data + "/" + subset + "/" + '/images/' + image_file)
        image_np = np.array(image)
        if len(lines)==1:
            line_x = [np.float64(i) for i in lines[0].split(' ')]
            image_cropped_np = np.array(cropp(line_x[1:], image))
            image_shape = ','.join([str(i) for i in image_cropped_np.shape])
            l.append([subset, image_file,int(line_x[0]),image_cropped_np,image_shape])
        if len(lines)>1:
            for line in lines:
                line_x = [np.float64(i) for i in line.split(' ')]
                image_cropped_np = np.array(cropp(line_x[1:], image))
                image_shape = ','.join([str(i) for i in image_cropped_np.shape])
                l.append([subset, name,int(line_x[0]),image_cropped_np,image_shape])
    df = pd.DataFrame(l, columns=['subset','file_name', 'label', 'cropped_image','image_shape'])
    df.loc[df['label']==0,'label_text'] = 'Taypec'
    df.loc[df['label']==1,'label_text'] = 'Taytaj'
    print(f"{df.shape=}")
    for i in range(df.shape[0]):
        ci_shape = [j for j in df.loc[i,'cropped_image'].shape]
        ci_px = ci_shape[0]*ci_shape[1]
        df.loc[i,'cropped_image_px'] = ci_px
    return df


def df_undersampling_strat(df, subset_col, label_col, random_state=0):
    """
    Balances the label_text within each subset by downsampling to the minimum count per subset.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - subset_col (str): The name of the subset column (e.g., 'test', 'train', 'valid').
    - label_col (str): The name of the label column to balance.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - pd.DataFrame: The balanced DataFrame.
    """
    min_counts = df.groupby(subset_col)[label_col].value_counts().groupby(level=0).min()
    # print("Minimum counts per subset:")
    # print(min_counts)
    # print("\n")
    def sample_group(group):
        subset = group.name[0]
        label = group.name[1]
        n = min_counts[subset]
        sampled = group.sample(n=n, random_state=random_state)
        return sampled
    balanced_df = df.groupby([subset_col, label_col]).apply(sample_group).reset_index(drop=True)
    return balanced_df


train_transform_labeled = transforms.Compose([
    #transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.RandomResizedCrop(size=TARGET_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # If applicable
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value='random',
        inplace=False
    ),
])
val_transform_labeled = transforms.Compose([
    #transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



train_transform_labeled_vit = transforms.Compose([
    #transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.RandomResizedCrop(size=TARGET_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # If applicable
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value='random',
        inplace=False
    ),
])
val_transform_labeled_vit = transforms.Compose([
    #transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


def load_unlabeled_metadata():
    with open(os.path.join(PATH_UNLABELED_METADATA,'megadetector_metadata_modified_05_aug_2024.json'), 'r') as file:
        metadata = json.load(file)
    l_metadata = []
    for i in range(len(metadata)):
        mi = metadata[i]['detectors']['megadetectorV5']['output']
        if len(mi['detections'][0]['category'])>0:
            if mi['detections'][0]['category'][0]==0:
                l_metadata.append([mi['file'],len(mi['detections']), mi['detections'][0]['confidence'], mi['detections'][0]['bbox']])

    df_meta = pd.DataFrame(l_metadata,columns=['file', 'total_detections','confidence','bbox'])
    df_meta['len_bbox'] = [len(i) for i in df_meta['bbox']]
    df_meta['len_confidence'] = [len(i) for i in df_meta['confidence']]

    df_meta['confidence'] = [i[0] for i in df_meta['confidence']]
    df_meta['bbox'] = [i[0] for i in df_meta['bbox']]
    l = []
    for i in df_meta['file']:
        i = i.split("-")
        i = i[len(i)-1]
        l.append(i)
    df_meta['possible_animal_name']=l

    df_meta = df_meta[lambda x: x['confidence']>=0.85].reset_index(drop=True)

    files_list = df_meta['file'].to_list()
    bbox_list = df_meta['bbox'].to_list()
    return files_list, bbox_list

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2, image_size = 96):
        self.base_transforms = base_transforms
        self.n_views = n_views
        self.image_size = image_size

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    

# contrast_transforms = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(size=224),
#         transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.GaussianBlur(kernel_size=9),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ]
# )

class RandomGaussianNoise(object):
    """Add Gaussian noise to a tensor image."""
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            tensor = F.to_tensor(img)
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0., 1.)
            return F.to_pil_image(tensor)
        return img

# contrast_transforms = transforms.Compose([
#     # 1. Random crop + resize to 224×224, stronger scale range
#     transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3/4, 4/3)),
#     # 2. Horizontal flip
#     transforms.RandomHorizontalFlip(p=0.5),
#     # 3. Strong color jitter
#     transforms.RandomApply([
#         transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
#     ], p=0.8),
#     # 4. Random grayscale
#     transforms.RandomGrayscale(p=0.2),
#     # 5. Add gaussian blur with random sigma
#     transforms.RandomApply([
#         transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
#     ], p=1.0),
#     # 6. Random solarization (inverts pixels above threshold)
#     transforms.RandomApply([
#         transforms.RandomSolarize(threshold=0.5)
#     ], p=0.2),
#     # 7. Random adjust sharpness to sometimes counteract blur
#     transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
#     # 8. Add synthetic Gaussian noise
#     RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
#     # 9. Perspective transform to simulate geometric distortion
#     transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
#     # 10. To tensor + normalize with ImageNet stats
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                          std=(0.229, 0.224, 0.225)),
#     # 11. Random erasing to block out regions
#     transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
# ])

# contrast_transforms = transforms.Compose([
#     # 1. Random crop + resize to 224×224, stronger scale range
#     transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(3/4, 4/3)),
#     # 2. Horizontal flip
#     transforms.RandomHorizontalFlip(p=0.5),
#     # 3. Strong color jitter
#     transforms.RandomApply([
#         transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
#     ], p=0.6),
#     # 4. Random grayscale
#     transforms.RandomGrayscale(p=0.2),
#     # 5. Add gaussian blur with random sigma
#     transforms.RandomApply([
#         transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
#     ], p=0.8),
#     # 6. Random solarization (inverts pixels above threshold)
#     transforms.RandomApply([
#         transforms.RandomSolarize(threshold=0.5)
#     ], p=0.2),
#     # 7. Random adjust sharpness to sometimes counteract blur
#     transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
#     # 7.1. Random additional blur
#     transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.3),
#     # 8. Add synthetic Gaussian noise
#     RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
#     # 9. Perspective transform to simulate geometric distortion
#     transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
#     # → simple night-vision: occasional grayscale + darker + low contrast
#     transforms.RandomApply([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ColorJitter(brightness=(0.5, 0.8), contrast=(0.5, 0.8))
#     ], p=0.3),
#     # 10. To tensor + normalize with ImageNet stats
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                          std=(0.229, 0.224, 0.225)),
#     # 11. Random erasing to block out regions
#     transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
# ])

class RandomSharpnessBinary(torch.nn.Module):
    def __init__(self, factors=(0.5, 2.0), p=0.3):
        super().__init__()
        self.factors = factors
        self.p = p
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            factor = random.choice(self.factors)
            if isinstance(img, torch.Tensor):
                img = self.to_pil(img)
            img = F.adjust_sharpness(img, sharpness_factor=factor)
            return self.to_tensor(img)
        return img

contrast_transforms = transforms.Compose([
    # 1. Resize first (to ensure a base image size)
    transforms.Resize(TARGET_SIZE),

    # 2. Random crop + resize (applied 80% of the time)
    transforms.RandomApply([
        transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(3/4, 4/3))
    ], p=0.8),

    # 3. Horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # 4. Strong color jitter
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    ], p=0.4),

    # 5. Random grayscale
    transforms.RandomGrayscale(p=0.2),

    # 6. Gaussian blur
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    ], p=0.6),

    # 7. Solarization
    transforms.RandomApply([
        transforms.RandomSolarize(threshold=0.5)
    ], p=0.2),

    # 8. Sharpness correction AFTER blur/solarization but BEFORE tensor
    RandomSharpnessBinary(factors=(0.5, 2.0), p=0.3),

    # 9. Perspective distortion
    transforms.RandomPerspective(distortion_scale=0.5, p=0.2),

    # 10. Simulated night vision: grayscale + dim contrast
    transforms.RandomApply([
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=(0.5, 0.8), contrast=(0.5, 0.8))
    ], p=0.3),

    # 11. Convert to tensor
    transforms.ToTensor(),

    # 12. Add Gaussian noise (AFTER ToTensor)
    RandomGaussianNoise(mean=0.0, std=0.05, p=0.4),

    # 13. Normalize (ImageNet stats or your own stats)
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),

    # 14. Random erasing (patch dropout)
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
])


contrast_transforms = transforms.Compose([
    # 1. Resize first (to ensure a base image size)
    transforms.Resize(TARGET_SIZE),

    # 2. Random crop + resize (applied 80% of the time)
    transforms.RandomApply([
        transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(3/4, 4/3))
    ], p=0.8),

    # 3. Horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # 4. Strong color jitter
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    ], p=0.4),

    # 5. Random grayscale
    transforms.RandomGrayscale(p=0.2),

    # 6. Gaussian blur
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    ], p=0.6),

    # 7. Solarization
    transforms.RandomApply([
        transforms.RandomSolarize(threshold=0.5)
    ], p=0.2),

    # 8. Sharpness correction AFTER blur/solarization but BEFORE tensor
    RandomSharpnessBinary(factors=(0.5, 2.0), p=0.3),

    # 9. Perspective distortion
    transforms.RandomPerspective(distortion_scale=0.5, p=0.2),

    # 10. Simulated night vision: grayscale + dim contrast
    transforms.RandomApply([
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=(0.5, 0.8), contrast=(0.5, 0.8))
    ], p=0.3),

    # 11. Convert to tensor
    transforms.ToTensor(),

    # 12. Add Gaussian noise (AFTER ToTensor)
    RandomGaussianNoise(mean=0.0, std=0.05, p=0.4),

    # 13. Normalize (ImageNet stats or your own stats)
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5)),

    # 14. Random erasing (patch dropout)
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
])

val_transform_labeled_simclr = transforms.Compose([
    transforms.Resize((224, 224)),       # or whatever size ViT expects
    transforms.ToTensor(),               # PIL → FloatTensor [0,1]
    transforms.Normalize( mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
])

# def to_tensor_rgb_safe(pil_img):
#     # 1) Convert PIL→NumPy
#     arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
#     # 2) Build the tensor (float32, C×H×W, normalized)
#     tensor = torch.as_tensor(arr, dtype=torch.float32)   \
#                   .permute(2, 0, 1)                     \
#                   .div(255.0)
#     return tensor
    
# def to_tensor_rgb_safe(pil_img):
#     import numpy as np
#     import torch

#     # Convertir a RGB
#     pil_rgb = pil_img.convert("RGB")

#     # Forzar exactamente un np.ndarray sin subclases
#     arr = np.array(pil_rgb, dtype=np.uint8)
#     arr = np.asarray(arr, dtype=np.uint8).copy()  # <== esto garantiza np.ndarray base

#     # Convertir a tensor (C, H, W) y normalizar
#     tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div(255)
#     return tensor






# val_transform_labeled_simclr = transforms.Compose([
#     transforms.Lambda(safe_numpy_to_pil),
#     transforms.Resize((224, 224)),
#     transforms.Lambda(to_tensor_rgb_safe),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
# ])
