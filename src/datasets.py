import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_list_dir, crop_list, transform=None):
        self.image_list_dir = image_list_dir
        self.transform = transform
        self.crop_list = crop_list

    def __len__(self):
        return len(self.image_list_dir)

    def __getitem__(self, idx):
        img_path = self.image_list_dir[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.crop(self.crop_list[idx])
            if self.transform:
                image = self.transform(image)
            label = -1
            return image, label
        except:
            return None, None




class FilteredDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
        # precompute only the good indices
        self.idxs = [
            i for i in range(len(base_ds))
            if (base_ds[i] is not None and base_ds[i][0] is not None)
        ]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        # map through the valid-idx list
        return self.base[self.idxs[i]]

# def collate_fn(batch):
#     # Filter out None entries
#     batch = [item for item in batch if item[0] is not None]
#     return torch.utils.data.default_collate(batch)

# class UnlabelDataModule(pl.LightningDataModule):
#     def __init__(self, image_list_dir, crop_list, batch_size=32, num_workers=os.cpu_count(), transform=None):
#         super().__init__()
#         self.image_list_dir = image_list_dir
#         self.crop_list = crop_list
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((128, 128)),
#             transforms.ToTensor(),
#         ])
#     def setup(self, stage=None):
#         self.dataset = UnlabeledImageDataset(
#             self.image_list_dir, 
#             self.crop_list, 
#             transform=self.transform)
#         #self.dataset = FilteredDataset(self.dataset)

#     def train_dataloader(self):
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#             drop_last=True,
#             pin_memory=True,
#             collate_fn=collate_fn  # Use custom collate_fn to handle None entries
#         )
#     def valid_dataloader(self):
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#             drop_last=False,
#             pin_memory=True,
#             collate_fn=collate_fn
#         )

def safe_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:            # every sample was filtered out
        return None          # <─ sentinel value
    return torch.utils.data.default_collate(batch)

# ---------- 2.  tiny DataLoader subclass that drops the sentinel ----------
class SkipNoneLoader(DataLoader):
    """DataLoader that silently skips batches that came back as None."""
    def __iter__(self):
        for batch in super().__iter__():
            if batch is None:    # collate_fn signalled an empty batch
                continue         # ← just start the next fetch cycle
            yield batch          # everything is fine → yield as usual

# ---------- 3.  use it inside your LightningDataModule ----------
class UnlabelDataModule(pl.LightningDataModule):
    def __init__(self, image_list_dir, crop_list,
                 batch_size=32, num_workers=os.cpu_count(), transform=None):
        super().__init__()
        self.image_list_dir = image_list_dir
        self.crop_list = crop_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.dataset = UnlabeledImageDataset(
            self.image_list_dir,
            self.crop_list,
            transform=self.transform
        )

    def train_dataloader(self):
        return SkipNoneLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=safe_collate,
        )

    def valid_dataloader(self):  # (Lightning uses `val_dataloader`, not `valid_dataloader`)
        return SkipNoneLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=safe_collate,
        )

class LabeledImageDataset(Dataset):
    def __init__(self, image_list, labels, transform=None, strict=True, debug=False):
        """
        image_list: list of np.ndarray or PIL.Image
        labels: list of int or other targets
        transform: torchvision.transforms
        strict: si es True, lanza error si algo falla; si es False, devuelve None
        debug: si es True, imprime información de cada imagen procesada
        """
        self.image_list = image_list
        self.labels = labels
        self.transform = transform
        self.strict = strict
        self.debug = debug

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.labels[idx]

        try:
            # Convertir a PIL si es numpy
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (255 * image).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(image).convert("RGB")

            # Validación de tipo
            if not isinstance(image, Image.Image):
                raise TypeError(f"[LabeledImageDataset] Expected PIL.Image or np.ndarray, got {type(image)}")

            if self.debug:
                print(f"[DEBUG] idx={idx} | type={type(image)} | mode={image.mode}")

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            if self.strict:
                raise e
            else:
                print(f"[WARNING] Error loading sample {idx}: {e}")
                return None  # o (dummy_tensor, dummy_label) si quieres reemplazar

class LabeledDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
    

class LatentFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label

    def __len__(self):
        return len(self.features)