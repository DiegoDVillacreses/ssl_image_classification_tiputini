import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import shutil
import random
import numpy as np
import os

import torch
import pytorch_lightning as pl

def seed_everything(seed: int = 0):
    # 1) Python builtin RNGs
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic= True

    # 5) PyTorch Lightning
    pl.seed_everything(seed, workers=True)
    


def delete_directory_recursively(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Directory {dir_path} deleted.")
    else:
        print(f"Directory {dir_path} not found.")

def cropp(l_norm, image):
    # Image dimensions
    W, H = image.size
    C_x, C_y, B_w, B_h = (l_norm[i] * W if i % 2 == 0 else l_norm[i] * H for i in range(4))
    T_x, T_y = C_x - (B_w / 2), C_y - (B_h / 2)
    bounding_box = (T_x, T_y, T_x + B_w, T_y + B_h)
    return image.crop(bounding_box)