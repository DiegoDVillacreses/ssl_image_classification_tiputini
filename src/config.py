import os
import torch

# General Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_WORKERS_SUPERVISED = 16
NUM_WORKERS_SIMCLR = 16
#dir_data_labeled = '/media/databases/tiputini/white-lipped-peccary-vs-collared-peccary/roboflow-yolov9-format-no-data-augmentation/'
dir_data_labeled = '/home/dvillacreses/datasets/tiputini/white-lipped-peccary-vs-collared-peccary/roboflow-yolov9-format-no-data-augmentation/'

CHECKPOINT_PATH = "/home/dvillacreses/simCLR_tiputini"
PATH_SUPERVISED_MODELS = '/home/dvillacreses/code/outputs'
PATH_OUTPUTS = '/home/dvillacreses/code/outputs'
PATH_RESULTS = '/home/dvillacreses/code/results'
#PATH_UNLABELED_METADATA = '/media/databases/tiputini/original_db'
PATH_UNLABELED_METADATA = '/home/dvillacreses/datasets/tiputini/original_db/'
PATH_RESULTS_GRAPHS = '/home/dvillacreses/code/results/graphs'
PATH_RESULTS_TABLES = '/home/dvillacreses/code/results/tables'

# General training configuration
MAX_EPOCHS_SUPERVISED = 500

TARGET_SIZE = (224,224)
BATCH_SIZE = 32
NUM_EPOCHS = 500
LEARNING_RATE = 10**-5
NUM_WORKERS = 20
CLASES = 2
PATIENCE = 30
RANDOM_STATE = 0