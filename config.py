import torch
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.DATASET = "MNIST"
__C.BATCH_SIZE = 64
__C.NUM_WORKERS = 4
__C.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.LOG_INTERVAL = 100
__C.MODEL_DIR = "models"
__C.MODEL_NAME = "nn_model.pth"
__C.DRY_RUN = False
__C.LEARNING_RATE = 0.001
__C.EPOCHS = 1
__C.MOMENTUM = 0.9
__C.N_INPUT = 784
__C.N_CLASSES = 10
__C.N_LAYER_1 = 128
__C.N_LAYER_2 = 256
__C.LOG_DIR = "logs"
__C.PATIENCE = 3
__C.LOG_EVERY_N_STEPS = 100
__C.WANDB_PROJECT = "simple_nn"
__C.WANDB_ENTITY = "pytorch-lightning"

def print_config():
    for key in cfg:
        print(key, ":", cfg[key])



