import torch


DATASET ="MNIST"
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 100
MODEL_DIR = "models"
MODEL_NAME = "nn_model.pth"
DRY_RUN = False
LEARNING_RATE = 0.001
EPOCHS = 1
MOMENTUM = 0.9
N_INPUT = 784
N_CLASSES = 10

def print_config():
    print(f"DATASET: {DATASET}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"DEVICE: {DEVICE}")
    print(f"LOG_INTERVAL: {LOG_INTERVAL}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"MOMENTUM: {MOMENTUM}")
    print(f"N_INPUT: {N_INPUT}")
    print(f"N_CLASSES: {N_CLASSES}")


