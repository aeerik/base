import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchtext
import torchtext.vocab as vocab
import os
from datetime import date

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

print("Num GPUs Available: ", torch.cuda.device_count())

gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPU names: ", gpu_names)