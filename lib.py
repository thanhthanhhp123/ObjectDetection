import os
import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)