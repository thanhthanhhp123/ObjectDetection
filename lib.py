import os
import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import itertools
import math
from math import sqrt

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)