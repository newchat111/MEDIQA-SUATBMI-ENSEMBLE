from transformers import AutoModelForImageTextToText, AutoProcessor
import draccus
from dataclasses import dataclass
import json
import torch
from tqdm import tqdm

