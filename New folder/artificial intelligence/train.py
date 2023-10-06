import numpy as np

import json
import torch
import torch.nn as nn
from torch.utils.data import dataset,dataloader
from brain2 import bag_of_words, tokenize

from brain import NeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)

all_words =[]
tags =[]
xy =[]


for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
# print(tags)

    for pattern in intent['patterns']:
        print(pattern)
        w = tokenize(pattern)
        all_words.extend(w)