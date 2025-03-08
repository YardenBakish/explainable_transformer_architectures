import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config

#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE_vis import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification

from helper_scripts.helper_functions import update_json
from attDatasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens


import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import matplotlib.pyplot as plt
import numpy as np
import json

MAPPER = {"llama_tiny_baseline":"Tiny-Llama", "llama_2_7b_baseline": "Llama-2-7b", "llama_2_7b_baseline2": "llama-2-7b-Quantized"}

INTERVAL = 5

with open("BAR_RES.json", 'r') as file:
    data = json.load(file)

group_labels = []  

grid_above = [[0 for i in range(3)] for j in range(3)]
grif_below = [[0 for i in range(3)] for j in range(3)]


i=0
for k in data:
    print(k)
    group_labels.append(MAPPER[k])
    grid_above[i] = [elem for elem  in data[k]["PE"]]
    grif_below[i] = [elem for elem  in data[k]["SEMANTIC"]]
    i+=1


bottom_lst = []
upper_lst  = []

for i in range(3):
    for j in range(3):
        upper_lst.append(grid_above[i][j])
        bottom_lst.append(grif_below[i][j])


positions = []
group_positions = [] 
for i in range(len(bottom_lst)):
    group_idx = i // 3
    within_idx = i % 3
    # Each group starts at position group_idx * 4
    # Within a group, bars are at positions 0, 1, 2
    # This creates a gap of 1 unit between groups
    positions.append(group_idx * 4 + within_idx)
    if within_idx == 1:
        group_positions.append(positions[-1])

group_labels2 = ["First", "Intermediate", "Penultimate"]
group_labels2 = group_labels
fig, ax = plt.subplots(figsize=(20, 15))
group_labels = ["First", "Intermediate", "Penultimate"]
bar_labels = group_labels * 3
ax.set_xticks(positions, bar_labels, rotation=45, ha='right', fontsize=31, fontweight='bold')

max_height = max([(b+u) for b, u in zip(bottom_lst, upper_lst)])
label_height = max_height * 1.05  # Position labels slightly above the tallest bar

for i, pos in enumerate(group_positions):
    ax.text(pos, label_height, group_labels2[i], ha='center', fontsize=32, fontweight='bold')

#colors = [(0,0,0.9), (0,0,0.95) ,(0,0,1.0)]  # Colors to repeat
#color_list = [colors[i % len(colors)] for i in range(len(bottom_lst))] 

ax.bar(positions, bottom_lst, label='AttnLRP',zorder=2)

    #pos_sums = [tensor.sum().item() for tensor in pos_relevance_trace]
ax.bar(positions,  upper_lst, bottom=bottom_lst,  label='PE Only' ,zorder=2)
#plt.xlabel('Layers')
#plt.ylabel('T(R)')
#ax.set_title('Conservation of Relevancy out of 100%\nMean across the IMDB dataset')
#ax.set_xticks([])
ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)
ax.tick_params(axis='y', labelsize=18)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.legend(loc='upper left',bbox_to_anchor=(-0.2, 1), markerscale=1,fontsize=35, frameon=False,  )


plt.savefig(f"conservation_firstVSlastLast_tiny.png", dpi=300, bbox_inches='tight')
