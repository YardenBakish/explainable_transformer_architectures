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


INTERVAL = 5

def save_heatmap(values, tokens, figsize, title, save_path):
    fig, ax = plt.subplots(figsize=figsize)

    abs_max = abs(values).max()
    im = ax.imshow(values, cmap='bwr', vmin=-abs_max, vmax=abs_max)

    layers = np.arange(values.shape[-1])

    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(layers)
    ax.set_yticklabels(tokens)

    plt.title(title)
    plt.xlabel('Layers')
    plt.ylabel('Tokens')
    plt.colorbar(im)

    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def hidden_relevance_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    module.hidden_relevance = output.detach().cpu()



import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny'],
                        default = 'llama_tiny',
                       # required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--pe', action='store_true')
parser.add_argument('--reform', action='store_true')

parser.add_argument('--quant', action='store_true')

parser.add_argument('--trained_model', type=str,)

parser.add_argument('--sequence-length', type=int,
                       )
args = parser.parse_args()
args.dataset = 'imdb'
#rgs.model_size = 'llama_tiny'

config.get_config(args, pert = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


if args.model_size == 'llama_tiny': 
    model_checkpoint = "finetuned_models/imdb/llama_tiny/baseline/checkpoint_0/pytorch_model.bin"
if args.model_size == 'llama_2_7b':
    model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline/best_checkpoint/pytorch_model.bin"
    if args.variant == "baseline2":
        model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline2/best_checkpoint/pytorch_model.bin"


PATH = args.original_models

#print(PATH)
#exit(1)

tokenizer = AutoTokenizer.from_pretrained(PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)



if args.quant:
    llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, local_files_only = True, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
else:
    llamaModel = LlamaForSequenceClassification.from_pretrained(PATH,  device_map="cuda",   attn_implementation="eager")

#llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")

conf = llamaModel.config
conf.num_labels = 2
conf.pad_token_id = tokenizer.pad_token_id


model = LlamaForSequenceClassification.from_pretrained(model_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda")
model.to(device)


MAX_LEN = 512
BATCH_SIZE = 1

df = load_imdb()
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()

# apply AttnLRP rules
attnlrp.register(model)
for layer in model.model.layers:
    layer.register_full_backward_hook(hidden_relevance_hook)

count = 0
gloval_sums = []
gloval_pos_sums = []
for d in tqdm(test_data_loader):

    if count >=150:
        break
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    
    input_embeds = model.get_input_embeddings()(input_ids)
    
    if args.pe:
        position_ids = torch.arange(
                0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
               dtype=torch.float32
            ).reshape(1, -1)
    
        position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for i in range(model.config.num_hidden_layers)]
        position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]

        outputs = model(
            inputs_embeds = input_embeds.requires_grad_(),
            position_embeddings = position_embeddings,
            #input_ids=input_ids,
            use_cache=False,
            attention_mask=attention_mask
          )['logits']
    
    else:
        outputs = model(
            inputs_embeds = input_embeds.requires_grad_(),
            #position_embeddings = position_embeddings,
            #input_ids=input_ids,
            use_cache=False,
            attention_mask=attention_mask
          )['logits']
   
  
    max_logits, max_indices = torch.max(outputs, dim=1)
    max_logits.backward(max_logits)

    next_token_id = max_indices.item() 
    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision

    #next_token = tokenizer.convert_ids_to_tokens(next_token_id) 
    if next_token_id != targets.item():
        continue
    #print(next_token_id == targets.item())

    
    count+=1
    
    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
    #relevance = relevance / relevance.abs().max()

    relevance_trace = [relevance]
    pos_relevance_trace = []

    for i, layer in enumerate(model.model.layers):
       
        relevance = layer.hidden_relevance[0].sum(-1)
        if args.pe:
                if args.reform:
                    #print(position_embeddings[i][0].shape)
                    #print(position_embeddings[i][0].grad.shape)
                    #exit(1)
                    x1 = torch.matmul(position_embeddings[i][0].grad.abs(), position_embeddings[i][0].transpose(-1, -2).abs()).sum(-1).cpu()[0] / 2*position_embeddings[i][0].shape[1]
                    x2 = torch.matmul(position_embeddings[i][1].grad.abs(), position_embeddings[i][1].transpose(-1, -2).abs()).sum(-1).cpu()[0] / 2*position_embeddings[i][0].shape[1]
                else:
                    x1= position_embeddings[i][0].grad.float().sum(-1).cpu()[0].abs()
                    x2= position_embeddings[i][1].grad.float().sum(-1).cpu()[0].abs()

                    pos_relevance_trace.append((x1+x2))

                #print(curr_relevancy.shape)

        
        # normalize relevance at each layer between -1 and 1
        #relevance = relevance / relevance.abs().max()
       
        relevance_trace.append(relevance)
    
    pos_relevance_trace.append(torch.tensor(0.0))
    #sums = np.array([tensor.sum().item() for tensor in relevance_trace])
    #sums = (sums - sums.min()) / (sums.max() - sums.min())


    sums = [tensor.sum().item() for tensor in relevance_trace]
    pos_sums = [tensor.sum().item() for tensor in pos_relevance_trace]

    #pos_sums[0] = sum(pos_sums)

    gloval_sums.append(sums)
    gloval_pos_sums.append(pos_sums)



gloval_sums =  [sum(x) / len(x) for x in zip(*gloval_sums)]
if args.pe:

    gloval_pos_sums =  [sum(x) / len(x) for x in zip(*gloval_pos_sums)]


#gloval_sums=[gloval_sums[0],gloval_sums[-1]]
#gloval_pos_sums=[gloval_pos_sums[0],gloval_pos_sums[-1]]

gloval_sums = np.array(gloval_sums)
gloval_sums /= gloval_sums[-1]
gloval_pos_sums = np.array(gloval_pos_sums)
gloval_pos_sums /= model.config.num_hidden_layers
gloval_pos_sums = np.cumsum(gloval_pos_sums[::-1])[::-1]

gloval_pos_sums = gloval_pos_sums[[0, 10, 21]].tolist() 
gloval_sums     = gloval_sums[[0, 10, 21]].tolist() 

update_json(f'BAR_RES.json',{f"{args.model_size}_{args.variant}": {f"PE":gloval_pos_sums, "SEMANTIC": gloval_sums }} )
exit(1)



fig, ax = plt.subplots(figsize=(10, 6))

print(len(gloval_sums))
print(len(gloval_pos_sums))
ax.bar(range(len(gloval_sums)), gloval_sums, label='Conventional relevancy',zorder=2)
if args.pe:
    #pos_sums = [tensor.sum().item() for tensor in pos_relevance_trace]
    ax.bar(range(len(gloval_sums)), gloval_pos_sums, bottom=gloval_sums, label='PE relevancy' ,zorder=2)
#plt.xlabel('Layers')
#plt.ylabel('T(R)')
#ax.set_title('Conservation of Relevancy out of 100%\nMean across the IMDB dataset')
ax.set_xticks([])
ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)

for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.savefig(f"conservation_firstVSlast_{args.model_size}_{args.variant}.png", dpi=300, bbox_inches='tight')
#plt.savefig(f"conservation_firstVSlast_{args.model_size}_{args.variant}.png", dpi=300, bbox_inches='tight')

#relevance_trace = torch.stack(relevance_trace)
 

'''
categories = ['Top layer', 'First layer']
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(categories, gloval_sums, label='Conventional relevancy', width = 0.3, zorder=2)

ax.bar(categories, gloval_pos_sums, label='PE relevancy',bottom=gloval_sums, width = 0.3, zorder=2)


ax.set_title('Conservation of relevancy out of 100%\nMean across the IMDB dataset', fontsize=14)
ax.set_ylabel('Percentage', fontsize=12)
#ax.set_ylim(0, 120) 
ax.yaxis.grid(True, linestyle='-', alpha=0.5,zorder=0)
ax.set_xlim(-0.5, len(categories) - 0.5)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.legend(loc='lower center',  frameon=False,  # Remove the legend border
          handlelength=1,  # Shorter handle length for squares
          handleheight=1, bbox_to_anchor=(0.5, -0.15), ncol=2)
#ax.set_yticks(np.arange(0, 121, 20))
plt.tight_layout()
plt.savefig(f"conservation_firstVSlast_{args.model_size}_{args.variant}.png", dpi=300, bbox_inches='tight')
'''