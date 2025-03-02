import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config

#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification

from helper_scripts.helper_functions import update_json
from datasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
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


import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny'],
                       # required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--trained_model', type=str,)

parser.add_argument('--sequence-length', type=int,
                       )
args = parser.parse_args()
args.dataset = 'imdb'
args.model_size = 'llama_tiny'

config.get_config(args, pert = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

model_checkpoint = "finetuned_models/imdb/llama_tiny/baseline/checkpoint_0/pytorch_model.bin"

PATH = args.original_models



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

#llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda",  attn_implementation="eager")

conf = llamaModel.config
conf.num_labels = 2
conf.pad_token_id = tokenizer.pad_token_id


model = LlamaForSequenceClassification.from_pretrained(model_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda")
#params = torch.load(model_checkpoint, map_location=torch.device(device))
#model.config.pad_token_id = tokenizer.pad_token_id
#model.config.num_labels = 2
'''
model_keys = set(model.state_dict().keys())    # Keys from the new model
saved_keys = set(params.keys())  

print("Keys in saved file but not in model:", saved_keys - model_keys)
print("\n\n")
print("Keys in model but not in saved file:", model_keys - saved_keys)

if len(list(model_keys)) != len(list(saved_keys)):
    print("ERROR!")
    exit(1)
'''
'''
for key in model.state_dict().keys():
    if model.state_dict()[key].shape != params[key].shape:
        print(f"Shape mismatch for {key}:")
        print(f"Model shape: {model.state_dict()[key].shape}")
        print(f"Loaded shape: {params[key].shape}")
        exit(1)

'''
#model.load_state_dict(params)
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

count = 0
for d in tqdm(test_data_loader):
    if count !=11:
        count+=1
        continue
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    
    input_embeds = model.get_input_embeddings()(input_ids)
    
    #print("A")
    #print(input_embeds)
    #print("\n\n")
    #print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    outputs = model(
        inputs_embeds = input_embeds.requires_grad_(),
        #input_ids=input_ids,
        use_cache=False,
        attention_mask=attention_mask
      )['logits']
  
    #input_embeds = model.get_input_embeddings()(input_ids)
   
    #print("B")
    #print(input_embeds)
    #print("\n\n")
  
    max_logits, max_indices = torch.max(outputs, dim=1)
    print(max_indices)
    print(targets)

    
    max_logits.backward(max_logits)
    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision

    relevance = relevance / relevance.abs().max()
    

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    #print(tokens)
    #exit(1)
    #tokens = []
    pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')


       