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
                        default = 'llama_tiny',
                       # required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--pe', action='store_true')

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

#llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
llamaModel = LlamaForSequenceClassification.from_pretrained(PATH,  device_map="cuda",   attn_implementation="eager")

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

count = 0
for d in tqdm(test_data_loader):
    if count !=22:
        count+=1
        continue
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    
    input_embeds = model.get_input_embeddings()(input_ids)
    
    if args.pe:
        position_ids = torch.arange(
                0.0, 512.0, device=input_embeds.device,  requires_grad=True,
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
    
   

    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
    relevance = relevance / relevance.abs().max()

    if args.pe:
        acc_relevancy = 0.
        for pos_embed in position_embeddings:
            for i in range(1):
                curr_relevancy = pos_embed[i].grad.float().sum(-1).cpu()[0]
                curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
                acc_relevancy += curr_relevancy.abs()

        acc_relevancy = (acc_relevancy - acc_relevancy.min()) / (acc_relevancy.max() - acc_relevancy.min())
        relevance+=acc_relevancy
        relevance = relevance / relevance.abs().max()
        relevance = acc_relevancy


    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    #print(tokens)
    #exit(1)
    #tokens = []
    pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')


       