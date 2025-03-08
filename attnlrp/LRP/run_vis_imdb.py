# [3 (Aust), 2 (Sigal) 4?  5! 8 9! 11! 13! 18! 19!!!! shows that it really focuses on actual review]
# [22 23 25!]

import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import os
#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification

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

save_dir = f'visualizations/{args.model_size}'
os.makedirs(save_dir, exist_ok=True)
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

count = 0
for k, d in enumerate(tqdm(test_data_loader)):

    if k ==30:
        break
    args.pe = False
    args.pe_only = False
    args.reform = False


    for j in range(4):
        args.pe = False
        args.pe_only = False
        args.reform = False
        ext = "baseline"
        if j ==1:
            args.pe = True
            ext = "PE+LRP"

        if j ==2:
            args.pe = True
            args.pe_only = False
            ext = "PE"

        if j == 3:
            args.pe = True
            args.reform = True
            ext = "LRP+PE+REFORM"



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

    

        relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
        relevance = relevance / relevance.abs().max()

        if args.pe:
            acc_relevancy = 0.
            for pos_embed in position_embeddings:
                for i in range(2):
                    #print(pos_embed[i].grad.float().dtype)
                    #print(pos_embed[i].dtype)
                    if args.reform:
                        curr_relevancy = torch.matmul(pos_embed[i].grad.abs(), pos_embed[i].transpose(-1, -2).abs()).detach()
                        curr_relevancy = curr_relevancy.float().sum(-1).cpu()[0]
                    else:
                        curr_relevancy = pos_embed[i].grad.float().sum(-1).cpu()[0]
                    curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
                    #print(curr_relevancy.requires_grad == True)
                    #exit(1)
                    acc_relevancy += curr_relevancy.abs()

            acc_relevancy = (acc_relevancy - acc_relevancy.min()) / (acc_relevancy.max() - acc_relevancy.min())
            if args.pe_only:
                relevance = acc_relevancy.detach()
            else:
                relevance+=acc_relevancy
                relevance = relevance / relevance.abs().max()
            #when we want to check only positional:
            #relevance = acc_relevancy


        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = clean_tokens(tokens)

        #print(tokens)
        #exit(1)
        #tokens = []
        print(f'{save_dir}/heatmap_{k}_{ext}.pdf')
        pdf_heatmap(tokens, relevance, path=f'{save_dir}/heatmap_{k}_{ext}.pdf', backend='xelatex')


       