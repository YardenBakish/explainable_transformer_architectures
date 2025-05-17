import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config

#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification

from helper_scripts.helper_functions import update_json
from attDatasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens
from llama_engine import run_LRP

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
                        choices=['llama_2_7b', 'llama_tiny', ],
                        default = 'llama_tiny',
                       # required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--pe', action='store_true')
parser.add_argument('--rule_matmul', action='store_true')

parser.add_argument('--reform', action='store_true')

parser.add_argument('--quant', action='store_true')

parser.add_argument('--trained_model', type=str,)
parser.add_argument('--clamp', action='store_true')


parser.add_argument('--sequence-length', type=int,
                       )
parser.add_argument('--pe_only', action='store_true')
parser.add_argument('--sep_heads', action='store_true')



parser.add_argument('--no-padding', action='store_true')
parser.add_argument('--without-abs', action='store_true')
parser.add_argument('--single-norm', action='store_true')





args = parser.parse_args()

if args.sep_heads and args.pe == False:
    print("for sep_head you must include --pe")
    exit(1)
args.dataset = 'imdb'
#rgs.model_size = 'llama_tiny'

config.get_config(args, pert = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


#if args.model_size == 'llama_tiny': 
#    model_checkpoint = "finetuned_models/imdb/llama_tiny/baseline/checkpoint_0/pytorch_model.bin"
#if args.model_size == 'llama_2_7b':
#    model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline/best_checkpoint/pytorch_model.bin"
#    if args.variant == "baseline2":
#        model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline2/best_checkpoint/pytorch_model.bin"


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


model = LlamaForSequenceClassification.from_pretrained(args.model_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda")
model.to(device)


MAX_LEN = 512
BATCH_SIZE = 1

df = load_imdb()
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE, args.no_padding)

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()

# apply AttnLRP rules
attnlrp.register(model)
run_LRP(model,
       test_data_loader,
        tokenizer,
        isBinary=True,
         withPE = args.pe,
          reform=args.reform,
           pe_only = args.pe_only,
            withoutABS = args.without_abs,
             clamp = args.clamp,
             sep_heads = args.sep_heads,
             single_norm = args.single_norm,
             vis_mode = True,
             debug_mode = False,
             sample_num=4,
             rule_matmul = args.rule_matmul,
               )


'''
count = 0
for d in tqdm(test_data_loader):
    if count !=11:
        count+=1
        continue
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    
    input_embeds = model.get_input_embeddings()(input_ids)
    
    if args.pe:
        position_ids = torch.arange(
                0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
               dtype=torch.float32
            ).reshape(1, -1)
    

        #exit(1)
        if args.sep_heads:
            position_embeddings = []
            for i in range(model.config.num_hidden_layers):
                pe_tuple = model.get_input_pos_embeddings()(input_embeds, position_ids)
                pe1_a = pe_tuple[0].repeat(model.config.num_attention_heads, 1, 1).requires_grad_()
                pe1_b = pe_tuple[0].repeat(model.config.num_key_value_heads, 1, 1).requires_grad_()

                pe2_a = pe_tuple[1].repeat(model.config.num_attention_heads, 1, 1).requires_grad_()
                pe2_b = pe_tuple[1].repeat(model.config.num_key_value_heads, 1, 1).requires_grad_()

                #print(pe2.shape)
            
                position_embeddings.append((pe1_a,pe1_b,pe2_a,pe2_b))
          
            #position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids)[0] for i in range(model.config.num_hidden_layers)]
 
        else:
            position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for i in range(model.config.num_hidden_layers)]
            position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]
        
    
        outputs = model(
            inputs_embeds = input_embeds.requires_grad_(),
            position_embeddings = position_embeddings,
            sep_heads = args.sep_heads,
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
    
    #print("HERE")
    #exit(1)

    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
    if args.single_norm:
        pass
    else:
        relevance = relevance / relevance.abs().max()

    if args.pe:
        acc_relevancy = 0.
        num_pos_embed_layer = 4 if args.sep_heads else 2
        for pos_embed in position_embeddings:
            for i in range(num_pos_embed_layer):
                #print(pos_embed[i].grad.float().dtype)
                #print(pos_embed[i].dtype)
                if args.reform:
                    if args.without_abs:
                        curr_relevancy = torch.matmul(pos_embed[i].grad, pos_embed[i].transpose(-1, -2)).detach()
                    else:
                        curr_relevancy = torch.matmul(pos_embed[i].grad.abs(), pos_embed[i].transpose(-1, -2).abs()).detach()
                    curr_relevancy /= (2*curr_relevancy.shape[-1])
                       
                    curr_relevancy = curr_relevancy.float().sum(-1).cpu()[0]
                else:
                    curr_relevancy = pos_embed[i].grad.float().sum(-1).cpu()[0]
                
                #curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
                
                if args.single_norm:
                    pass
                else:
                    curr_relevancy = curr_relevancy / curr_relevancy.abs().max()

                if args.without_abs:
                    pass
                else:
                    curr_relevancy = curr_relevancy.abs()

                acc_relevancy += curr_relevancy

        
        if args.clamp:
            acc_relevancy = acc_relevancy.clamp(min=0)

        
        if args.single_norm == False:
            acc_relevancy = acc_relevancy / acc_relevancy.abs().max()
        if args.pe_only:
            relevance=acc_relevancy.detach()

        else:
            if args.reform:
                acc_relevancy = acc_relevancy / model.config.num_hidden_layers
            relevance+=acc_relevancy
        relevance = relevance / relevance.abs().max()
        #when we want to check only positional:
        #relevance = acc_relevancy


    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    #print(tokens)
    #exit(1)
    #tokens = []
    print("reached")
    pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')
    exit(1)

       '''