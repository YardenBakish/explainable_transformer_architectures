import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import os
os.environ['TRANSFORMERS_CACHE'] = '/home/ai_center/ai_users/yardenbakish/'
os.environ['HF_HOME'] = '/home/ai_center/ai_users/yardenbakish/'
#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.bert_PE import BertForSequenceClassification, attnlrp
import matplotlib.pyplot as plt

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
import json
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 



import argparse
def parse_args():

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
    parser.add_argument('--mode', type=str,
                                choices=['generate', 'analyze', 'generate_analyze'],
                                required = True,
                                help='')
    
    args = parser.parse_args()



    args.work_dir = f'visualizations/conservation_scatter_bert_imdb'

    os.makedirs(args.work_dir, exist_ok=True)
    return args






def analyze(args):
    res_file =  f'{args.work_dir}/data.json'
    save_file = f'{args.work_dir}/conservation_plot.png'


    with open(res_file, 'r') as file:
        data = json.load(file)
    res = data
    a_values = res['logits']
    b_values = res['relevance']  
    c_values = res['relevance_plus_pos']   
    # Create a figure and axis  
    plt.figure(figsize=(10, 8)) 
    ax = plt.gca()
    plt.scatter(a_values, b_values,  s=5, label='LRP')
    plt.scatter(a_values, c_values,  s=5, label='LRP+PE')

    a_min, a_max = min(a_values), max(a_values)
    margin = (a_max - a_min) * 0.1  # 10% margin
    plt.xlim(0.0, a_max + margin)
    plt.ylim(0.0, a_max + margin)

    tick_count = 4
    tick_step = (a_max - -2) / (tick_count - 1)
    tick_positions = np.arange(0.0, a_max + tick_step, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    ax.tick_params(axis='x', labelsize=12)  # Very large font size for x-axis values
    ax.tick_params(axis='y', labelsize=12)  # Normal font size for y-axis values

    #plt.ylim(min(min(b_values), min(c_values)) - margin, max(max(b_values), max(c_values)) + margin)
    plt.xlabel(r'output $f$', fontsize=20)
    plt.ylabel(r'$\Sigma_{i} R(x_i)$', fontsize=20)


    line_range = np.linspace(0.0, a_max, 100)

    # Add the y=x line
    plt.plot(line_range, line_range, color='black', linestyle='-', linewidth=1)

    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_file)




def generate(args):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
    )


    #model = BertForSequenceClassification.from_pretrained("lvwerra/bert-imdb", device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage = True)
    model = BertForSequenceClassification.from_pretrained("lvwerra/bert-imdb", device_map="cuda",quantization_config=quantization_config, torch_dtype=torch.bfloat16)
    
    #model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/bert-imdb")

    # optional gradient checkpointing to save memory (2x forward pass)
    model.gradient_checkpointing_enable()

    # apply AttnLRP rules
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model.to(device)


    #attnlrp.register(model)

    MAX_LEN = 512
    BATCH_SIZE = 1
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    df = load_imdb()
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # optional gradient checkpointing to save memory (2x forward pass)
    model.gradient_checkpointing_enable()
    attnlrp.register(model)

    count = 0
    sem = []
    sem_with_pos = []
    logitsArr = []

    num_correct = 0
    max_count = 100
    for d in tqdm(test_data_loader):
        if count >=max_count:

            break
        count+=1

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)


        input_embeds = model.get_input_embeddings()(input_ids)

        position_ids = torch.arange(input_ids.shape[-1], device="cuda:0").view(1, -1)
        #print(model.config.max_position_embeddings)
        #print(model.config.hidden_size)
        position_embeddings = model.bert.embeddings.position_embeddings(position_ids).requires_grad_()

        outputs = model(
            inputs_embeds = input_embeds.requires_grad_(),
            position_embeddings = position_embeddings,
          )['logits']
    
    
        max_logits, max_indices = torch.max(outputs, dim=1)

        next_token_id = max_indices.item() 
      
    
        if 1-targets.item() == next_token_id:
            num_correct+=1

        max_logits.backward(max_logits)
      
        relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precisio
        
        sem_relevance = relevance.clone()
        #sem_relevance = sem_relevance / sem_relevance.abs().max()
        
        
        PE_relevance = position_embeddings.grad.float().sum(-1).cpu()[0]
        relevance+=PE_relevance
        #relevance = relevance / relevance.abs().max()

        #print(relevance.shape)
        model.zero_grad()

        sem_with_pos.append(relevance.sum().detach_().cpu().item())
        sem.append(sem_relevance.sum().detach_().cpu().item())
        logitsArr.append(max_logits.sum().detach_().cpu().item())



    save_file = f'{args.work_dir}/data.json'
    update_json(save_file, {'logits':logitsArr,'relevance':sem, 'relevance_plus_pos': sem_with_pos, 'samples': max_count, 'num_correct':num_correct})
    


if __name__ == "__main__":
    args          = parse_args()
    if args.mode == "generate":
       
        generate(args)
   
    elif args.mode == "generate_analyze":
        generate(args)
        analyze(args)


    else:
        analyze(args)