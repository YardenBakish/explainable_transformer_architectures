import os
os.environ['TRANSFORMERS_CACHE'] = '/home/ai_center/ai_users/yardenbakish/'
os.environ['HF_HOME'] = '/home/ai_center/ai_users/yardenbakish/'



import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import matplotlib.pyplot as plt
from lxt.models.mixtral_PE import MixtralForCausalLM, attnlrp
from helper_scripts.helper_functions import update_json
from attDatasets.squad import load_squad, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens
from utils import flip_squad, get_latest_checkpoint
from sklearn.metrics import auc


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
import os
import json

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation')

    parser.add_argument('--mode', type=str,
                                choices=['pert', 'analyze'],
                                required = True,
                                help='')
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--reverse_default_abs', action='store_true')
   


    parser.add_argument('--pe', action='store_true')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--reform', action='store_true')
    parser.add_argument('--should-keep', action='store_true')
    parser.add_argument('--without-abs', action='store_true')
    parser.add_argument('--single-norm', action='store_true')
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--correct-subset', action='store_true')


    parser.add_argument('--sep_heads', action='store_true')

    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--eval', action='store_true')

    
    parser.add_argument('--sequence-length', type=int,
                           )

    parser.add_argument('--variant', type=str,
                           default="baseline")
    parser.add_argument('--pe_only', action='store_true')


    args = parser.parse_args()
    args.model_size = 'mixtral'
    args.dataset = 'squad'

    if (args.pe_only and not args.pe) or (args.reform and not args.pe):
        print("no")
        exit(1)

    config.get_config(args, pert=True)

    if args.debug:
        args.pe = True
        args.reform = True
        args.single_norm = True
        args.without_abs = True
        args.clamp = True
    model_dir = f"pert_results_squad_total" if args.should_keep == False else f"pert_results_squad_partial"
    model_dir = model_dir if args.reverse_default_abs == False else f"reverse_squad/{model_dir}"
    model_dir = model_dir if args.correct_subset == False else f"squad_correct_subset/{model_dir}"


    model_dir = f"{model_dir}/abs" if args.without_abs == False else f"{model_dir}/no_abs"


    args.save_dir = f'finetuned_models/{model_dir}/{args.ext}'


    if args.debug:
        args.save_dir =f'finetuned_models/{model_dir}/debug/{args.ext}'
    
    if args.eval:
        args.save_dir = f'finetuned_models/eval/squad/{args.ext}'
       
    os.makedirs(args.save_dir, exist_ok=True)
    return args


#path = "meta-llama/Llama-3.1-8B-Instruct"
#
#if args.model_size == 'llama_tiny':
#    path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#if args.model_size == 'llama_2_7b':
#    path = f"original_models/{args.model_size}/vanilla" 


def analyze(args):
    root_dir = 'finetuned_models'
    model_dir = f"pert_results_squad_total" if args.should_keep == False else f"pert_results_squad_partial"

    model_dirs = [f'{model_dir}/no_abs', f'{model_dir}/abs']
    dirs = [f'{root_dir}/{model_dir}/{args.ext}' for model_dir in model_dirs]

    res = []
    for dir in dirs:
        if os.path.isdir(dir):
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith('.json'):
                        path = os.path.join(dir, file)
                        print(os.path.join(dir, file))
                        name = "baseline"

                        if "peOnly" in path:
                            name = "PE Only"
                        elif "_pe.json" in path:
                            name = "AttnLRP+PE"
                        if "abs" in path and "no_abs" not in path:
                            name = f"{name} (abs)"
                        if "experimental" in path:
                            name = f"{name}+experimental"
                        if "clamp" in path:
                            name = f"{name}+clamp"
                        if "Reform" in path:
                            name = f"{name}+reform"
                        with open(path, 'r') as file:
                            data = json.load(file)
                        curr = [name,data["generate_AU_AC"], data["generate_AU_MSE"], data["pruning_AU_AC"], data["pruning_AU_MSE"], data["LERF"], data["MERF"] ]
                        res.append(curr)
        else:
            print(f"{dir} is not a directory.")
            exit(1)
   
    
    latex_code = r'\begin{table}[h!]\centering' + '\n' + r'\begin{tabular}{c c c c c |c c c}' + '\n' 
    latex_code += r'\hline & \multicolumn{2}{c}{Generation} & \multicolumn{2}{c|}{Pruning}' r'\\ ' +'\n'
    
    latex_code += r'& AUAC $\uparrow$ & AU-MSE $\downarrow$ & AUAC $\uparrow$ & AU-MSE $\downarrow$ & LERF $\downarrow$ & MERF $\uparrow$' r'\\ ' +'\hline \n'
    res.sort( key=lambda x: x[0])

    for elem in res:
      row = elem[0]
      for cat in elem[1:]:
        row += f' & {cat:.3f}'
      row += r'\\ ' f'\n'
      latex_code += row
      
    latex_code += "\\hline\n\\end{tabular}\n\\caption{Segmentation Results using}\n\\end{table}"

    print(latex_code)









def run_LRP(model, data_loader, 
            tokenizer, 
            withPE=False, 
            reform=False, 
            pe_only=False, 
            withoutABS= False, 
            clamp=False, 
            sep_heads=False, 
            single_norm=False, 
            vis_mode=False, 
            sample_num=None,
            rule_matmul         = False,
  
            save_dir = None,
            eval = False,
            ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    count = 0

    for d in tqdm(data_loader):
        total_samples_num+=1
        if vis_mode and count !=sample_num:
            count+=1
            continue
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        input_embeds = model.get_input_embeddings()(input_ids)
        if withPE:
            position_ids = torch.arange(
                    0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
                   dtype=torch.float32
                ).reshape(1, -1)
            if sep_heads:
                position_embeddings = []
                for i in range(model.config.num_hidden_layers):
                    pe_tuple = model.get_input_pos_embeddings()(input_embeds, position_ids)
                    pe1_a = pe_tuple[0].repeat(model.config.num_attention_heads, 1, 1).requires_grad_()
                    pe1_b = pe_tuple[0].repeat(model.config.num_key_value_heads, 1, 1).requires_grad_()
                    pe2_a = pe_tuple[1].repeat(model.config.num_attention_heads, 1, 1).requires_grad_()
                    pe2_b = pe_tuple[1].repeat(model.config.num_key_value_heads, 1, 1).requires_grad_()
                    position_embeddings.append((pe1_a,pe1_b,pe2_a,pe2_b))
                #position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids)[0] for i in range(model.config.num_hidden_layers)]
            else:
                position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for i in range(model.config.num_hidden_layers)]
                position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]
                identity_matrices   = None
                if rule_matmul:
                    identity_matrices = model.identity_matrices
                    identity_matrices = [x.requires_grad_() for x in identity_matrices] 
            outputs = model(
                inputs_embeds = input_embeds.requires_grad_(),
                position_embeddings = position_embeddings,
                sep_heads = sep_heads,
                identity_matrices = identity_matrices,
                #input_ids=input_ids,
                use_cache=False,
                attention_mask= None
              )['logits']
        else:
            outputs = model(
                inputs_embeds = input_embeds.requires_grad_(),
                use_cache=False,
                attention_mask=None
              )['logits']


        exit(1)
        max_logits, max_indices = torch.max(outputs[0, -1, :], dim=-1)
        next_token_id = max_indices.item() 
       
        next_token = tokenizer.convert_ids_to_tokens(next_token_id) 
        
        max_logits.backward(max_logits)
        relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
        if single_norm:
            pass
        else:
            relevance = relevance / relevance.abs().max()
        if withPE:
            acc_relevancy = 0.
            num_pos_embed_layer = 4 if sep_heads else 2
            for t, pos_embed in enumerate(position_embeddings):
                if rule_matmul:
                    acc_relevancy+=(identity_matrices[t].grad.float().diagonal().cpu() ) # / model.config.num_hidden_layers
                    #print(acc_relevancy.shape)
                    continue
                for i in range(num_pos_embed_layer):
                    if reform:
                        if withoutABS:
                            if clamp:
                                curr_relevancy = torch.matmul(pos_embed[i].grad.clamp(min=0), pos_embed[i].transpose(-1, -2).clamp(min=0)).detach()
                            else:
                                curr_relevancy = torch.matmul(pos_embed[i].grad, pos_embed[i].transpose(-1, -2)).detach()
                        else:
                            curr_relevancy = torch.matmul(pos_embed[i].grad.abs(), pos_embed[i].transpose(-1, -2).abs()).detach()
                        #curr_relevancy /= (2*curr_relevancy.shape[-1])
                        curr_relevancy = curr_relevancy.float().sum(-1).cpu()[0]
                    else:
                        curr_relevancy = pos_embed[i].grad.float().sum(-1).cpu()[0]
                    #curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
                    if experimental:
                        curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
                    if single_norm:
                        pass
                    else:
                        curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
                    if withoutABS:
                        pass
                    else:
                        curr_relevancy = curr_relevancy.abs()
                    acc_relevancy += curr_relevancy
            if clamp:
                acc_relevancy = acc_relevancy.clamp(min=0)
            if single_norm == False:
                acc_relevancy = acc_relevancy / acc_relevancy.abs().max()
            if pe_only:
                relevance=acc_relevancy.detach()
            else:
                #if reform:
                #    acc_relevancy = acc_relevancy / model.config.num_hidden_layers
                relevance+=acc_relevancy
            relevance = relevance / relevance.abs().max()
        if vis_mode==True:
            next_token = tokenizer.convert_ids_to_tokens(next_token_id) 
            print(next_token)
            print(targets.item())
            print("------------------")
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            tokens = clean_tokens(tokens)
            print("reached")
            pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')   
            exit(1)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        if dataset == "arc":
            m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip_arc(model,
                x=relevance, 
                highest_idx = next_token_id,
                logit0 = max_logits.detach().item(),
                token_ids=input_ids, 
                tokens=tokens,
                y_true=targets, 
                should_keep = should_keep,
                attention_mask = attention_mask,
                fracs=fracs, 
                flip_case=flip_case,
                tokenizer=tokenizer,
                reverse_default_abs = reverse_default_abs,
                correct_subset = correct_subset,
                device=device,)
        
        elif dataset == "imdb":
            m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip(model,
                                  x=relevance, 
                                 token_ids=input_ids, 
                                 tokens=tokens,
                                 y_true=targets, 
                                 attention_mask = attention_mask,
                                 fracs=fracs, 
                                 flip_case=flip_case,
                                 tokenizer=tokenizer,
                                 reverse_default_abs = reverse_default_abs,
                                 device=device)
        elif dataset == "wiki":
            m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip_wiki(model,
                x=relevance, 
                highest_idx = next_token_id,
                logit0 = max_logits.detach().item(),
                token_ids=input_ids, 
                tokens=tokens,
                y_true=targets, 
                attention_mask = attention_mask,
                fracs=fracs, 
                flip_case=flip_case,
                tokenizer=tokenizer,
                reverse_default_abs = reverse_default_abs,
                device=device,)
            
        M.append(m)
        E.append(e)

        E_LOGIT.append(e_logit)
        if less_relevantArr != []:
            INSERT_LEAST_IMPORTANT.append(less_relevantArr)
            INSERT_LEAST_IMPORTANT_LOGIT.append(less_relevantArr_logit)
        EVOLUTION.append(evolution)
        M_str.append([str(elem) for elem in m])
        E_str.append([str(elem) for elem in e])
        EVOLUTION_str.append(evolution)
        all_flips["res"]= {'E':E, 'M':M, 'E_LOGIT': E_LOGIT, 'INSERT_LEAST_IMPORTANT': INSERT_LEAST_IMPORTANT, 'INSERT_LEAST_IMPORTANT_LOGIT': INSERT_LEAST_IMPORTANT_LOGIT,    'Evolution':EVOLUTION} 
        all_flips_str["res"]= {'E':E_str, 'M':M_str} 
    if eval:
        update_json(f'{save_dir}/eval.json', {'num_correct': num_correct, 'samples_num': total_samples_num,})
        exit(1)
    f, axs = plt.subplots(1, 2, figsize=(14, 8))
    for k, v in all_flips.items():
        print(len(v['M']))
        axs[0].plot(np.nanmean(v['M'], axis=0), label=k)
        axs[0].set_title('($y_0$-$y_p$)$^2$')
        axs[1].plot(np.nanmean(v['E'], axis=0), label=k)
        axs[1].set_title('logits$_k$')    
    plt.legend()

    pe = "_pe" if withPE else ""
    if pe_only:
        pe = "_peOnly"
    if reform:
        pe = "_peReform"
    if rule_matmul:
        pe = "_peruleMatmul"
    
    if experimental:
        pe = f"{pe}_experimental"
    if clamp:
        pe = f"{pe}_clamp"
    if sep_heads:
        pe = f"{pe}_sepHeads"
    f.savefig(f'{save_dir}/logits_{flip_case}_{pe}.png' , dpi=300)
    #update_json(f'imdb_{flip_case}.json', all_flips_str)



    if flip_case == "generate":
           update_json(f'{save_dir}/pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)),rule_matmul:rule_matmul, 'pe': withPE, 'pe_only': pe_only, 'reform': reform,  'single_norm': single_norm, 'w.o abs': withoutABS, 'num_correct': num_correct, 'samples_num': total_samples_num,
                                        f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0)), f'MERF': auc(fracs, np.nanmean(v['E_LOGIT'], axis=0)), f'LERF': auc(fracs, np.nanmean(v['INSERT_LEAST_IMPORTANT_LOGIT'], axis=0))})
           continue
    update_json(f'{save_dir}/pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)),
                                        f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0))})
    #pickle.dump(all_flips, open(os.path.join(args.save_dir, 'all_flips_{}_imdb.p'.format(flip_case)), 'wb'))







def eval_pert(args):


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
    )

    #model_dir = f"{args.model_size}" if args.quant == False else f"{args.model_size}_QUANT"
    #model_dir = f"{model_dir}_KEEP" if args.should_keep == True else model_dir
    #
    #save_dir = f'finetuned_models/{model_dir}/pert_results_squad'
    #os.makedirs(save_dir, exist_ok=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


    MAX_LEN = 512
    BATCH_SIZE = 1

    if args.quant:
        model = MixtralForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch.bfloat16,quantization_config=quantization_config, device_map="cuda", low_cpu_mem_usage = True)
    #model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        model = MixtralForCausalLM.from_pretrained(args.model_checkpoint, device_map="cuda", torch_dtype=torch.bfloat16)



    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.unk_token_id == None:
        #tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("<|unkown|>")
        tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("#")
     
 

    #model.to(device)


    model.gradient_checkpointing_enable()
    attnlrp.register(model)

    df = load_squad()
    test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

    run_LRP(model,
        test_data_loader,
        tokenizer,
        isBinary=False,
        withPE = args.pe,
        reform=args.reform,
        pe_only = args.pe_only,
        withoutABS = args.without_abs,
        clamp = args.clamp,
        sep_heads = args.sep_heads,
        single_norm = args.single_norm,
     
        experimental = args.experimental,
        skip_if_wrong = False,
     
        reverse_default_abs = args.reverse_default_abs,
        save_dir = args.save_dir,
        should_keep = args.should_keep,
        dataset="squad",
        eval = args.eval,

        )








if __name__ == "__main__":
    args          = parse_args()
    if args.mode == "pert":
        eval_pert(args)
    else:
        analyze(args)
       