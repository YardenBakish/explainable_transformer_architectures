import os
os.environ['TRANSFORMERS_CACHE'] = '/home/ai_center/ai_users/yardenbakish/'
os.environ['HF_HOME'] = '/home/ai_center/ai_users/yardenbakish/'
import config

ANSWERS  = {'A': 0, 'B': 1, 'C': 2,'D': 3, '▁A': 0, '▁B': 1, '▁C': 2,'▁D': 3, 'ĠA':0,'ĠB':1,'ĠC':2,'ĠD':3}

import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from llama_engine import run_LRP

from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens
from attDatasets.ai2_arc import load_ai2_arc, AI2_ARC_Dataset, create_data_loader
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny', 'llama_3_8b'],
                        default = 'llama_tiny',
                       # required = True,
                        help='')

parser.add_argument('--pe', action='store_true')
parser.add_argument('--pe_only', action='store_true')
parser.add_argument('--clamp', action='store_true')
parser.add_argument('--debug', action='store_true')

parser.add_argument('--reform', action='store_true')
parser.add_argument('--dataset', type=str,
                           default="arc")
parser.add_argument('--variant', type=str,
                           default="baseline")
parser.add_argument('--without-abs', action='store_true')
parser.add_argument('--single-norm', action='store_true')
parser.add_argument('--quant', action='store_true')

parser.add_argument('--sep_heads', action='store_true')


parser.add_argument('--sequence-length', type=int,
                           )


args = parser.parse_args()
config.get_config(args, pert=True)

#path = "meta-llama/Llama-3.1-8B-Instruct"
#
#if args.model_size == 'llama_tiny':
#    path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#if args.model_size == 'llama_2_7b':
#    path = "meta-llama/Llama-2-7b-hf"



#path = "meta-llama/Llama-2-7b-hf"
#path = "llama_tiny"
# optional 4bit quantization to reduce memory footprint
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

if args.quant:
    model = LlamaForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch.bfloat16,quantization_config=quantization_config, device_map="cuda",  low_cpu_mem_usage = True)
#model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
else:
    model = LlamaForCausalLM.from_pretrained(args.model_checkpoint,torch_dtype=torch.bfloat16,  device_map="cuda")



tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()

# apply AttnLRP rules
attnlrp.register(model)
MAX_LEN = 512
BATCH_SIZE = 1
df = load_ai2_arc()
test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)


run_LRP(
    
    
    model,
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
     
        #experimental = args.experimental,
        skip_if_wrong = False,
        sample_num=30,

        mapper_from_token_to_target= ANSWERS,
        #reverse_default_abs = args.reverse_default_abs,
        #save_dir = args.save_dir,
        #should_keep = args.should_keep,
        dataset="arc",
        vis_mode = True,

    )


'''
count = 0
for d in tqdm(test_data_loader):
    if count !=17:
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
    
        position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for i in range(model.config.num_hidden_layers)]
        position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]

        output_logits = model(
            inputs_embeds = input_embeds.requires_grad_(),
            position_embeddings = position_embeddings,
            #input_ids=input_ids,
            use_cache=False,
            #attention_mask=attention_mask
          ).logits

    
    else:
        output_logits = model(inputs_embeds=input_embeds.requires_grad_(), 
                          use_cache=False).logits
    


    max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

    next_token_id = max_indices.item() 
    next_token = tokenizer.convert_ids_to_tokens(next_token_id) 
    print(targets.item())
    print(next_token)
    print(targets.item() ==  ANSWERS[next_token])
    
    max_logits.backward(max_logits)



    relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
    if args.single_norm:
        pass
    else:
        relevance = relevance / relevance.abs().max()


    if args.pe:
        acc_relevancy = 0.
        for pos_embed in position_embeddings:
            for i in range(2):
                #print(pos_embed[i].grad.float().dtype)
                #print(pos_embed[i].dtype)
                if args.reform:
                    if args.without_abs:
                        curr_relevancy = torch.matmul(pos_embed[i].grad, pos_embed[i].transpose(-1, -2)).detach()
                    else:
                        curr_relevancy = torch.matmul(pos_embed[i].grad.abs(), pos_embed[i].transpose(-1, -2).abs()).detach()
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
            relevance+=acc_relevancy
        relevance = relevance / relevance.abs().max()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')
    exit(1)'''